import os
import pickle
import argparse

import h5py
import numpy as np
import torch

import libsymm
import system_common


def build_selected_grid_operator(symm, perm, phase, conjugate=True):
    # Representation on the compact translation-projected selected-grid basis
    # |k,I>.  For operation g,
    #
    #     |k,I> -> phase[g,k,I]^* |gk,gI>
    #
    # This is a monomial matrix: one nonzero entry per column.  Store only the
    # target row and phase instead of materializing the sparse matrix.
    nops = symm.nops
    nkpts = symm.kmap.shape[0]
    nI = perm.shape[1]
    n = nkpts * nI

    row = torch.empty((nops, n), dtype=torch.long, device=symm.device)
    val = torch.empty((nops, n), dtype=symm.dtype, device=symm.device)
    for iop in range(nops):
        for k in range(nkpts):
            kg = int(symm.kmap[k, iop])
            phase_k = phase[iop, k].conj() if conjugate else phase[iop, k]
            col = k * nI + torch.arange(nI, device=symm.device)
            row[iop, col] = kg * nI + perm[iop]
            val[iop, col] = phase_k
    return row, val


def build_ao_representation(symm):
    # Representation on the compact translation-projected AO basis |k,a>.
    # With this convention, X transforms as
    #
    #     X -> D_grid X D_ao^\dagger
    #
    # matching libsymm.transform_isdf_X.
    nops = symm.nops
    nkpts = symm.kmap.shape[0]
    nao = symm.U.shape[-1]
    n = nkpts * nao

    D = torch.zeros((nops, n, n), dtype=symm.dtype, device=symm.device)
    for iop in range(nops):
        for k in range(nkpts):
            kg = int(symm.kmap[k, iop])
            row = kg * nao + torch.arange(nao, device=symm.device)[:, None]
            col = k * nao + torch.arange(nao, device=symm.device)[None, :]
            D[iop, row, col] = symm.U[iop, k]
    return D


def translation_project_matrix(A, nkpts, nI):
    A = A.reshape(nkpts, nI, nkpts, nI)
    out = torch.zeros_like(A)
    idx = torch.arange(nkpts, device=A.device)
    out[idx, :, idx, :] = A[idx, :, idx, :]
    return out.reshape(nkpts * nI, nkpts * nI)


def symmetrize_matrix(D, A):
    A_avg = torch.zeros_like(A)
    for iop in range(D.shape[0]):
        A_avg += D[iop] @ A @ D[iop].conj().T
    return A_avg / D.shape[0]


def symmetrize_matrix_monomial(row, val, A):
    A_avg = torch.zeros_like(A)
    for iop in range(row.shape[0]):
        A_avg += transform_matrix_monomial(row, val, A, iop)
    return A_avg / row.shape[0]


def transform_matrix_monomial(row, val, A, iop):
    r = row[iop]
    v = val[iop]
    A_g = v[:, None] * A * v.conj()[None, :]
    out = torch.zeros_like(A)
    out[r[:, None], r[None, :]] = A_g
    return out


def group_degenerate_eigenvalues(e, tol):
    groups = []
    i0 = 0
    while i0 < len(e):
        i1 = i0 + 1
        scale = max(abs(float(e[i0])), 1.0)
        while i1 < len(e) and abs(float(e[i1] - e[i0])) <= tol * scale:
            i1 += 1
        groups.append((i0, i1))
        i0 = i1
    return groups


def character_vectors(D, Q, groups):
    chars = []
    for i0, i1 in groups:
        Qa = Q[:, i0:i1]
        chi = torch.einsum("ia,oij,ja->o", Qa.conj(), D, Qa)
        chars.append(chi)
    return chars


def character_vectors_monomial(row, val, Q, groups):
    chars = []
    for i0, i1 in groups:
        Qa = Q[:, i0:i1]
        chi = []
        for iop in range(row.shape[0]):
            r = row[iop]
            v = val[iop]
            chi.append(torch.einsum("ia,i,ia->", Qa[r].conj(), v, Qa))
        chars.append(torch.stack(chi))
    return chars


def group_equivalent_irreps(groups, chars, tol):
    classes = []
    used = np.zeros(len(groups), dtype=bool)
    for i, (i0, i1) in enumerate(groups):
        if used[i]:
            continue
        used[i] = True
        cls = [i]
        chi_i = chars[i]
        for j in range(i + 1, len(groups)):
            if used[j]:
                continue
            chi_j = chars[j]
            scale = max(torch.linalg.norm(chi_i).item(), 1.0)
            err = (torch.linalg.norm(chi_i - chi_j) / scale).item()
            if err < tol:
                used[j] = True
                cls.append(j)
        classes.append(cls)
    return classes


def summarize_classes(groups, classes):
    summary = []
    for cls in classes:
        dims = [groups[i][1] - groups[i][0] for i in cls]
        assert len(set(dims)) == 1
        d = dims[0]
        m = len(cls)
        summary.append((d, m, d * m, cls))
    summary.sort(key=lambda x: (x[0], x[1], x[2]))
    return summary


def decompose_representation(D, nkpts, ninner, tol):
    n = nkpts * ninner
    A = torch.randn((n, n), dtype=torch.float64)
    A = A + 1j * torch.randn((n, n), dtype=torch.float64)
    A = (A + A.conj().T) / 2.0

    # Average over the full finite group by first projecting translations, then
    # averaging over point/space operations.  This is equivalent to summing over
    # all t_R p elements, but avoids explicitly constructing translations.
    A = translation_project_matrix(A, nkpts, ninner)
    A_sym = symmetrize_matrix(D, A)
    A_sym = (A_sym + A_sym.conj().T) / 2.0

    comm_err = []
    for iop in range(D.shape[0]):
        err = torch.linalg.norm(D[iop] @ A_sym - A_sym @ D[iop]) / torch.linalg.norm(A_sym)
        comm_err.append(err.item())

    e, Q = torch.linalg.eigh(A_sym)
    groups = group_degenerate_eigenvalues(e.detach().cpu().numpy(), tol)
    chars = character_vectors(D, Q, groups)
    classes = group_equivalent_irreps(groups, chars, tol * 10)
    summary = summarize_classes(groups, classes)
    return summary, chars, max(comm_err)


def decompose_monomial_representation(row, val, nkpts, ninner, tol):
    n = nkpts * ninner
    A = torch.randn((n, n), dtype=torch.float64)
    A = A + 1j * torch.randn((n, n), dtype=torch.float64)
    A = (A + A.conj().T) / 2.0

    A = translation_project_matrix(A, nkpts, ninner)
    A_sym = symmetrize_matrix_monomial(row, val, A)
    A_sym = (A_sym + A_sym.conj().T) / 2.0

    comm_err = []
    for iop in range(row.shape[0]):
        A_g = transform_matrix_monomial(row, val, A_sym, iop)
        err = torch.linalg.norm(A_g - A_sym) / torch.linalg.norm(A_sym)
        comm_err.append(err.item())

    e, Q = torch.linalg.eigh(A_sym)
    groups = group_degenerate_eigenvalues(e.detach().cpu().numpy(), tol)
    chars = character_vectors_monomial(row, val, Q, groups)
    classes = group_equivalent_irreps(groups, chars, tol * 10)
    summary = summarize_classes(groups, classes)
    return summary, chars, max(comm_err)


def character_match_error(chi_a, chi_b):
    scale = max(torch.linalg.norm(chi_a).item(), torch.linalg.norm(chi_b).item(), 1.0)
    return (torch.linalg.norm(chi_a - chi_b) / scale).item()


def print_summary(title, summary):
    print(title)
    print("irrep_dim multiplicity total_size")
    for d, m, size, _cls in summary:
        print("%9d %12d %10d" % (d, m, size))
    print("")


parser = argparse.ArgumentParser()
parser.add_argument("system")
parser.add_argument("kx", type=int)
parser.add_argument("ky", type=int)
parser.add_argument("kz", type=int)
parser.add_argument("basis")
parser.add_argument("c_isdf", type=int)
parser.add_argument("-suffix", default=None)
parser.add_argument("-tol", type=float, default=1e-8)
parser.add_argument("--ov", action="store_true")
parser.add_argument("--screen", action="store_true")
args = parser.parse_args()

system = args.system
kmesh = (args.kx, args.ky, args.kz)
klabel = system_common.get_klabel(kmesh)
basis = args.basis
suffix = args.suffix
c_isdf = args.c_isdf
use_ov = args.ov
screen = args.screen

data_dir = system_common.get_data_dir(system, basis, suffix=suffix)
dft_pkl = os.path.join(data_dir, f"DFT_{klabel}_symm.pkl")
block_tag = "ov" if use_ov else "full"
screen_tag = "screen" if screen else "bare"
chkfile = os.path.join(data_dir, f"ISDF{block_tag}_{screen_tag}GDF_symm_{klabel}_c{c_isdf}.chk")

with open(dft_pkl, "rb") as f:
    mf = pickle.load(f)

with h5py.File(chkfile, "r") as f:
    X = np.asarray(f["inpv_kpt"])
    W = np.asarray(f["coul_kpt"])
    if "mesh" not in f or "ix_sel" not in f:
        raise RuntimeError("ISDF file must contain mesh and ix_sel. Regenerate it with generate_isdf_gdf_symm.py.")
    mesh = np.asarray(f["mesh"], dtype=np.int64)
    ix_sel = np.asarray(f["ix_sel"], dtype=np.int64)
    group_sel = np.asarray(f["group_sel"], dtype=np.int64) if "group_sel" in f else None

cell = mf.cell
kpts = cell.make_kpts(kmesh)
cell_isdf = cell.copy()
cell_isdf.mesh = mesh
coords = cell_isdf.gen_uniform_grids(cell_isdf.mesh)

symm = libsymm.PBCSymmetry(cell_isdf, kmesh, kpts, dtype=torch.complex128)
perm, phase = libsymm.build_isdf_grid_transform(symm, coords, ix_sel, mesh=cell_isdf.mesh)
row_grid, val_grid = build_selected_grid_operator(symm, perm, phase)
D_ao = build_ao_representation(symm)

nkpts = len(kpts)
nI = len(ix_sel)
nao = cell.nao_nr()

X = torch.from_numpy(X).to(dtype=torch.complex128)
W = torch.from_numpy(W).to(dtype=torch.complex128)
X_symm = libsymm.symmetrize_isdf_X_fast(symm, X, perm, phase)
W_symm = libsymm.symmetrize_isdf_W_fast(symm, W, perm, phase)
X_symm_err = torch.linalg.norm(X_symm - X) / torch.linalg.norm(X)
W_symm_err = torch.linalg.norm(W_symm - W) / torch.linalg.norm(W)

grid_summary, grid_chars, grid_comm = decompose_monomial_representation(row_grid, val_grid, nkpts, nI, args.tol)
ao_summary, ao_chars, ao_comm = decompose_representation(D_ao, nkpts, nao, args.tol)

print("system =", system)
print("kmesh =", kmesh)
print("basis =", basis)
print("suffix =", suffix)
print("use_ov =", use_ov)
print("screen =", screen)
print("c_isdf =", c_isdf)
print("chkfile =", chkfile)
print("DFT pkl =", dft_pkl)
print("mesh =", mesh)
print("nkpts =", nkpts)
print("n selected points =", nI)
print("n selected groups =", None if group_sel is None else len(group_sel))
print("nao =", nao)
print("n symmetry operations =", symm.nops)
print("grid representation size =", nkpts * nI)
print("AO representation size =", nkpts * nao)
print("grid max invariance rel_fro =", grid_comm)
print("AO max commutator rel_fro =", ao_comm)
print("loaded X symmetry rel_fro =", X_symm_err.item())
print("loaded W symmetry rel_fro =", W_symm_err.item())
print("")

print_summary("W / selected-grid block structure", grid_summary)
print("W dimension check =", sum(size for _d, _m, size, _cls in grid_summary), "/", nkpts * nI)
print("")

print_summary("AO/orbital block structure", ao_summary)
print("AO dimension check =", sum(size for _d, _m, size, _cls in ao_summary), "/", nkpts * nao)
print("")

print("X allowed blocks: selected-grid irreps against AO irreps")
print("irrep_dim  grid_mult  ao_mult  dense_shape")
for d_g, m_g, _size_g, cls_g in grid_summary:
    chi_g = grid_chars[cls_g[0]]
    for d_a, m_a, _size_a, cls_a in ao_summary:
        if d_g != d_a:
            continue
        chi_a = ao_chars[cls_a[0]]
        if character_match_error(chi_g, chi_a) < args.tol * 10:
            print("%9d %10d %8d  %dx%d" % (d_g, m_g, m_a, m_g, m_a))
