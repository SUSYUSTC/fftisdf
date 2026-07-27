import os
import pickle
import argparse

import h5py
import numpy as np
import torch

import libsymm
import system_common


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


def character_vectors(U, Q, groups, kmap, nkpts, ninner, trans_phase):
    chars = []
    kmap_op = kmap.T.contiguous()
    for i0, i1 in groups:
        Qa = Q[:, i0:i1].reshape(nkpts, ninner, i1 - i0)
        Qa_out = Qa[kmap_op]
        B_in = torch.einsum("okpa,okpq,kqa->ok", Qa_out.conj(), U, Qa)
        B = torch.zeros_like(B_in)
        B.scatter_add_(1, kmap_op, B_in)
        chi = torch.einsum("tk,ok->ot", trans_phase, B)
        chars.append(chi.reshape(-1))
    return chars


def build_translation_phase(cell, kpts, kmesh, dtype, device):
    kpts_scaled = cell.get_scaled_kpts(kpts)
    kpts_int = np.rint(kpts_scaled * np.asarray(kmesh)[None, :]).astype(np.int64) % np.asarray(kmesh)[None, :]
    R = np.asarray(np.meshgrid(*[np.arange(n) for n in kmesh], indexing="ij"))
    R = R.reshape(3, -1).T
    kr = np.einsum("kd,td,d->tk", kpts_int, R, 1.0 / np.asarray(kmesh))
    phase = np.exp(2j * np.pi * kr)
    return torch.from_numpy(phase).to(dtype=dtype, device=device)


def character_vectors_monomial(kmap, perm, phase, Q, groups, nkpts, ninner, trans_phase):
    chars = []
    kmap_op = kmap.T.contiguous()
    for i0, i1 in groups:
        Qa = Q[:, i0:i1].reshape(nkpts, ninner, i1 - i0)
        Qa_out = Qa[kmap_op]
        idx = perm[:, None, :, None].expand(perm.shape[0], nkpts, ninner, i1 - i0)
        Qa_row = torch.gather(Qa_out, 2, idx)
        B_in = torch.einsum("okia,oki,kia->ok", Qa_row.conj(), phase, Qa)
        B = torch.zeros_like(B_in)
        B.scatter_add_(1, kmap_op, B_in)
        chi = torch.einsum("tk,ok->ot", trans_phase, B)
        chars.append(chi.reshape(-1))
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


def symmetrize_block_matrix_monomial(kmap, perm, phase, A):
    A_avg = torch.zeros_like(A)
    nops = perm.shape[0]
    nkpts = A.shape[0]
    for iop in range(nops):
        p = perm[iop]
        v = phase[iop]
        A_g = v[:, :, None] * A * v.conj()[:, None, :]
        A_perm = torch.zeros_like(A_g)
        A_perm[:, p[:, None], p[None, :]] = A_g
        A_avg.scatter_add_(0, kmap[:, iop].reshape(nkpts, 1, 1).expand_as(A_perm), A_perm)
    return A_avg / nops


def transform_block_matrix_monomial(kmap, perm, phase, A, iop):
    out = torch.zeros_like(A)
    p = perm[iop]
    v = phase[iop]
    A_g = v[:, :, None] * A * v.conj()[:, None, :]
    A_perm = torch.zeros_like(A_g)
    A_perm[:, p[:, None], p[None, :]] = A_g
    out.scatter_add_(0, kmap[:, iop].reshape(A.shape[0], 1, 1).expand_as(A_perm), A_perm)
    return out


def symmetrize_block_matrix_unitary(kmap, U, A):
    A_avg = torch.zeros_like(A)
    nops = U.shape[0]
    nkpts = A.shape[0]
    for iop in range(nops):
        A_g = torch.einsum("kpa,kab,kqb->kpq", U[iop], A, U[iop].conj())
        A_avg.scatter_add_(0, kmap[:, iop].reshape(nkpts, 1, 1).expand_as(A_g), A_g)
    return A_avg / nops


def transform_block_matrix_unitary(kmap, U, A, iop):
    out = torch.zeros_like(A)
    A_g = torch.einsum("kpa,kab,kqb->kpq", U[iop], A, U[iop].conj())
    out.scatter_add_(0, kmap[:, iop].reshape(A.shape[0], 1, 1).expand_as(A_g), A_g)
    return out


def block_eigh_to_global(e, Qk):
    nkpts, ninner = e.shape
    n = nkpts * ninner
    e_flat = e.reshape(n)
    order = torch.argsort(e_flat)
    e_sort = e_flat[order]

    Q = torch.zeros((n, n), dtype=Qk.dtype, device=Qk.device)
    for col_new, col_old in enumerate(order):
        k = int(col_old // ninner)
        a = int(col_old % ninner)
        Q[k * ninner:(k + 1) * ninner, col_new] = Qk[k, :, a]
    return e_sort, Q


def decompose_unitary_representation(kmap, U, nkpts, ninner, trans_phase, tol):
    A = torch.randn((nkpts, ninner, ninner), dtype=torch.float64)
    A = A + 1j * torch.randn((nkpts, ninner, ninner), dtype=torch.float64)
    A = A.to(dtype=U.dtype, device=U.device)
    A = (A + A.conj().transpose(-1, -2)) / 2.0

    A_sym = symmetrize_block_matrix_unitary(kmap, U, A)
    A_sym = (A_sym + A_sym.conj().transpose(-1, -2)) / 2.0

    comm_err = []
    for iop in range(U.shape[0]):
        A_g = transform_block_matrix_unitary(kmap, U, A_sym, iop)
        err = torch.linalg.norm(A_g - A_sym) / torch.linalg.norm(A_sym)
        comm_err.append(err.item())

    e, Qk = torch.linalg.eigh(A_sym)
    e, Q = block_eigh_to_global(e, Qk)
    groups = group_degenerate_eigenvalues(e.detach().cpu().numpy(), tol)
    chars = character_vectors(U, Q, groups, kmap, nkpts, ninner, trans_phase)
    classes = group_equivalent_irreps(groups, chars, tol * 10)
    summary = summarize_classes(groups, classes)
    return summary, chars, max(comm_err)


def decompose_monomial_representation(kmap, perm, phase, nkpts, ninner, trans_phase, tol):
    A = torch.randn((nkpts, ninner, ninner), dtype=torch.float64)
    A = A + 1j * torch.randn((nkpts, ninner, ninner), dtype=torch.float64)
    A = (A + A.conj().transpose(-1, -2)) / 2.0

    A_sym = symmetrize_block_matrix_monomial(kmap, perm, phase, A)
    A_sym = (A_sym + A_sym.conj().transpose(-1, -2)) / 2.0

    comm_err = []
    for iop in range(perm.shape[0]):
        A_g = transform_block_matrix_monomial(kmap, perm, phase, A_sym, iop)
        err = torch.linalg.norm(A_g - A_sym) / torch.linalg.norm(A_sym)
        comm_err.append(err.item())

    e, Qk = torch.linalg.eigh(A_sym)
    e, Q = block_eigh_to_global(e, Qk)
    groups = group_degenerate_eigenvalues(e.detach().cpu().numpy(), tol)
    chars = character_vectors_monomial(kmap, perm, phase, Q, groups, nkpts, ninner, trans_phase)
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
phase_grid = phase.conj()
trans_phase = build_translation_phase(cell, kpts, kmesh, torch.complex128, symm.device)

nkpts = len(kpts)
nI = len(ix_sel)
nao = cell.nao_nr()

X = torch.from_numpy(X).to(dtype=torch.complex128)
W = torch.from_numpy(W).to(dtype=torch.complex128)
X_symm = libsymm.symmetrize_isdf_X_fast(symm, X, perm, phase)
W_symm = libsymm.symmetrize_isdf_W_fast(symm, W, perm, phase)
X_symm_err = torch.linalg.norm(X_symm - X) / torch.linalg.norm(X)
W_symm_err = torch.linalg.norm(W_symm - W) / torch.linalg.norm(W)

grid_summary, grid_chars, grid_comm = decompose_monomial_representation(
    symm.kmap, perm, phase_grid, nkpts, nI, trans_phase, args.tol
)
ao_summary, ao_chars, ao_comm = decompose_unitary_representation(symm.kmap, symm.U, nkpts, nao, trans_phase, args.tol)

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
