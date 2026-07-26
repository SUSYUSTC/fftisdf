import os
import psutil
if 'SLURM_MEM_PER_NODE' in os.environ:
    os.environ['PYSCF_MAX_MEMORY'] = os.environ['SLURM_MEM_PER_NODE']
else:
    os.environ['PYSCF_MAX_MEMORY'] = str(int(psutil.virtual_memory().available / 1e6))
import time
import argparse
import pickle
import signal

import h5py
import numpy as np
import scipy.linalg
import torch
import pyscf.lib
import pyscf.pbc.df
import pyscf.pbc.mp

import fft
import fft.isdf
import libsymm
import system_common
import utils
signal.signal(signal.SIGINT, signal.SIG_DFL)


def get_cholesky_mesh(mesh, max_size):
    mesh = np.asarray(mesh, dtype=int)
    size = int(np.prod(mesh))
    if size <= max_size:
        return mesh.copy()

    c = (max_size / size) ** (1.0 / 3.0)
    mesh1 = np.maximum(np.floor(mesh * c).astype(int), 1)

    while int(np.prod(mesh1)) > max_size:
        i = np.argmax(mesh1 / mesh)
        mesh1[i] -= 1
    return mesh1


def screen_gdf_tensor(R, eps_inv_half, kmesh):
    nkpts = R.shape[0]
    negative = utils.negative_k(torch.arange(eps_inv_half.shape[0], device=R.device), kmesh)
    eps_inv_half = eps_inv_half[negative]
    for i in range(nkpts):
        R[i] = torch.einsum("qyx,qxab->qyab", eps_inv_half, R[i])
    return R


def save_isdf(chkfile, X, W):
    with h5py.File(chkfile, "w") as f:
        f["inpv_kpt"] = X
        f["coul_kpt"] = W


def k1k2_to_k1q(R, kpts_int, kmesh):
    k1_all, k2_all, q_all = utils._get_gdf_layout_indices(kpts_int, kmesh, "k1q")
    k1_all = torch.from_numpy(k1_all.copy()).to(device=R.device)
    k2_all = torch.from_numpy(k2_all.copy()).to(device=R.device)
    return R[k1_all, k2_all]


def project_ov(isdf, ao_kpt):
    cocc, cvir = isdf.ov
    occ_kpt = [pyscf.lib.dot(ao, co) for ao, co in zip(ao_kpt, cocc)]
    vir_kpt = [pyscf.lib.dot(ao, cv) for ao, cv in zip(ao_kpt, cvir)]
    occ_kpt = np.asarray(occ_kpt, dtype=np.complex128)
    vir_kpt = np.asarray(vir_kpt, dtype=np.complex128)
    return occ_kpt, vir_kpt


def compute_residual_diag(metx, ix_sel):
    diag0 = np.diag(metx).copy()
    if len(ix_sel) == 0:
        return diag0

    metx_xI = metx[:, ix_sel]
    metx_II = metx_xI[ix_sel]
    y = scipy.linalg.solve(metx_II, metx_xI.T, assume_a="sym")
    diag_fit = np.einsum("xi,ix->x", metx_xI, y, optimize=True)
    diag_res = diag0 - diag_fit
    return np.maximum(diag_res, 0.0)


def select_interpolating_points_manual(isdf, cisdf=None, use_symmetry=False):
    cell = isdf.cell
    nao = cell.nao_nr()
    ov = isdf.ov
    grids = isdf.grids
    coord = grids.coords
    ngrid = coord.shape[0]
    kpts = isdf.kpts

    phi_kpt = cell.pbc_eval_gto("GTOval", coord, kpts=kpts)
    phi_kpt = np.asarray(phi_kpt, dtype=np.complex128)
    if ov is None:
        metx = fft.isdf.compute_metx(phi_kpt)
    else:
        occ_kpt, vir_kpt = project_ov(isdf, phi_kpt)
        metx = fft.isdf.compute_metx_pair(occ_kpt, vir_kpt)

    if use_symmetry:
        symm_info = libsymm.classify_grid_by_symmetry(cell, cell.mesh, coords=coord)
        groups = symm_info["orbits"]
        orbit_id = symm_info["orbit_id"]
    else:
        groups = [np.array([i], dtype=np.int64) for i in range(ngrid)]
        orbit_id = np.arange(ngrid, dtype=np.int64)

    nip_target = ngrid
    if cisdf is not None:
        nip_target = min(nip_target, int(cisdf * nao))

    ix_sel = []
    group_sel = []
    group_active = np.ones(len(groups), dtype=bool)
    while len(ix_sel) < nip_target:
        diag_res = compute_residual_diag(metx, ix_sel)
        diag_res_grouped = diag_res.copy()
        diag_res_grouped[~group_active[orbit_id]] = -1.0
        ip = int(np.argmax(diag_res_grouped))
        if diag_res_grouped[ip] <= fft.isdf.CHOLESKY_TOL:
            break

        iorb = int(orbit_id[ip])
        group = groups[iorb]
        ix_sel.extend(int(i) for i in group)
        group_sel.append(iorb)
        group_active[iorb] = False

    ix_sel = np.asarray(ix_sel, dtype=np.int64)
    group_sel = np.asarray(group_sel, dtype=np.int64)
    coord_sel = coord[ix_sel]
    return coord_sel, ix_sel, group_sel


parser = argparse.ArgumentParser()
parser.add_argument("system")
parser.add_argument("kx", type=int)
parser.add_argument("ky", type=int)
parser.add_argument("kz", type=int)
parser.add_argument("basis")
parser.add_argument("c_isdf", type=int)
parser.add_argument("factor", type=float)
parser.add_argument("-suffix", default=None)
parser.add_argument("--abs", action="store_true")
parser.add_argument("--save", action="store_true")
parser.add_argument("--MP2", action="store_true")
parser.add_argument("--screen", action="store_true")
parser.add_argument("--ov", action="store_true")
parser.add_argument("--nosymm", action="store_true")
parser.add_argument("-cholesky_max", type=int, default=None)
args = parser.parse_args()

system = args.system
kmesh = (args.kx, args.ky, args.kz)
klabel = system_common.get_klabel(kmesh)
basis = args.basis
c_isdf = args.c_isdf
factor = args.factor
suffix = args.suffix
use_abs = args.abs
use_ov = args.ov
use_symm = not args.nosymm
screen = args.screen
do_MP2 = args.MP2
fit_AO = True
reg0 = 1e-9

data_dir = system_common.get_data_dir(system, basis, suffix=suffix)
if use_symm:
    dft_pkl = os.path.join(data_dir, f"DFT_{klabel}_symm.pkl")
else:
    dft_pkl = os.path.join(data_dir, f"DFT_{klabel}.pkl")
gdf_chk = os.path.join(data_dir, f"GDF_{klabel}.chk")
screening_path = os.path.join(data_dir, f"screening_eps_ext_{klabel}.npy")

block_tag = "ov" if use_ov else "full"
screen_tag = "screen" if screen else "bare"
symm_tag = "_symm" if use_symm else ""
if args.save:
    chkfile = os.path.join(data_dir, f"ISDF{block_tag}_{screen_tag}GDF{symm_tag}_{klabel}_c{c_isdf}.chk")
else:
    chkfile = "chk.isdf"

with open(dft_pkl, "rb") as f:
    mf = pickle.load(f)
if isinstance(mf.with_df, pyscf.pbc.df.GDF):
    mf.with_df._cderi = gdf_chk

cell = mf.cell
kpts = cell.make_kpts(kmesh)
kpts_int = np.round(cell.get_scaled_kpts(kpts) * kmesh).astype(int) % kmesh
assert utils.is_k_ordered(kpts_int, kmesh)

cholesky_max = fft.isdf.CHOLESKY_MAX_SIZE if args.cholesky_max is None else args.cholesky_max
fft.isdf.CHOLESKY_MAX_SIZE = cholesky_max
cholesky_mesh = get_cholesky_mesh(cell.mesh, cholesky_max)
f4 = lambda n: (n - 1) // 4 * 4 + 4
cholesky_mesh = tuple([f4(m) for m in cholesky_mesh])  # make sure the mesh is a multiple of 4

C = np.asarray(mf.mo_coeff)
nkpts, nao, nmo = C.shape
nocc = cell.nelectron // 2
Cocc = np.array(C[:, :, :nocc], order="C", copy=True)
Cvir = np.array(C[:, :, nocc:], order="C", copy=True)

print("")
print("cell.mesh      =", cell.mesh)
print("cholesky mesh  =", cholesky_mesh)
print("cholesky size  =", int(np.prod(cholesky_mesh)))
print("cholesky max   =", cholesky_max)
print("kmesh          =", kmesh)
print("basis          =", basis)
print("dft pkl        =", dft_pkl)
print("nao            =", nao)
print("nocc           =", nocc)
print("nvir           =", Cvir.shape[2])
print("use_ov         =", use_ov)
print("use_symm       =", use_symm)
print("fit_AO         =", fit_AO)
print("screen         =", screen)
print("do_MP2         =", do_MP2)
print("c_isdf         =", c_isdf)
print("reg0           =", reg0)
print("save chk       =", chkfile)
print("")

cell_isdf = cell.copy()
cell_isdf.mesh = cholesky_mesh
if use_ov:
    isdf_new = fft.ISDF(cell_isdf, kpts, ov=(Cocc, Cvir))
else:
    if fit_AO:
        isdf_new = fft.ISDF(cell_isdf, kpts)
    else:
        isdf_new = fft.ISDF(cell_isdf, kpts, ov=(C, C))
isdf_new.verbose = 5

t0 = time.time()
coord_sel, ix_sel, group_sel = select_interpolating_points_manual(isdf_new, cisdf=c_isdf, use_symmetry=use_symm)
coords = np.asarray(isdf_new.grids.coords)
X_ao = cell_isdf.pbc_eval_gto("GTOval", coord_sel, kpts=kpts)
X_ao = np.asarray(X_ao, dtype=np.complex128)
print("custom selection time =", time.time() - t0)
print("X_ao shape            =", X_ao.shape)
print("selected ix shape     =", ix_sel.shape)
print("selected group shape  =", group_sel.shape)
print("first 20 ix           =", ix_sel[:20])

complex_dtype = torch.complex128
real_dtype = torch.float64

if use_symm:
    symm = libsymm.PBCSymmetry(cell_isdf, kmesh, kpts, dtype=complex_dtype)
    perm, phase = libsymm.build_isdf_grid_transform(symm, coords, ix_sel, mesh=cell_isdf.mesh)
    X_ao = torch.from_numpy(X_ao).to(dtype=complex_dtype)
    X_ao_new = libsymm.symmetrize_isdf_X(symm, X_ao, perm, phase)
    X_symm_err = (torch.linalg.norm(X_ao_new - X_ao) / torch.linalg.norm(X_ao)).detach().cpu().numpy()
    X_ao = X_ao_new.detach().cpu().numpy()
    print("")
    print("X symmetry err         = %.16e" % X_symm_err)

Xo = torch.from_numpy(X_ao @ Cocc).to(dtype=complex_dtype)
Xv = torch.from_numpy(X_ao @ Cvir).to(dtype=complex_dtype)

print("")
print("Loading GDF tensor in AO k1k2 layout ...", flush=True)
R_ao = utils.get_gdf_tensor_compact(mf.with_df, kpts_int, kmesh, progressbar=True, layout="k1k2")
R_ao = torch.from_numpy(R_ao).to(dtype=complex_dtype)
if use_ov:
    print("Transforming GDF tensor AO -> OV ...", flush=True)
    Cocc_t = torch.from_numpy(Cocc).to(dtype=complex_dtype)
    Cvir_t = torch.from_numpy(Cvir).to(dtype=complex_dtype)
    R = torch.einsum("klxcd,kca,ldb->klxab", R_ao, Cocc_t.conj(), Cvir_t)
else:
    if fit_AO:
        R = R_ao
        X = torch.from_numpy(X_ao).to(dtype=complex_dtype)
    else:
        print("Transforming GDF tensor AO -> MO ...", flush=True)
        C_t = torch.from_numpy(C).to(dtype=complex_dtype)
        R = torch.einsum("klxcd,kca,ldb->klxab", R_ao, C_t.conj(), C_t)
        X = torch.from_numpy(X_ao @ C).to(dtype=complex_dtype)
del R_ao

print("Converting GDF tensor k1k2 -> k1q ...", flush=True)
R = k1k2_to_k1q(R, kpts_int, kmesh)

if screen:
    print("Applying screening ...", flush=True)
    eps = np.load(screening_path)[:, 1:, 1:].copy()
    eps_inv_half = utils.matrix_power(eps, 0.5)
    eps_inv_half = torch.from_numpy(eps_inv_half).to(dtype=R.dtype, device=R.device)
    R = screen_gdf_tensor(R, eps_inv_half, kmesh)

R_t = R


def solve_w_error2_with_AB(A, B, df_norm2, reg):
    W = utils.torch_lstsq_oinv_PSD(A, B, reg=reg)
    ab = torch.sum(W.conj() * B).real
    bb = ((W.conj().transpose(-1, -2) @ A @ W @ A.conj().transpose(-1, -2)).diagonal(dim1=-1, dim2=-2).sum()).real
    error2 = (df_norm2 + bb - 2.0 * ab).real
    return W, error2


def scan_reg(A, B, df_norm2):
    reg = reg0
    if use_abs:
        rel_error0 = 1
    else:
        rel_error0 = None
    for ireg in range(100):
        W, error2 = solve_w_error2_with_AB(A, B, df_norm2, reg)
        rel_error = torch.sqrt(error2 / df_norm2)
        if rel_error0 is None:
            if not np.isnan(rel_error.detach().cpu().numpy()):
                rel_error0 = rel_error
            else:
                reg *= 2.0
                continue
        print(
            "reg_scan %3d  reg %.16e  rel_error %.16e  ratio %.8e"
            % (ireg, reg, rel_error.detach().cpu().numpy(), (rel_error / rel_error0).detach().cpu().numpy()),
            flush=True,
        )
        if ireg > 0 and rel_error >= factor * rel_error0:
            return reg, W, error2
        reg *= 2.0
    return reg, W, error2


if use_ov:
    df_norm2 = utils.df_inner_from_mo(R_t, kmesh).real
    A = utils.thc_ovvo_build_Lbar(Xo, Xv, Xo, Xv, kmesh)
    B = utils.thc_df_ov_rhs_from_mo(Xo, Xv, R_t, kmesh)
else:
    df_norm2 = utils.df_inner_from_mo(R_t, kmesh).real
    A = utils.thc_build_Lbar(X, X, kmesh)
    B = utils.thc_df_ov_rhs_from_mo(X, X, R_t, kmesh)

reg, W, error2 = scan_reg(A, B, df_norm2.to(dtype=real_dtype))

rel_error = torch.sqrt(error2 / df_norm2).detach().cpu().numpy()
if use_symm:
    W_new = libsymm.symmetrize_isdf_W(symm, W, perm, phase)
    W_symm_err = (torch.linalg.norm(W_new - W) / torch.linalg.norm(W)).detach().cpu().numpy()
    W = W_new
    ab_sym = torch.sum(W.conj() * B).real
    bb_sym = ((W.conj().transpose(-1, -2) @ A @ W @ A.conj().transpose(-1, -2)).diagonal(dim1=-1, dim2=-2).sum()).real
    error2_sym = (df_norm2 + bb_sym - 2.0 * ab_sym).real
    rel_error_sym = torch.sqrt(error2_sym / df_norm2).detach().cpu().numpy()
else:
    W_symm_err = 0.0
    error2_sym = error2
    rel_error_sym = rel_error
save_isdf(chkfile, X_ao, W.detach().cpu().numpy())
print("Saved ISDF-GDF chk =", chkfile)

print("")
print("df_norm2  = %.16e" % df_norm2.detach().cpu().numpy())
print("error2    = %.16e" % error2.detach().cpu().numpy())
print("rel_error = %.16e" % rel_error)
if use_symm:
    print("W symmetry err         = %.16e" % W_symm_err)
    print("error2 symmetrized W   = %.16e" % error2_sym.detach().cpu().numpy())
    print("rel_error symm W       = %.16e" % rel_error_sym)
print("reg       = %.16e" % reg)
print("X norm    = %.16e" % np.linalg.svd(X_ao.reshape(nkpts * X_ao.shape[1], nao), compute_uv=False).max())
print("W norm    = %.16e" % torch.abs(W).max().detach().cpu().numpy())

if do_MP2:
    mf_isdf = mf.copy()
    mf_isdf.with_df = fft.ISDF(cell, kpts)
    mf_isdf.with_df._isdf = chkfile
    mf_isdf.with_df.build()

    print("")
    print("Computing PySCF MP2 with GDF ...", flush=True)
    mp2_gdf = pyscf.pbc.mp.KMP2(mf)
    mp2_gdf.verbose = 0
    emp2_gdf, _ = mp2_gdf.kernel(with_t2=False)

    print("Computing PySCF MP2 with ISDF ...", flush=True)
    mp2_isdf = pyscf.pbc.mp.KMP2(mf_isdf)
    mp2_isdf.verbose = 0
    emp2_isdf, _ = mp2_isdf.kernel(with_t2=False)

    print("MP2 GDF energy       = %.16e" % emp2_gdf)
    print("MP2 ISDF energy      = %.16e" % emp2_isdf)
    print("MP2 diff             = %.16e" % (emp2_isdf - emp2_gdf))
