import os
import psutil
if 'SLURM_MEM_PER_NODE' in os.environ:
    os.environ['PYSCF_MAX_MEMORY'] = os.environ['SLURM_MEM_PER_NODE']
else:
    os.environ['PYSCF_MAX_MEMORY'] = str(int(psutil.virtual_memory().available / 1e6))
import argparse
import pickle

import h5py
import numpy as np
import torch
from pyscf.pbc import df, mp

import fft
import fft.isdf
import system_common
import utils
import signal
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


def screen_gdf_tensor(R, eps):
    eps_inv_half = utils.matrix_power(eps, 0.5)
    eps_inv_half = torch.from_numpy(eps_inv_half).to(dtype=R.dtype, device=R.device)
    R = torch.einsum("qyx,kqxab->kqyab", eps_inv_half, R)
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


parser = argparse.ArgumentParser()
parser.add_argument("system")
parser.add_argument("kx", type=int)
parser.add_argument("ky", type=int)
parser.add_argument("kz", type=int)
parser.add_argument("basis")
parser.add_argument("c_isdf", type=int)
parser.add_argument("factor", type=float)
parser.add_argument("-suffix", default=None)
parser.add_argument("--abs", action='store_true')
parser.add_argument("--save", action='store_true')
parser.add_argument("--MP2", action='store_true')
parser.add_argument("--screen", action='store_true')
parser.add_argument("--ov", action='store_true')
args = parser.parse_args()

system = args.system
kmesh = (args.kx, args.ky, args.kz)
klabel = system_common.get_klabel(kmesh)
basis = args.basis
c_isdf = args.c_isdf
factor = args.factor
suffix = args.suffix
use_abs = args.abs
reg0 = 1e-9
use_ov = args.ov
fit_AO = True
screen = args.screen
do_MP2 = args.MP2
data_dir = system_common.get_data_dir(system, basis, suffix=suffix)
dft_pkl = os.path.join(data_dir, f"DFT_{klabel}.pkl")
gdf_chk = os.path.join(data_dir, f"GDF_{klabel}.chk")
screening_path = os.path.join(data_dir, f"screening_eps_ext_{klabel}.npy")

block_tag = 'ov' if use_ov else 'full'
screen_tag = "screen" if screen else "bare"
if args.save:
    chkfile = os.path.join(data_dir, f"ISDF{block_tag}_{screen_tag}GDF_{klabel}_c{c_isdf}.chk")
else:
    chkfile = 'chk.isdf'

with open(dft_pkl, "rb") as f:
    mf = pickle.load(f)
if isinstance(mf.with_df, df.GDF):
    mf.with_df._cderi = gdf_chk
cell = mf.cell
kpts = cell.make_kpts(kmesh)
kpts_int = np.round(cell.get_scaled_kpts(kpts) * kmesh).astype(int) % kmesh
assert utils.is_k_ordered(kpts_int, kmesh)
cholesky_mesh = get_cholesky_mesh(cell.mesh, fft.isdf.CHOLESKY_MAX_SIZE)

C = np.asarray(mf.mo_coeff)
nkpts, nao, nmo = C.shape
nocc = cell.nelectron // 2
o = slice(0, nocc)
v = slice(nocc, nmo)
Cocc = np.array(C[:, :, o], order="C", copy=True)
Cvir = np.array(C[:, :, v], order="C", copy=True)
nvir = Cvir.shape[2]

print("")
print("cell.ke_cutoff =", cell.ke_cutoff)
print("cell.mesh      =", cell.mesh)
print("cholesky mesh  =", cholesky_mesh)
print("cholesky size  =", int(np.prod(cholesky_mesh)))
print("cholesky max   =", fft.isdf.CHOLESKY_MAX_SIZE)
print("kmesh          =", kmesh)
print("basis          =", basis)
print("nao            =", nao)
print("nocc           =", nocc)
print("nvir           =", nvir)
print("use_ov         =", use_ov)
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
    isdf = fft.ISDF(cell_isdf, kpts, ov=(Cocc, Cvir))
else:
    if fit_AO:
        isdf = fft.ISDF(cell_isdf, kpts)
    else:
        isdf = fft.ISDF(cell_isdf, kpts, ov=(C, C))
isdf.verbose = 10
X_ao = isdf.build_inpv_only(cisdf=c_isdf)

complex_dtype = torch.complex128
real_dtype = torch.float64
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
        X =  torch.from_numpy(X_ao).to(dtype=complex_dtype)
    else:
        print("Transforming GDF tensor AO -> MO ...", flush=True)
        C_t = torch.from_numpy(C).to(dtype=complex_dtype)
        R = torch.einsum("klxcd,kca,ldb->klxab", R_ao, C_t.conj(), C_t)
        X =  torch.from_numpy(X_ao @ C).to(dtype=complex_dtype)
del R_ao

print("Converting GDF tensor k1k2 -> k1q ...", flush=True)
R = k1k2_to_k1q(R, kpts_int, kmesh)

if screen:
    print("Applying screening ...", flush=True)
    eps = np.load(screening_path)
    R = screen_gdf_tensor(R, eps)

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
            if not np.isnan(rel_error):
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
W_np = W.detach().cpu().numpy()
save_isdf(chkfile, X_ao, W_np)
print("Saved ISDF-GDF chk =", chkfile)

print("")
print("df_norm2  = %.16e" % df_norm2.detach().cpu().numpy())
print("error2    = %.16e" % error2.detach().cpu().numpy())
print("rel_error = %.16e" % rel_error)
print("reg       = %.16e" % reg)
print("X norm    = %.16e" % np.linalg.svd(X_ao.reshape(nkpts * X_ao.shape[1], nao), compute_uv=False).max())
print("W norm    = %.16e" % (np.abs(W_np).max()))

if do_MP2:
    mf_isdf = mf.copy()
    mf_isdf.with_df = fft.ISDF(cell, kpts)
    mf_isdf.with_df._isdf = chkfile
    mf_isdf.with_df.build()

    print("")
    print("Computing PySCF MP2 with GDF ...", flush=True)
    mp2_gdf = mp.KMP2(mf)
    mp2_gdf.verbose = 0
    emp2_gdf, _ = mp2_gdf.kernel(with_t2=False)

    print("Computing PySCF MP2 with ISDF ...", flush=True)
    mp2_isdf = mp.KMP2(mf_isdf)
    mp2_isdf.verbose = 0
    emp2_isdf, _ = mp2_isdf.kernel(with_t2=False)

    print("MP2 GDF energy       = %.16e" % emp2_gdf)
    print("MP2 ISDF energy      = %.16e" % emp2_isdf)
    print("MP2 diff             = %.16e" % (emp2_isdf - emp2_gdf))
