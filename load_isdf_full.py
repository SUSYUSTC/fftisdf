import argparse
import os
import pickle
import types

import numpy as np
import torch

import optimize_X_common
import system_common
import utils


def fmt_label(x):
    return "%g" % x


def keep_q0(X, W):
    W_q0 = np.zeros_like(W)
    W_q0[0] = W[0]
    return types.SimpleNamespace(inpv_kpt=X, coul_kpt=W_q0)


parser = argparse.ArgumentParser()
parser.add_argument("system")
parser.add_argument("nk", type=int)
parser.add_argument("basis")
parser.add_argument("c_isdf", type=int)
parser.add_argument("c_ref", type=int)
parser.add_argument("norm_const", type=float)
parser.add_argument("-suffix", default=None)
parser.add_argument("--screen", action="store_true")
parser.add_argument("--symm", action="store_true")
parser.add_argument("--real", action="store_true")
args = parser.parse_args()

system = args.system
nk = args.nk
kmesh = (nk, nk, nk)
klabel = system_common.get_klabel(kmesh)
basis = args.basis
c_isdf = args.c_isdf
c_ref = args.c_ref
norm_label = "norm%s" % fmt_label(args.norm_const)
suffix = args.suffix
screen_tag = "screen" if args.screen else "bare"
symm_tag = "_symm" if args.symm else ""
real_tag = "_real" if args.real else ""

data_dir = system_common.get_data_dir(system, basis, suffix=suffix)
dft_pkl = os.path.join(data_dir, f"DFT_{klabel}{symm_tag}.pkl")
isdf_chk = os.path.join(
    data_dir,
    f"ISDFfull_opt_{screen_tag}GDF{symm_tag}{real_tag}_{klabel}_c{c_isdf}_cref{c_ref}_{norm_label}.chk",
)
ref_chk = os.path.join(
    data_dir,
    f"ISDFfull_{screen_tag}GDF{symm_tag}_{klabel}_c{c_ref}.chk",
)

with open(dft_pkl, "rb") as f:
    mf = pickle.load(f)

C = torch.from_numpy(np.asarray(mf.mo_coeff)).to(dtype=torch.complex128)
nkpts, nao, nmo = C.shape
nocc = mf.cell.nelectron // 2
nvir = nmo - nocc
Cocc = C[:, :, :nocc]
Cvir = C[:, :, nocc:]

X, W = optimize_X_common.load_isdf(isdf_chk)
X_ref, W_ref = optimize_X_common.load_isdf(ref_chk)
isdf = types.SimpleNamespace(inpv_kpt=X, coul_kpt=W)
isdf_ref = types.SimpleNamespace(inpv_kpt=X_ref, coul_kpt=W_ref)

X_t = torch.from_numpy(X).to(dtype=torch.complex128)
W_t = torch.from_numpy(W).to(dtype=torch.complex128)
Xmo = X_t @ C
Xo = X_t @ Cocc
Xv = X_t @ Cvir
norm_Xmo = torch.linalg.svdvals(Xmo).max()
norm_Xo = torch.linalg.svdvals(Xo).max()
norm_Xv = torch.linalg.svdvals(Xv).max()
W_R = utils.fourier_transform_3d(W_t, axis=0, kmesh=kmesh, inverse=False) / np.sqrt(nkpts).item()
norm_W = torch.abs(W_R).max()

full_norm = norm_Xmo**4 * norm_W
oooo_norm = norm_Xo**4 * norm_W
oovv_norm = norm_Xo**2 * norm_Xv**2 * norm_W
vvvv_norm = norm_Xv**4 * norm_W

rel_error = utils.compare_two_isdf(isdf_ref, isdf, kmesh)
rel_error_q0 = utils.compare_two_isdf(
    keep_q0(X_ref, W_ref), keep_q0(X, W), kmesh,
)

e = np.asarray(mf.mo_energy)
eocc = e[:, :nocc]
evir = e[:, nocc:]

print("isdf_chk =", isdf_chk)
print("ref_chk =", ref_chk)
print("dft_pkl =", dft_pkl)
print("nkpts =", nkpts)
print("nao =", nao)
print("nocc =", nocc)
print("nvir =", nvir)
print("Xmo_2norm = %.16e" % norm_Xmo.item())
print("Xo_2norm = %.16e" % norm_Xo.item())
print("Xv_2norm = %.16e" % norm_Xv.item())
print("W_R_max = %.16e" % norm_W.item())
print("W_q_max = %.16e" % torch.abs(W_t).max().item())
print("full_norm = %.16e" % full_norm.item())
print("oooo_norm = %.16e" % oooo_norm.item())
print("oovv_norm = %.16e" % oovv_norm.item())
print("vvvv_norm = %.16e" % vvvv_norm.item())
print("eocc min = %.16e" % eocc.min())
print("eocc max = %.16e" % eocc.max())
print("evir min = %.16e" % evir.min())
print("evir max = %.16e" % evir.max())
print("eov min = %.16e" % (evir.min() - eocc.max()))
print("eov max = %.16e" % (evir.max() - eocc.min()))
print("AO rel_error = %.16e" % rel_error)
print("AO q0 rel_error = %.16e" % rel_error_q0)
