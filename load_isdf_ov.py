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
parser.add_argument("-power", type=int, default=0)
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
norm_const = args.norm_const
norm_power = args.power
norm_label = "norm%sp%s" % (fmt_label(norm_const), fmt_label(norm_power))
suffix = args.suffix
screen_tag = "screen" if args.screen else "bare"
symm_tag = "_symm" if args.symm else ""
real_tag = "_real" if args.real else ""

data_dir = system_common.get_data_dir(system, basis, suffix=suffix)
dft_pkl = os.path.join(data_dir, f"DFT_{klabel}{symm_tag}.pkl")
isdf_chk = os.path.join(
    data_dir,
    f"ISDFov_opt_{screen_tag}GDF{symm_tag}{real_tag}_{klabel}_c{c_isdf}_cref{c_ref}_{norm_label}.chk",
)
ref_chk = os.path.join(
    data_dir,
    f"ISDFov_{screen_tag}GDF{symm_tag}_{klabel}_c{c_ref}.chk",
)

with open(dft_pkl, "rb") as f:
    mf = pickle.load(f)

C = np.asarray(mf.mo_coeff)
nkpts, nao, nmo = C.shape
nocc = mf.cell.nelectron // 2
nvir = nmo - nocc
Cocc = C[:, :, :nocc]
Cvir = C[:, :, nocc:]

X, W = optimize_X_common.load_isdf(isdf_chk)
X_ref, W_ref = optimize_X_common.load_isdf(ref_chk)
isdf = types.SimpleNamespace(inpv_kpt=X, coul_kpt=W)
isdf_ref = types.SimpleNamespace(inpv_kpt=X_ref, coul_kpt=W_ref)

Xo = torch.from_numpy(X @ Cocc)
Xv = torch.from_numpy(X @ Cvir)
W_t = torch.from_numpy(W)
norm_Xo = torch.linalg.svdvals(Xo).max()
norm_Xv = torch.linalg.svdvals(Xv).max()
norm_W = torch.linalg.eigvalsh(W_t).abs().max()
norm_W_by_q = torch.linalg.eigvalsh(W_t).abs().max(dim=1).values
norm_W_F = torch.linalg.norm(W_t, dim=(1, 2)).max()
norm_loss = norm_Xo**2 * norm_Xv**2 * norm_W
norm_loss_F = norm_Xo**2 * norm_Xv**2 * norm_W_F

error, ref_norm = utils.compare_two_isdf(
    isdf_ref, isdf, kmesh, C1=Cocc, C2=Cvir, abs=True,
)
error_q0, ref_norm_q0 = utils.compare_two_isdf(
    keep_q0(X_ref, W_ref), keep_q0(X, W),
    kmesh, C1=Cocc, C2=Cvir, abs=True,
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
print("Xo_2norm = %.16e" % norm_Xo.item())
print("Xv_2norm = %.16e" % norm_Xv.item())
print("W_2norm / nkpts = %.16e" % norm_W.item())
print("W_Fnorm / nkpts = %.16e" % norm_W_F.item())
print("norm_loss = %.16e" % norm_loss.item())
print("norm_loss_F = %.16e" % norm_loss_F.item())
print("eocc min = %.16e" % eocc.min())
print("eocc max = %.16e" % eocc.max())
print("evir min = %.16e" % evir.min())
print("evir max = %.16e" % evir.max())
print("eov min = %.16e" % (evir.min() - eocc.max()))
print("eov max = %.16e" % (evir.max() - eocc.min()))
print("ovvo error = %.16e" % error)
print("ovvo ref norm = %.16e" % ref_norm)
print("ovvo rel_error = %.16e" % (error / ref_norm))
print("ovvo q0 error = %.16e" % error_q0)
print("ovvo q0 ref norm = %.16e" % ref_norm_q0)
print("ovvo q0 rel_error = %.16e" % (error_q0 / ref_norm_q0))
