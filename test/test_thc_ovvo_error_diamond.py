import pickle
import signal
import sys
import os

import numpy as np
import torch

import fft
import fft.isdf_ao2mo
import utils

signal.signal(signal.SIGINT, signal.SIG_DFL)

rng = np.random.default_rng(12)
use_gpu = False
device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

basis = "gth-szv"
kmesh = np.array([1, 3, 3])
klabel = f"{kmesh[0]}x{kmesh[1]}x{kmesh[2]}"

test_data_dir = os.path.join(os.path.dirname(__file__), "data_thc_overlap")
scf_pkl = os.path.join(test_data_dir, f"DFT_diamond_{klabel}_{basis}.pkl")
with open(scf_pkl, "rb") as f:
    mf = pickle.load(f)

cell = mf.cell
kpts = cell.make_kpts(kmesh)
kpts_int = np.round(cell.get_scaled_kpts(kpts) * kmesh).astype(int) % kmesh
assert utils.is_k_ordered(kpts_int, kmesh)
nkpts = len(kpts_int)

C = np.asarray(mf.mo_coeff)
nocc = cell.nelectron // 2
o = slice(None, nocc)
v = slice(nocc, None)
Cocc = np.array(C[:, :, o], order="C", copy=True)
Cvir = np.array(C[:, :, v], order="C", copy=True)
nvir = Cvir.shape[2]
C_ovvo = [Cocc, Cvir, Cvir, Cocc]

naux = 5
X_A = rng.normal(size=(nkpts, naux, cell.nao_nr())) + 1j * rng.normal(size=(nkpts, naux, cell.nao_nr()))
W_A = rng.normal(size=(nkpts, naux, naux)) + 1j * rng.normal(size=(nkpts, naux, naux))

X_B = rng.normal(size=(nkpts, naux, cell.nao_nr())) + 1j * rng.normal(size=(nkpts, naux, cell.nao_nr()))
W_B = rng.normal(size=(nkpts, naux, naux)) + 1j * rng.normal(size=(nkpts, naux, naux))


df_A = fft.ISDF(cell, kpts=kpts)
df_A._inpv_kpt = X_A
df_A._coul_kpt = W_A
df_B = fft.ISDF(cell, kpts=kpts)
df_B._inpv_kpt = X_B
df_B._coul_kpt = W_B

eri_A = df_A.ao2mo_7d(C_ovvo)
eri_B = df_B.ao2mo_7d(C_ovvo)
ref_error2 = np.linalg.norm(eri_A - eri_B) ** 2
ref_ab = np.vdot(eri_A, eri_B)

Xo_A_t = torch.from_numpy(X_A @ Cocc).to(device)
Xv_A_t = torch.from_numpy(X_A @ Cvir).to(device)
W_A_t = torch.from_numpy(W_A).to(device)
Xo_B_t = torch.from_numpy(X_B @ Cocc).to(device)
Xv_B_t = torch.from_numpy(X_B @ Cvir).to(device)
W_B_t = torch.from_numpy(W_B).to(device)

print("cell.basis =", cell.basis)
print("kmesh      =", kmesh)
print("nao        =", cell.nao_nr())
print("nocc       =", nocc)
print("nvir       =", nvir)
print("naux       =", naux)
print("")

for use_fast in [False, True]:
    if use_fast:
        utils.enable_fast_fft()
    else:
        utils.disable_fast_fft()

    fast_error2 = utils.thc_ovvo_error2_from_mo(
        Xo_A_t, Xv_A_t, W_A_t,
        Xo_B_t, Xv_B_t, W_B_t,
        kmesh,
    )
    fast_ab = utils.thc_ovvo_inner_from_mo(
        Xo_A_t, Xv_A_t, W_A_t,
        Xo_B_t, Xv_B_t, W_B_t,
        kmesh,
    )

    fast_error2_np = fast_error2.detach().cpu().numpy()
    fast_ab_np = fast_ab.detach().cpu().numpy()
    abs_diff = abs(fast_error2_np - ref_error2)
    rel_diff = abs_diff / abs(ref_error2)
    ab_diff = abs(fast_ab_np - ref_ab)
    ab_reldiff = ab_diff / abs(ref_ab)

    print("use_fast_fft =", use_fast)
    print("ref_error2   = %16.8e" % ref_error2)
    print("fast_error2  = %16.8e" % fast_error2_np)
    print("abs_diff     = %16.8e" % abs_diff)
    print("rel_diff     = %16.8e" % rel_diff)
    print("ab_reldiff   = %16.8e" % ab_reldiff)
    print("")
    assert rel_diff < 1e-8
    assert ab_reldiff < 1e-8


def test_thc_ovvo_error_diamond():
    pass
