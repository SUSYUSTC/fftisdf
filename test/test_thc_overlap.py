import os
import pickle

import h5py
import numpy as np
import torch

import fft
import utils


kmesh = (1, 3, 3)
data_dir = os.path.join(os.path.dirname(__file__), "data", "diamond_1x3x3_gth-szv")

with open(os.path.join(data_dir, "DFT.pkl"), "rb") as f:
    mf = pickle.load(f)
with h5py.File(os.path.join(data_dir, "THC_ov_ref.chk"), "r") as f:
    X_ao_ref = np.asarray(f["inpv_kpt"])
    W_ref = np.asarray(f["coul_kpt"])

cell = mf.cell
kpts = cell.make_kpts(kmesh)
C = np.asarray(mf.mo_coeff)
nocc = cell.nelectron // 2
Cocc = C[:, :, :nocc]
Cvir = C[:, :, nocc:]

X_ref = torch.from_numpy(X_ao_ref @ C)
Xo_ref = torch.from_numpy(X_ao_ref @ Cocc)
Xv_ref = torch.from_numpy(X_ao_ref @ Cvir)
W_ref = torch.from_numpy(W_ref)

nI = 24
X_ao = X_ao_ref[:, :nI].copy()
X = torch.from_numpy(X_ao @ C)
Xo = torch.from_numpy(X_ao @ Cocc)
Xv = torch.from_numpy(X_ao @ Cvir)
W = W_ref[:, :nI, :nI].clone()

assert X_ao_ref.shape == (9, 40, 8)
assert W_ref.shape == (9, 40, 40)


def make_thc(X_ao, W):
    df_thc = fft.ISDF(cell, kpts=kpts)
    df_thc._inpv_kpt = X_ao
    df_thc._coul_kpt = W
    return df_thc


def rel_diff(a, b):
    return abs(a - b) / abs(b)


def test_thc_full_inner_error_and_solve_w():
    eri_ref = make_thc(X_ao_ref, W_ref.numpy()).ao2mo_7d(C)
    eri = make_thc(X_ao, W.numpy()).ao2mo_7d(C)

    inner_ref = np.vdot(eri_ref, eri)
    error2_ref = np.linalg.norm(eri_ref - eri)**2
    inner = utils.thc_inner_from_mo(X_ref, W_ref, X, W, kmesh)
    error2 = utils.thc_error2_from_mo(X_ref, W_ref, X, W, kmesh)

    ref_norm2 = torch.linalg.norm(torch.from_numpy(eri_ref))**2
    W_fit, fit_error2 = utils.thc_solve_w_error2_from_mo(
        X_ref, W_ref, X, kmesh, ref_norm2,
    )
    eri_fit = make_thc(X_ao, W_fit.numpy()).ao2mo_7d(C)
    fit_error2_ref = np.linalg.norm(eri_ref - eri_fit)**2

    assert rel_diff(inner.item(), inner_ref) < 1e-10
    assert rel_diff(error2.item(), error2_ref) < 1e-10
    assert rel_diff(fit_error2.item(), fit_error2_ref) < 1e-10


def test_thc_ovvo_inner_error_and_solve_w():
    C_ovvo = [Cocc, Cvir, Cvir, Cocc]
    eri_ref = make_thc(X_ao_ref, W_ref.numpy()).ao2mo_7d(C_ovvo)
    eri = make_thc(X_ao, W.numpy()).ao2mo_7d(C_ovvo)

    inner_ref = np.vdot(eri_ref, eri)
    error2_ref = np.linalg.norm(eri_ref - eri)**2
    inner = utils.thc_ovvo_inner_from_mo(
        Xo_ref, Xv_ref, W_ref, Xo, Xv, W, kmesh,
    )
    error2 = utils.thc_ovvo_error2_from_mo(
        Xo_ref, Xv_ref, W_ref, Xo, Xv, W, kmesh,
    )

    ref_norm2 = torch.linalg.norm(torch.from_numpy(eri_ref))**2
    W_fit, fit_error2 = utils.thc_ovvo_solve_w_error2_from_mo(
        Xo_ref, Xv_ref, W_ref, Xo, Xv, kmesh, ref_norm2,
    )
    eri_fit = make_thc(X_ao, W_fit.numpy()).ao2mo_7d(C_ovvo)
    fit_error2_ref = np.linalg.norm(eri_ref - eri_fit)**2

    W_test = W_fit.detach().clone().requires_grad_(True)
    error2_test = utils.thc_ovvo_error2_from_mo(
        Xo_ref, Xv_ref, W_ref, Xo, Xv, W_test, kmesh,
    )
    error2_test.backward()

    assert rel_diff(inner.item(), inner_ref) < 1e-10
    assert rel_diff(error2.item(), error2_ref) < 1e-10
    assert rel_diff(fit_error2.item(), fit_error2_ref) < 1e-10
    assert torch.linalg.norm(W_test.grad) < 1e-9
