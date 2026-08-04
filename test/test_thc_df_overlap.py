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
mf.with_df._cderi = os.path.join(data_dir, "GDF.chk")
with h5py.File(os.path.join(data_dir, "THC_ov_ref.chk"), "r") as f:
    X_ao = np.asarray(f["inpv_kpt"])
    W = np.asarray(f["coul_kpt"])

cell = mf.cell
kpts = cell.make_kpts(kmesh)
kpts_int = np.round(cell.get_scaled_kpts(kpts) * kmesh).astype(int) % kmesh
assert utils.is_k_ordered(kpts_int, kmesh)

C = np.asarray(mf.mo_coeff)
nocc = cell.nelectron // 2
Cocc = np.array(C[:, :, :nocc], order="C", copy=True)
Cvir = np.array(C[:, :, nocc:], order="C", copy=True)
X = torch.from_numpy(X_ao @ C)
Xo = torch.from_numpy(X_ao @ Cocc)
Xv = torch.from_numpy(X_ao @ Cvir)
W = torch.from_numpy(W)

assert X_ao.shape == (9, 40, 8)


def make_thc(W):
    df_thc = fft.ISDF(cell, kpts=kpts)
    df_thc._inpv_kpt = X_ao
    df_thc._coul_kpt = W
    return df_thc


def rel_diff(a, b):
    return abs(a - b) / abs(b)


def test_df_df_and_thc_df_full():
    R = utils.get_gdf_tensor_compact(mf.with_df, kpts_int, kmesh, C, C, layout="k1q")
    R = torch.from_numpy(R)
    eri_df = mf.with_df.ao2mo_7d(C)
    eri_thc = make_thc(W.numpy()).ao2mo_7d(C)

    df_norm2_ref = np.vdot(eri_df, eri_df).real
    inner_ref = np.vdot(eri_thc, eri_df)
    error2_ref = np.linalg.norm(eri_thc - eri_df)**2
    df_norm2 = utils.df_inner_from_mo(R, kmesh).real
    inner = utils.thc_df_inner_from_mo(X, W, R, kmesh)
    error2 = utils.thc_df_error2_from_mo(X, W, R, kmesh, df_norm2)

    W_fit, fit_error2 = utils.thc_df_solve_w_error2_from_mo(
        X, R, kmesh, df_norm2,
    )
    eri_fit = make_thc(W_fit.numpy()).ao2mo_7d(C)
    fit_error2_ref = np.linalg.norm(eri_fit - eri_df)**2

    assert rel_diff(df_norm2.item(), df_norm2_ref) < 1e-10
    assert rel_diff(inner.item(), inner_ref) < 1e-10
    assert rel_diff(error2.item(), error2_ref) < 1e-10
    assert rel_diff(fit_error2.item(), fit_error2_ref) < 1e-10


def test_df_df_and_thc_df_ovvo():
    R = utils.get_gdf_tensor_compact(mf.with_df, kpts_int, kmesh, Cocc, Cvir, layout="k1q")
    R = torch.from_numpy(R)
    C_ovvo = [Cocc, Cvir, Cvir, Cocc]
    eri_df = mf.with_df.ao2mo_7d(C_ovvo)
    eri_thc = make_thc(W.numpy()).ao2mo_7d(C_ovvo)

    df_norm2_ref = np.vdot(eri_df, eri_df).real
    inner_ref = np.vdot(eri_thc, eri_df)
    error2_ref = np.linalg.norm(eri_thc - eri_df)**2
    df_norm2 = utils.df_inner_from_mo(R, kmesh).real
    inner = utils.thc_df_ov_inner_from_mo(Xo, Xv, W, R, kmesh)
    error2 = utils.thc_df_ov_error2_from_mo(Xo, Xv, W, R, kmesh, df_norm2)

    W_fit, fit_error2 = utils.thc_df_ov_solve_w_error2_from_mo(
        Xo, Xv, R, kmesh, df_norm2,
    )
    eri_fit = make_thc(W_fit.numpy()).ao2mo_7d(C_ovvo)
    fit_error2_ref = np.linalg.norm(eri_fit - eri_df)**2

    assert rel_diff(df_norm2.item(), df_norm2_ref) < 1e-10
    assert rel_diff(inner.item(), inner_ref) < 1e-10
    assert rel_diff(error2.item(), error2_ref) < 1e-10
    assert rel_diff(fit_error2.item(), fit_error2_ref) < 1e-10
