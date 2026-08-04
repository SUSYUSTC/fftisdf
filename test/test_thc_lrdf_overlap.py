import os
import pickle

import h5py
import numpy as np
import torch

import utils


kmesh = (1, 3, 3)
data_dir = os.path.join(os.path.dirname(__file__), "data", "diamond_1x3x3_gth-szv")

with open(os.path.join(data_dir, "DFT.pkl"), "rb") as f:
    mf = pickle.load(f)
with h5py.File(os.path.join(data_dir, "THC_ov_ref.chk"), "r") as f:
    X_ao = np.asarray(f["inpv_kpt"])
    W = torch.from_numpy(np.asarray(f["coul_kpt"]))
lrdf = torch.load(os.path.join(data_dir, "LRDF_ov_ref.pt"), weights_only=False)

C = np.asarray(mf.mo_coeff)
nocc = mf.cell.nelectron // 2
Xo = torch.from_numpy(X_ao @ C[:, :, :nocc])
Xv = torch.from_numpy(X_ao @ C[:, :, nocc:])
D = lrdf["D"]
G = lrdf["G"]

assert lrdf["kmesh"] == kmesh
assert lrdf["rank"] == 5
assert lrdf["orbital_basis"] == "canonical_mo"
assert D.shape == (9, 5, 4, 4)
assert G.shape == (9, 5, 5)

k = torch.arange(9)
q = torch.arange(9)
kq = utils.add_k(k[None, :], q[:, None], kmesh)
P = torch.einsum("kIi,qkIa->qkIia", Xo, Xv[kq].conj())


def test_thc_lrdf_inner():
    eri_thc = torch.einsum("qkIia,qIJ,qlJjb->qkliajb", P.conj(), W, P)
    eri_lrdf = torch.einsum("kria,qrs,lsjb->qkliajb", D.conj(), G, D)
    inner_ref_q = torch.einsum("qkliajb,qkliajb->q", eri_thc.conj(), eri_lrdf)
    inner_q = utils.thc_lrdf_ov_inner_from_mo(Xo, Xv, W, D, G, kmesh, by_q=True)

    assert torch.linalg.norm(inner_q - inner_ref_q) / torch.linalg.norm(inner_ref_q) < 1e-12
    assert torch.abs(inner_q.sum() - utils.thc_lrdf_ov_inner_from_mo(Xo, Xv, W, D, G, kmesh)) < 1e-10


def test_lrdf_lrdf_inner():
    eri_lrdf = torch.einsum("kria,qrs,lsjb->qkliajb", D.conj(), G, D)
    inner_ref_q = torch.einsum("qkliajb,qkliajb->q", eri_lrdf.conj(), eri_lrdf)
    inner_q = utils.lrdf_ov_inner_from_mo(D, G, D, G, kmesh, by_q=True)

    assert torch.linalg.norm(inner_q - inner_ref_q) / torch.linalg.norm(inner_ref_q) < 1e-12
    assert torch.abs(inner_q.sum() - utils.lrdf_ov_inner_from_mo(D, G, D, G, kmesh)) < 1e-10
