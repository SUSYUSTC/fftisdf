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
    X_ao_ref = np.asarray(f["inpv_kpt"])
    W_ref = torch.from_numpy(np.asarray(f["coul_kpt"]))
lrdf = torch.load(os.path.join(data_dir, "LRDF_ov_ref.pt"), weights_only=False)

C = np.asarray(mf.mo_coeff)
nocc = mf.cell.nelectron // 2
Cocc = C[:, :, :nocc]
Cvir = C[:, :, nocc:]
Xo_ref = torch.from_numpy(X_ao_ref @ Cocc)
Xv_ref = torch.from_numpy(X_ao_ref @ Cvir)

nI = 24
Xo = Xo_ref[:, :nI].clone()
Xv = Xv_ref[:, :nI].clone()
D = lrdf["D"]
G = lrdf["G"]

k = torch.arange(9)
q = torch.arange(9)
kq = utils.add_k(k[None, :], q[:, None], kmesh)


def test_thc_thclrdf_solve_w_error2():
    P_ref = torch.einsum("kIi,qkIa->qkIia", Xo_ref, Xv_ref[kq].conj())
    eri_ref = torch.einsum("qkIia,qIJ,qlJjb->qkliajb", P_ref.conj(), W_ref, P_ref)
    ref_norm2 = torch.linalg.norm(eri_ref)**2

    W, error2 = utils.thc_thclrdf_ov_solve_w_error2_from_mo(
        Xo_ref, Xv_ref, W_ref,
        Xo, Xv, D, G,
        kmesh, ref_norm2, reg=1e-5,
    )

    P = torch.einsum("kIi,qkIa->qkIia", Xo, Xv[kq].conj())
    eri_thc = torch.einsum("qkIia,qIJ,qlJjb->qkliajb", P.conj(), W, P)
    eri_lrdf = torch.einsum("kria,qrs,lsjb->qkliajb", D.conj(), G, D)
    error2_ref = torch.linalg.norm(eri_ref - eri_thc - eri_lrdf)**2

    assert torch.abs(error2 - error2_ref) / error2_ref < 1e-12
