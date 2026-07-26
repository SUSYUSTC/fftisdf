import argparse
import os
import pickle
import time
import h5py
import numpy as np
import torch
from pyscf.pbc import mp

import fft
import system_common
import utils

complex_dtype = torch.complex128
real_dtype = torch.float64


def get_kpts_int(cell, kpts, kmesh):
    kpts_int = np.round(cell.get_scaled_kpts(kpts) * kmesh).astype(int) % kmesh
    assert utils.is_k_ordered(kpts_int, kmesh)
    return kpts_int


def get_part(nocc):
    o = slice(0, nocc)
    v = slice(nocc, None)
    return o, v


def get_quadrature(mo_energy, nocc, M, a=0.5):
    eocc = torch.from_numpy(np.asarray(mo_energy)[:, :nocc].reshape(-1)).to(dtype=real_dtype)
    evir = torch.from_numpy(np.asarray(mo_energy)[:, nocc:].reshape(-1)).to(dtype=real_dtype)
    Delta_min = 2 * torch.min(evir) - 2 * torch.max(eocc)
    nodes, weights = np.polynomial.legendre.leggauss(M)
    x_nodes = 0.5 * (nodes + 1.0)
    x_weights = 0.5 * weights
    beta_weight = []
    for x, w in zip(x_nodes, x_weights):
        beta = -np.log(x) / (a * Delta_min.item())
        weight = w / (a * Delta_min.item() * x)
        beta_weight.append((beta, weight))
    return beta_weight


def load_thc(chkfile, C):
    with h5py.File(chkfile, "r") as f:
        X_ao = np.asarray(f["inpv_kpt"])
        W = np.asarray(f["coul_kpt"])
    X = np.einsum("kIa,kab->kIb", X_ao, C)
    X = torch.from_numpy(X).to(device=device, dtype=complex_dtype)
    W = torch.from_numpy(W).to(device=device, dtype=complex_dtype)
    return X, W


def apply_laplace_X(X, mo_energy, nocc, beta, weight):
    o, v = get_part(nocc)
    eocc = torch.from_numpy(np.asarray(mo_energy)[:, o]).to(device=device, dtype=real_dtype)
    evir = torch.from_numpy(np.asarray(mo_energy)[:, v]).to(device=device, dtype=real_dtype)
    Xo = X[:, :, o] * torch.exp(0.5 * beta * eocc)[:, None, :] * (weight ** 0.125)
    Xv = X[:, :, v] * torch.exp(-0.5 * beta * evir)[:, None, :] * (weight ** 0.125)
    return Xo, Xv


def laplace_mp2_from_thc(X, W, mo_energy, nocc, kmesh, M):
    nkpts = X.shape[0]
    k = torch.arange(nkpts, device=device)
    neg = utils.negative_k(k, kmesh).to(device=device)
    kq = utils.add_k(k[:, None], k[None, :], kmesh).to(device=device)

    J = torch.tensor(0.0, dtype=complex_dtype, device=device)
    K = torch.tensor(0.0, dtype=complex_dtype, device=device)
    for ibeta, (beta, weight) in enumerate(get_quadrature(mo_energy, nocc, M)):
        t0 = time.time()
        Xo, Xv = apply_laplace_X(X, mo_energy, nocc, beta, weight)
        Oo = torch.einsum("kIi,kKi->kIK", Xo.conj(), Xo)
        Vv = torch.einsum("kIa,kKa->kIK", Xv, Xv.conj())
        J_beta = torch.tensor(0.0, dtype=complex_dtype, device=device)
        K_beta = torch.tensor(0.0, dtype=complex_dtype, device=device)
        for q in range(nkpts):
            kp = kq[:, q]
            km = kq[:, neg[q]]
            A = torch.einsum("kIK,kIK->IK", Oo, Vv[kp])
            B = torch.einsum("kJL,kJL->JL", Oo, Vv[km])
            J_beta += torch.einsum("IJ,KL,IK,JL->", W[q], W[q].conj(), A, B)
            for ik in range(nkpts):
                km_ik = km
                q2 = utils.add_k(km_ik, -ik, kmesh).to(device=device)
                K_beta += torch.einsum(
                    "IJ,lKL,IK,IL,lJL,lJK->",
                    W[q], W[q2].conj(), Oo[ik], Vv[kp[ik]], Oo, Vv[km],
                )
        J += J_beta
        K += K_beta
        print("beta %2d  beta %.8e  weight %.8e  J %.12e  K %.12e  time %.4f" % (
            ibeta, beta, weight, J_beta.real.item(), K_beta.real.item(), time.time() - t0,
        ), flush=True)
    emp2 = (K - 2.0 * J).real / (nkpts ** 3)
    return emp2


parser = argparse.ArgumentParser()
parser.add_argument("system")
parser.add_argument("kx", type=int)
parser.add_argument("ky", type=int)
parser.add_argument("kz", type=int)
parser.add_argument("basis")
parser.add_argument("-suffix", default=None)
parser.add_argument("-cuda", type=int, default=None)
parser.add_argument("-M", type=int, default=12)
parser.add_argument("--exact", action="store_true")
ov_group = parser.add_mutually_exclusive_group(required=True)
ov_group.add_argument("-ov_ref", type=int, default=None)
ov_group.add_argument("-ov_opt", default=None)
args = parser.parse_args()

device = torch.device(f"cuda:{args.cuda}" if args.cuda is not None else "cpu")
system = args.system
kmesh = (args.kx, args.ky, args.kz)
basis = args.basis
suffix = args.suffix
klabel = system_common.get_klabel(kmesh)
data_dir = system_common.get_data_dir(system, basis, suffix=suffix)
dft_pkl = os.path.join(data_dir, f"DFT_{klabel}.pkl")

if args.ov_ref is not None:
    ov_chk = os.path.join(data_dir, f"ISDFov_bareGDF_{klabel}_c{args.ov_ref}.chk")
else:
    ov_chk = os.path.join(data_dir, f"ISDFov_opt_bareGDF_{klabel}_{args.ov_opt}.chk")

with open(dft_pkl, "rb") as f:
    mf = pickle.load(f)

kpts = np.asarray(mf.kpts)
kpts_int = get_kpts_int(mf.cell, kpts, kmesh)
nocc = mf.cell.nelectron // 2
mo_energy = np.asarray(mf.mo_energy)
C = np.asarray(mf.mo_coeff)

print("kmesh", kmesh)
print("basis", basis)
print("suffix", suffix)
print("cuda", args.cuda)
print("M", args.M)
print("exact", args.exact)
print("ov_ref", args.ov_ref)
print("ov_opt", args.ov_opt)
print("nocc", nocc, "nmo", mo_energy.shape[-1], "nkpts", len(kpts))
print("SCF energy", mf.e_tot)
print("OV THC", ov_chk)

t0 = time.time()
X, W = load_thc(ov_chk, C)
print("load THC time", time.time() - t0)

emp2_lt = laplace_mp2_from_thc(X, W, mo_energy, nocc, kmesh, args.M)
print("LT THC MP2 energy    = %.16e" % emp2_lt.detach().cpu().numpy())

if args.exact:
    mf_isdf = mf.copy()
    mf_isdf.with_df = fft.ISDF(mf.cell, kpts)
    mf_isdf.with_df._isdf = ov_chk
    mf_isdf.with_df.build()
    pt = mp.KMP2(mf_isdf)
    pt.verbose = 0
    emp2_pyscf, _ = pt.kernel(with_t2=False)
    print("PySCF THC MP2 energy = %.16e" % emp2_pyscf)
    print("LT - PySCF           = %.16e" % (emp2_lt.detach().cpu().numpy() - emp2_pyscf))
