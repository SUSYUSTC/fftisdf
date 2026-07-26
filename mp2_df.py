import argparse
import os
import pickle
import time
import numpy as np
import torch
from pyscf.pbc import df
from pyscf.pbc import mp

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


def build_Rov(mf, nocc, kpts_int, kmesh):
    o, v = get_part(nocc)
    C = np.asarray(mf.mo_coeff)
    Cocc = C[:, :, o]
    Cvir = C[:, :, v]
    R = utils.get_gdf_tensor_compact(
        mf.with_df,
        kpts_int,
        kmesh,
        C1=Cocc,
        C2=Cvir,
        layout="k1k2",
        progressbar=False,
    )
    return torch.from_numpy(R).to(device=device, dtype=complex_dtype)


def apply_laplace_R(R, mo_energy, nocc, beta, weight):
    o, v = get_part(nocc)
    eocc = torch.from_numpy(np.asarray(mo_energy)[:, o]).to(device=device, dtype=real_dtype)
    evir = torch.from_numpy(np.asarray(mo_energy)[:, v]).to(device=device, dtype=real_dtype)
    focc = torch.exp(0.5 * beta * eocc)
    fvir = torch.exp(-0.5 * beta * evir)
    Rhalf = R * focc[:, None, None, :, None] * fvir[None, :, None, None, :]
    return Rhalf * (weight ** 0.25)


def laplace_mp2_from_R(R, mo_energy, nocc, kmesh, M):
    nkpts = R.shape[0]
    k = torch.arange(nkpts, device=device)
    neg = utils.negative_k(k, kmesh).to(device=device)
    kq = utils.add_k(k[:, None], k[None, :], kmesh).to(device=device)

    J = torch.tensor(0.0, dtype=complex_dtype, device=device)
    K = torch.tensor(0.0, dtype=complex_dtype, device=device)
    for ibeta, (beta, weight) in enumerate(get_quadrature(mo_energy, nocc, M)):
        t0 = time.time()
        Rhalf = apply_laplace_R(R, mo_energy, nocc, beta, weight)
        J_beta = torch.tensor(0.0, dtype=complex_dtype, device=device)
        K_beta = torch.tensor(0.0, dtype=complex_dtype, device=device)
        for q in range(nkpts):
            kp = kq[:, q]
            km = kq[:, neg[q]]
            Ria = Rhalf[k, kp]
            Rjb = Rhalf[k, km]
            J_beta += torch.einsum("kxia,lxjb,kyia,lyjb->", Ria, Rjb, Ria.conj(), Rjb.conj())
            for ik in range(nkpts):
                Rib = Rhalf[ik, km]
                Rja = Rhalf[k, kp[ik]]
                K_beta += torch.einsum("xia,lxjb,lyib,lyja->", Ria[ik], Rjb, Rib.conj(), Rja.conj())
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
args = parser.parse_args()

device = torch.device(f"cuda:{args.cuda}" if args.cuda is not None else "cpu")
system = args.system
kmesh = (args.kx, args.ky, args.kz)
basis = args.basis
suffix = args.suffix
klabel = system_common.get_klabel(kmesh)
data_dir = system_common.get_data_dir(system, basis, suffix=suffix)
dft_pkl = os.path.join(data_dir, f"DFT_{klabel}.pkl")
gdf_chk = os.path.join(data_dir, f"GDF_{klabel}.chk")

with open(dft_pkl, "rb") as f:
    mf = pickle.load(f)
if isinstance(mf.with_df, df.GDF):
    mf.with_df._cderi = gdf_chk

kpts = np.asarray(mf.kpts)
kpts_int = get_kpts_int(mf.cell, kpts, kmesh)
nocc = mf.cell.nelectron // 2
mo_energy = np.asarray(mf.mo_energy)

print("kmesh", kmesh)
print("basis", basis)
print("suffix", suffix)
print("cuda", args.cuda)
print("M", args.M)
print("exact", args.exact)
print("nocc", nocc, "nmo", mo_energy.shape[-1], "nkpts", len(kpts))
print("SCF energy", mf.e_tot)

t0 = time.time()
R = build_Rov(mf, nocc, kpts_int, kmesh)
print("build R time", time.time() - t0)

emp2_lt = laplace_mp2_from_R(R, mo_energy, nocc, kmesh, args.M)
print("LT DF MP2 energy     = %.16e" % emp2_lt.detach().cpu().numpy())

if args.exact:
    pt = mp.KMP2(mf)
    pt.verbose = 0
    emp2_pyscf, _ = pt.kernel(with_t2=False)
    print("PySCF MP2 energy     = %.16e" % emp2_pyscf)
    print("LT - PySCF           = %.16e" % (emp2_lt.detach().cpu().numpy() - emp2_pyscf))
