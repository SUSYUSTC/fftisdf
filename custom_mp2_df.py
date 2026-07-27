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


def canonical_mp2_from_R(R, mo_energy, nocc, kmesh, verbose=False):
    nkpts = R.shape[0]
    nvir = R.shape[-1]
    k = torch.arange(nkpts, device=device)
    neg = utils.negative_k(k, kmesh).to(device=device)
    kq = utils.add_k(k[:, None], k[None, :], kmesh).to(device=device)
    e = torch.from_numpy(np.asarray(mo_energy)).to(device=device, dtype=real_dtype)
    eocc = e[:, :nocc]
    evir = e[:, nocc:nocc+nvir]

    emp2 = torch.tensor(0.0, dtype=real_dtype, device=device)
    for q in range(nkpts):
        t0 = time.time()
        kp = kq[:, q]
        km = kq[:, neg[q]]
        Ria = R[k, kp]
        Rjb = R[k, km]
        G = torch.einsum("kxia,lxjb->kliajb", Ria, Rjb) / nkpts
        denom = (
            eocc[:, None, :, None, None, None]
            - evir[kp][:, None, None, :, None, None]
            + eocc[None, :, None, None, :, None]
            - evir[km][None, :, None, None, None, :]
        )
        T = G.conj() / denom
        direct = torch.einsum("kliajb,kliajb->", T, G).real
        exchange = torch.tensor(0.0, dtype=real_dtype, device=device)
        for ik in range(nkpts):
            ka = kp[ik]
            Rib = R[ik, km]
            Rja = R[k, ka]
            H = torch.einsum("lxib,lxja->lijba", Rib, Rja) / nkpts
            exchange -= torch.einsum("liajb,lijba->", T[ik], H).real
        emp2 += 2.0 * direct + exchange
        if verbose:
            print("q %3d  direct %.12e  exchange %.12e  time %.4f" % (
                q, direct.item(), exchange.item(), time.time() - t0,
            ), flush=True)
    emp2 /= nkpts
    return emp2


parser = argparse.ArgumentParser()
parser.add_argument("system")
parser.add_argument("kx", type=int)
parser.add_argument("ky", type=int)
parser.add_argument("kz", type=int)
parser.add_argument("basis")
parser.add_argument("-suffix", default=None)
parser.add_argument("-cuda", type=int, default=None)
parser.add_argument("--exact", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--symm", action="store_true")
args = parser.parse_args()

device = torch.device(f"cuda:{args.cuda}" if args.cuda is not None else "cpu")
system = args.system
kmesh = (args.kx, args.ky, args.kz)
basis = args.basis
suffix = args.suffix
klabel = system_common.get_klabel(kmesh)
symm_tag = "_symm" if args.symm else ""
data_dir = system_common.get_data_dir(system, basis, suffix=suffix)
dft_pkl = os.path.join(data_dir, f"DFT_{klabel}{symm_tag}.pkl")
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
print("exact", args.exact)
print("verbose", args.verbose)
print("symm", args.symm)
print("nocc", nocc, "nmo", mo_energy.shape[-1], "nkpts", len(kpts))
print("SCF energy", mf.e_tot)
print("DFT pkl", dft_pkl)

t0 = time.time()
R = build_Rov(mf, nocc, kpts_int, kmesh)
print("build R time", time.time() - t0)
print("naux", R.shape[2])

emp2_custom = canonical_mp2_from_R(R, mo_energy, nocc, kmesh, verbose=args.verbose)
print("custom DF MP2 energy = %.16e" % emp2_custom.detach().cpu().numpy())

if args.exact:
    pt = mp.KMP2(mf)
    pt.verbose = 0
    emp2_pyscf, _ = pt.kernel(with_t2=False)
    print("PySCF MP2 energy     = %.16e" % emp2_pyscf)
    print("custom - PySCF       = %.16e" % (emp2_custom.detach().cpu().numpy() - emp2_pyscf))
