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


def load_thc(chkfile, C):
    with h5py.File(chkfile, "r") as f:
        X_ao = np.asarray(f["inpv_kpt"])
        W = np.asarray(f["coul_kpt"])
    X = np.einsum("kIa,kab->kIb", X_ao, C)
    X = torch.from_numpy(X).to(device=device, dtype=complex_dtype)
    W = torch.from_numpy(W).to(device=device, dtype=complex_dtype)
    return X, W



def build_Rov_thc(X, W, nocc, kmesh):
    nkpts = X.shape[0]
    nvir = X.shape[-1] - nocc
    nth = X.shape[1]
    o, v = get_part(nocc)
    Xo = X[:, :, o]
    Xv = X[:, :, v]
    k = torch.arange(nkpts, device=device)
    neg = utils.negative_k(k, kmesh).to(device=device)

    Rleft = torch.empty((nkpts, nkpts, nth, nocc, nvir), dtype=complex_dtype, device=device)
    Rright = torch.empty((nkpts, nkpts, nth, nocc, nvir), dtype=complex_dtype, device=device)
    for k1 in range(nkpts):
        for k2 in range(nkpts):
            q = utils.add_k(k2, neg[k1], kmesh).item()
            P = torch.einsum('Ii,Ia->Iia', Xo[k1].conj(), Xv[k2])
            P = P.reshape((nth, nocc * nvir))
            Rleft[k1, k2] = (P.T @ W[q]).T.reshape((nth, nocc, nvir))
            Rright[k1, k2] = P.reshape((nth, nocc, nvir))
    return Rleft, Rright


def canonical_mp2_from_Rlr(Rleft, Rright, mo_energy, nocc, kmesh, verbose=False):
    nkpts = Rleft.shape[0]
    nvir = Rleft.shape[-1]
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
        Ria = Rleft[k, kp]
        Rjb = Rright[k, km]
        G = torch.einsum('kxia,lxjb->kliajb', Ria, Rjb) / nkpts
        denom = (
            eocc[:, None, :, None, None, None]
            - evir[kp][:, None, None, :, None, None]
            + eocc[None, :, None, None, :, None]
            - evir[km][None, :, None, None, None, :]
        )
        T = G.conj() / denom
        direct = torch.einsum('kliajb,kliajb->', T, G).real
        exchange = torch.tensor(0.0, dtype=real_dtype, device=device)
        for ik in range(nkpts):
            ka = kp[ik]
            Rib = Rleft[ik, km]
            Rja = Rright[k, ka]
            H = torch.einsum('lxib,lxja->lijba', Rib, Rja) / nkpts
            exchange -= torch.einsum('liajb,lijba->', T[ik], H).real
        emp2 += 2.0 * direct + exchange
        if verbose:
            print('q %3d  direct %.12e  exchange %.12e  time %.4f' % (
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

nocc = mf.cell.nelectron // 2
mo_energy = np.asarray(mf.mo_energy)
C = np.asarray(mf.mo_coeff)

print("kmesh", kmesh)
print("basis", basis)
print("suffix", suffix)
print("cuda", args.cuda)
print("exact", args.exact)
print("verbose", args.verbose)
print("ov_ref", args.ov_ref)
print("ov_opt", args.ov_opt)
print("nocc", nocc, "nmo", mo_energy.shape[-1], "nkpts", len(mf.kpts))
print("SCF energy", mf.e_tot)
print("OV THC", ov_chk)

t0 = time.time()
X, W = load_thc(ov_chk, C)
print("load THC time", time.time() - t0)
print("nth", X.shape[1])

t0 = time.time()
Rleft, Rright = build_Rov_thc(X, W, nocc, kmesh)
print("build THC-R time", time.time() - t0)
emp2_custom = canonical_mp2_from_Rlr(Rleft, Rright, mo_energy, nocc, kmesh, verbose=args.verbose)
print("custom THC MP2 energy = %.16e" % emp2_custom.detach().cpu().numpy())

if args.exact:
    mf_isdf = mf.copy()
    mf_isdf.with_df = fft.ISDF(mf.cell, mf.kpts)
    mf_isdf.with_df._isdf = ov_chk
    mf_isdf.with_df.build()
    pt = mp.KMP2(mf_isdf)
    pt.verbose = 0
    emp2_pyscf, _ = pt.kernel(with_t2=False)
    print("PySCF THC MP2 energy  = %.16e" % emp2_pyscf)
    print("custom - PySCF        = %.16e" % (emp2_custom.detach().cpu().numpy() - emp2_pyscf))
