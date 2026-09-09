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

def build_Pov_thc(Xo, Xv, k1, k2):
    return torch.einsum("...Ii,...Ia->...Iia", Xo[k1].conj(), Xv[k2])


def build_Pvo_thc(Xo, Xv, k1, k2):
    return torch.einsum("...Ia,...Ii->...Iai", Xv[k1].conj(), Xo[k2])


def build_time_reversal_U(C, nocc, kmesh):
    nkpts = C.shape[0]
    o, v = get_part(nocc)
    k = torch.arange(nkpts, device=device)
    neg = utils.negative_k(k, kmesh).to(device=device)
    Uo = torch.empty((nkpts, nocc, nocc), dtype=complex_dtype, device=device)
    Uv = torch.empty((nkpts, C.shape[-1] - nocc, C.shape[-1] - nocc), dtype=complex_dtype, device=device)
    for ik in range(nkpts):
        Uo[ik] = torch.linalg.lstsq(C[ik, :, o].conj(), C[neg[ik], :, o]).solution
        Uv[ik] = torch.linalg.lstsq(C[ik, :, v].conj(), C[neg[ik], :, v]).solution
    return Uo, Uv


def build_Pvo_time_reversal(Xo, Xv, Uo, Uv, kj, kb, neg):
    P = build_Pvo_thc(Xo, Xv, neg[kb], neg[kj])
    return torch.einsum("...Ixy,...xa,...yj->...Iaj", P, Uv[kb].transpose(-1, -2), Uo[kj].conj().transpose(-1, -2))


def build_Rov_thc(X, W, nocc, kmesh):
    nkpts = X.shape[0]
    nvir = X.shape[-1] - nocc
    nth = X.shape[1]
    o, v = get_part(nocc)
    Xo = X[:, :, o]
    Xv = X[:, :, v]
    k = torch.arange(nkpts, device=device)
    neg = utils.negative_k(k, kmesh).to(device=device)

    R = torch.empty((nkpts, nkpts, nth, nocc, nvir), dtype=complex_dtype, device=device)
    for k1 in range(nkpts):
        for k2 in range(nkpts):
            q = utils.add_k(k2, neg[k1], kmesh).item()
            P = build_Pov_thc(Xo, Xv, k1, k2).reshape((nth, nocc * nvir))
            R[k1, k2] = (P.T @ W[q]).T.reshape((nth, nocc, nvir))
    return R, Xo, Xv


def canonical_mp2_from_R(R, Xo, Xv, mo_energy, nocc, kmesh, verbose=False, convert=False, Uo=None, Uv=None):
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
        if convert:
            Pbj = build_Pvo_time_reversal(Xo, Xv, Uo, Uv, k, km, neg)
        else:
            Pjb = build_Pov_thc(Xo, Xv, k, km)
        direct = torch.tensor(0.0, dtype=real_dtype, device=device)
        exchange = torch.tensor(0.0, dtype=real_dtype, device=device)
        for ik in range(nkpts):
            ka = kp[ik]
            Ria = R[ik, ka]
            if convert:
                G = torch.einsum('xia,lxbj->liajb', Ria, Pbj) / nkpts
            else:
                G = torch.einsum('xia,lxjb->liajb', Ria, Pjb) / nkpts
            denom = (
                eocc[ik][None, :, None, None, None]
                - evir[ka][None, None, :, None, None]
                + eocc[:, None, None, :, None]
                - evir[km][:, None, None, None, :]
            )
            T = G.conj() / denom
            direct += torch.einsum('liajb,liajb->', T, G).real
            Rib = R[ik, km]
            k2 = torch.full((nkpts,), ka, dtype=torch.long, device=device)
            if convert:
                Paj = build_Pvo_time_reversal(Xo, Xv, Uo, Uv, k, k2, neg)
                H = torch.einsum('lxib,lxaj->lijba', Rib, Paj) / nkpts
            else:
                Pja = build_Pov_thc(Xo, Xv, k, k2)
                H = torch.einsum('lxib,lxja->lijba', Rib, Pja) / nkpts
            exchange -= torch.einsum('liajb,lijba->', T, H).real
        emp2 += 2.0 * direct + exchange
        if verbose:
            print('q %3d/%d  direct %.12e  exchange %.12e  time %.4f' % (
                q + 1, nkpts, direct.item(), exchange.item(), time.time() - t0,
            ), flush=True)
    emp2 /= nkpts
    return emp2


def canonical_mp2_from_thc(X, C, W, mo_energy, nocc, kmesh, verbose=False, convert=False):
    R, Xo, Xv = build_Rov_thc(X, W, nocc, kmesh)
    if convert:
        Uo, Uv = build_time_reversal_U(C, nocc, kmesh)
    else:
        Uo, Uv = None, None
    print("R memory GB", R.numel() * R.element_size() / 1024**3)
    return canonical_mp2_from_R(R, Xo, Xv, mo_energy, nocc, kmesh, verbose=verbose, convert=convert, Uo=Uo, Uv=Uv)



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
parser.add_argument("--real", action="store_true")
parser.add_argument("--convert", action="store_true")
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
symm_tag = "_symm" if args.symm else ""
real_tag = "_real" if args.real else ""
dft_pkl = os.path.join(data_dir, f"DFT_{klabel}{symm_tag}.pkl")

if args.ov_ref is not None:
    ov_chk = os.path.join(data_dir, f"ISDFov_bareGDF{symm_tag}_{klabel}_c{args.ov_ref}.chk")
else:
    ov_chk = os.path.join(data_dir, f"ISDFov_opt_bareGDF{symm_tag}{real_tag}_{klabel}_{args.ov_opt}.chk")

with open(dft_pkl, "rb") as f:
    mf = pickle.load(f)

nocc = mf.cell.nelectron // 2
mo_energy = np.asarray(mf.mo_energy)
C = np.asarray(mf.mo_coeff)
C_t = torch.from_numpy(C).to(device=device, dtype=complex_dtype)

print("kmesh", kmesh)
print("basis", basis)
print("suffix", suffix)
print("cuda", args.cuda)
print("exact", args.exact)
print("verbose", args.verbose)
print("symm", args.symm)
print("real", args.real)
print("convert", args.convert)
print("ov_ref", args.ov_ref)
print("ov_opt", args.ov_opt)
print("nocc", nocc, "nmo", mo_energy.shape[-1], "nkpts", len(mf.kpts))
print("SCF energy", mf.e_tot)
print("DFT pkl", dft_pkl)
print("OV THC", ov_chk)

t0 = time.time()
X, W = load_thc(ov_chk, C)
print("load THC time", time.time() - t0)
print("nth", X.shape[1])

t0 = time.time()
emp2_custom = canonical_mp2_from_thc(X, C_t, W, mo_energy, nocc, kmesh, verbose=args.verbose, convert=args.convert)
print("MP2 contraction time", time.time() - t0)
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
