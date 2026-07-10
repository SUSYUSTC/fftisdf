import argparse
import os
import pickle
import h5py
import numpy as np
import scipy.sparse.linalg
import torch
import system_common
import utils

complex_dtype = torch.complex64
real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
scipy_dtype = np.complex128 if complex_dtype == torch.complex128 else np.complex64


def get_part(nocc):
    o = slice(0, nocc)
    v = slice(nocc, None)
    return o, v


def get_kpts_int(cell, kpts, kmesh):
    kpts_int = np.round(cell.get_scaled_kpts(kpts) * kmesh).astype(int) % kmesh
    assert utils.is_k_ordered(kpts_int, kmesh)
    return kpts_int


def load_thc(chkfile, C):
    with h5py.File(chkfile, "r") as f:
        X_ao = np.asarray(f["inpv_kpt"])
        W = np.asarray(f["coul_kpt"])
    X = np.einsum("kIa,kab->kIb", X_ao, C)
    X = torch.from_numpy(X).to(device=device, dtype=complex_dtype)
    W = torch.from_numpy(W).to(device=device, dtype=complex_dtype)
    return X, W


def fft_k(A, inverse=False):
    return utils.fourier_transform_3d(A, axis=0, kmesh=kmesh, inverse=inverse)


def get_W_fft_neg(W):
    nkpts = W.shape[0]
    negative = utils.negative_k(torch.arange(nkpts, device=W.device), kmesh)
    W_fft = fft_k(W, inverse=False)
    return W_fft[negative]


def get_eia(nocc, mo_energy):
    o, v = get_part(nocc)
    eia = mo_energy[:, o, None] * -1.0
    eia = eia + mo_energy[:, None, v]
    return torch.from_numpy(eia).to(device=device, dtype=real_dtype)


def get_eia_q(nocc, mo_energy, kq_q):
    o, v = get_part(nocc)
    mo_energy_t = torch.from_numpy(mo_energy).to(device=device, dtype=real_dtype)
    eia = mo_energy_t[kq_q][:, None, v] - mo_energy_t[:, o, None]
    return eia


def build_kq_map(kmesh):
    nkpts = int(np.prod(kmesh))
    k = torch.arange(nkpts, dtype=torch.long)
    q = torch.arange(nkpts, dtype=torch.long)
    kq = utils.add_k(k[:, None], q[None, :], kmesh)
    return kq.to(device=device, dtype=torch.long)


def apply_V_thc(nocc, X, W, x):
    o, v = get_part(nocc)
    nkpts = X.shape[0]
    Xi = X[:, :, o]
    Xa = X[:, :, v]
    t = torch.einsum("lJj,lJb,ljb->J", Xi.conj(), Xa, x)
    z = torch.einsum("IJ,J->I", W[0], t)
    y = (4.0 / nkpts) * torch.einsum("kIi,kIa,I->kia", Xi, Xa.conj(), z)
    return y


def apply_V_thc_q(nocc, X, W, kq_q, q, x):
    o, v = get_part(nocc)
    nkpts = X.shape[0]
    Xi = X[:, :, o]
    Xa = X[:, :, v]
    Xa_q = Xa[kq_q]
    t = torch.einsum("lJj,lJb,ljb->J", Xi.conj(), Xa_q, x)
    z = torch.einsum("IJ,J->I", W[q], t)
    y = (4.0 / nkpts) * torch.einsum("kIi,kIa,I->kia", Xi, Xa_q.conj(), z)
    return y


def apply_Woovv_thc(nocc, X, W_fft_neg, x):
    o, v = get_part(nocc)
    Xi = X[:, :, o]
    Xa = X[:, :, v]
    B = torch.einsum("lJb,ljb->lJj", Xa, x)
    E = torch.einsum("lIj,lJj->lIJ", Xi.conj(), B)
    y = apply_screened_E_thc(nocc, X, W_fft_neg, E)
    return y


def apply_Woovv_thc_q(nocc, X, W_fft_neg, kq_q, x):
    o, v = get_part(nocc)
    Xi = X[:, :, o]
    Xa = X[:, :, v]
    Xa_q = Xa[kq_q]
    B = torch.einsum("lJb,ljb->lJj", Xa_q, x)
    E = torch.einsum("lIj,lJj->lIJ", Xi.conj(), B)
    y = apply_screened_E_thc(nocc, X, W_fft_neg, E)
    return y


def apply_screened_E_thc(nocc, X, W_fft_neg, E):
    o, v = get_part(nocc)
    nkpts = X.shape[0]
    Xi = X[:, :, o]
    Xa = X[:, :, v]
    E_fft = fft_k(E, inverse=False)
    F_fft = np.sqrt(nkpts).item() * W_fft_neg * E_fft
    F = fft_k(F_fft, inverse=True)
    G = torch.einsum("kIi,kIJ->kiJ", Xi, F)
    y = -(1.0 / nkpts) * torch.einsum("kiJ,kJa->kia", G, Xa.conj())
    return y


def apply_A_thc(nocc, X_bare, W_bare, X_screen, W_screen_fft_neg, eia, gamma, x):
    y = eia * x
    y += apply_Woovv_thc(nocc, X_screen, W_screen_fft_neg, x)
    y -= gamma * x
    y += 0.5 * apply_V_thc(nocc, X_bare, W_bare, x)
    return y


def apply_A_thc_q(nocc, X_bare, W_bare, X_screen, W_screen_fft_neg, kq_q, q, eia, gamma, x):
    y = eia * x
    y += apply_Woovv_thc_q(nocc, X_screen, W_screen_fft_neg, kq_q, x)
    y -= gamma * x
    y += 0.5 * apply_V_thc_q(nocc, X_bare, W_bare, kq_q, q, x)
    return y


def make_tda_operator_thc(nocc, X_bare, W_bare, X_screen, W_screen_fft_neg, eia, gamma):
    nkpts, nocc, nvir = eia.shape
    dim = nkpts * nocc * nvir
    it = 0

    def matvec(x):
        nonlocal it
        print('it', it, end='\r')
        it += 1

        x = torch.from_numpy(x.reshape(nkpts, nocc, nvir)).to(device=device, dtype=complex_dtype)
        y = apply_A_thc(nocc, X_bare, W_bare, X_screen, W_screen_fft_neg, eia, gamma, x)
        return y.reshape(-1).detach().cpu().numpy()

    return scipy.sparse.linalg.LinearOperator((dim, dim), matvec=matvec, dtype=scipy_dtype)


def make_tda_operator_thc_q(nocc, X_bare, W_bare, X_screen, W_screen_fft_neg, kq_q, q, eia, gamma):
    nkpts, nocc, nvir = eia.shape
    dim = nkpts * nocc * nvir
    it = 0

    def matvec(x):
        nonlocal it
        print("it", it, end="\r")
        it += 1
        x = torch.from_numpy(x.reshape(nkpts, nocc, nvir)).to(device=device, dtype=complex_dtype)
        y = apply_A_thc_q(nocc, X_bare, W_bare, X_screen, W_screen_fft_neg, kq_q, q, eia, gamma, x)
        return y.reshape(-1).detach().cpu().numpy()

    return scipy.sparse.linalg.LinearOperator((dim, dim), matvec=matvec, dtype=scipy_dtype)


def solve_tda_thc(nocc, X_bare, W_bare, X_screen, W_screen_fft_neg, eia, gamma):
    op = make_tda_operator_thc(nocc, X_bare, W_bare, X_screen, W_screen_fft_neg, eia, gamma)
    e, x = scipy.sparse.linalg.eigsh(op, k=nroot, which="SA", tol=eig_tol)
    idx = np.argsort(e.real)
    return e[idx].real, x[:, idx].T


def solve_tda_thc_q(nocc, X_bare, W_bare, X_screen, W_screen_fft_neg, kq_q, q, eia, gamma):
    op = make_tda_operator_thc_q(nocc, X_bare, W_bare, X_screen, W_screen_fft_neg, kq_q, q, eia, gamma)
    e, x = scipy.sparse.linalg.eigsh(op, k=nroot, which="SA", tol=eig_tol)
    idx = np.argsort(e.real)
    return e[idx].real, x[:, idx].T


def get_indirect_q(nocc, mo_energy, kq):
    o, v = get_part(nocc)
    mo_energy_t = torch.from_numpy(mo_energy).to(device=device, dtype=real_dtype)
    e0 = []
    for q in range(kq.shape[1]):
        kq_q = kq[:, q]
        eq = mo_energy_t[kq_q][:, None, v] - mo_energy_t[:, o, None]
        e0.append(eq.min().item())
    e0 = np.array(e0)
    q = int(np.argmin(e0))
    return q, e0


nroot = 1
eig_tol = 1e-4

parser = argparse.ArgumentParser()
parser.add_argument("system")
parser.add_argument("kx", type=int)
parser.add_argument("ky", type=int)
parser.add_argument("kz", type=int)
parser.add_argument("basis")
parser.add_argument("-c_isdf", type=int, default=20)
parser.add_argument("-cuda", type=int, default=None)
parser.add_argument("-nroots", type=int, default=1)
parser.add_argument("-pattern-ov", default=None)
parser.add_argument("-pattern-full", default=None)
parser.add_argument("--use-Edft", action="store_true")
parser.add_argument("--unscreen", action="store_true")
parser.add_argument("--indirect", action="store_true")
args = parser.parse_args()

use_gpu = args.cuda is not None
device = torch.device(f"cuda:{args.cuda}" if use_gpu else "cpu")
nroot = args.nroots
system = args.system
kmesh = (args.kx, args.ky, args.kz)
basis = args.basis
c_isdf = args.c_isdf
pattern_ov = args.pattern_ov
pattern_full = args.pattern_full
use_Edft = args.use_Edft
unscreen = args.unscreen
TDA = True
klabel = system_common.get_klabel(kmesh)
data_dir = system_common.get_data_dir(system, basis)
dft_pkl = os.path.join(data_dir, f"DFT_{klabel}.pkl")
gw_path = os.path.join(data_dir, f"GWenergy_{klabel}.npy")
head_path = os.path.join(data_dir, f"bse_head_{klabel}.npy")
bare_ref_chk = os.path.join(data_dir, f"ISDFfull_bareGDF_{klabel}_c{c_isdf}.chk")
screen_ref_chk = os.path.join(data_dir, f"ISDFfull_screenGDF_{klabel}_c{c_isdf}.chk")


def get_full_chk(kind, pattern):
    if pattern is None:
        return bare_ref_chk if kind == "bare" else screen_ref_chk
    return os.path.join(data_dir, f"ISDFfull_opt_{kind}GDF_{klabel}_{pattern}.chk")


def get_ov_chk(pattern):
    if pattern is None:
        return None
    return os.path.join(data_dir, f"ISDFov_opt_bareGDF_{klabel}_{pattern}.chk")


with open(dft_pkl, "rb") as f:
    mf = pickle.load(f)


kpts = np.asarray(mf.kpts)
kpts_int = get_kpts_int(mf.cell, kpts, kmesh)
mo_energy = np.load(gw_path)
nocc = mf.cell.nelectron // 2
C = np.asarray(mf.mo_coeff)
utils.enable_fast_fft()
mo_energy_dft = np.asarray(mf.mo_energy)
mo_energy_qp = mo_energy_dft if use_Edft else mo_energy
gamma = 0.0 if unscreen else float(np.load(head_path))
nkpts = len(kpts)
eia_gw_q0 = mo_energy_qp[:, None, nocc:] - mo_energy_qp[:, :nocc, None]
eia_gw_q0 = np.sort(eia_gw_q0.reshape(-1))
eia_dft = mo_energy_dft[:, None, nocc:] - mo_energy_dft[:, :nocc, None]
eia_dft = np.sort(eia_dft.reshape(-1))

#print(mo_energy)
print("kmesh", kmesh)
print("basis", basis)
print("c_isdf", c_isdf)
print("cuda", args.cuda)
print("nroot", nroot)
print("pattern_ov", pattern_ov)
print("pattern_full", pattern_full)
print("use_Edft", use_Edft)
print("unscreen", unscreen)
print("indirect", args.indirect)
print("TDA", TDA)
print("gamma_head", gamma)
print("nocc", nocc, "nmo", mo_energy.shape[-1], "nkpts", nkpts)
print("DFT excitation Q=0", eia_dft[:5])
print("SCF energy", mf.e_tot)

bare_chk = get_full_chk("bare", pattern_full)
screen_chk = get_full_chk("screen", pattern_full)
ov_chk = get_ov_chk(pattern_ov)
if ov_chk is None:
    X_bare, W_bare = load_thc(bare_chk, C)
else:
    X_bare, W_bare = load_thc(ov_chk, C)
if unscreen:
    X_screen, W_screen = X_bare, W_bare
else:
    X_screen, W_screen = load_thc(screen_chk, C)
W_screen_fft_neg = get_W_fft_neg(W_screen)
kq = build_kq_map(kmesh)

print("bare THC", bare_chk if ov_chk is None else ov_chk)
print("screen THC", "bare" if unscreen else screen_chk)

if args.indirect:
    q_indirect, e0_all = get_indirect_q(nocc, mo_energy_qp, kq)
    q_int = np.unravel_index(q_indirect, kmesh)
    kq_q = kq[:, q_indirect]
    eia = get_eia_q(nocc, mo_energy_qp, kq_q)
    eia_gw = np.sort(eia.detach().cpu().numpy().reshape(-1))
    print("indirect q", q_indirect, "q_int", q_int)
    print("lowest QP excitation by q", e0_all)
    print()
    print("q", q_indirect, "q_int", q_int)
    print("QP excitation", eia_gw[:nroot])
    e, vec = solve_tda_thc_q(nocc, X_bare, W_bare, X_screen, W_screen_fft_neg, kq_q, q_indirect, eia, gamma)
    print()
    print("singlet", e[:nroot])
    print("binding", eia_gw[0] - e[0])
else:
    eia = get_eia(nocc, mo_energy_qp)
    print("QP excitation Q=0", eia_gw_q0[:nroot])
    e, vec = solve_tda_thc(nocc, X_bare, W_bare, X_screen, W_screen_fft_neg, eia, gamma)
    print()
    print("singlet Q=0", e[:nroot])
