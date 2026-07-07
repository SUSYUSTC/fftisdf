import argparse
import os
import pickle
import numpy as np
import scipy.sparse.linalg
import torch
from pyscf.pbc import df

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


def build_R(mf, kpts_int, kmesh):
    C = np.asarray(mf.mo_coeff)
    R = utils.get_gdf_tensor_compact(mf.with_df, kpts_int, kmesh, C, C, layout="k1k2")
    return torch.from_numpy(R).to(device=device, dtype=complex_dtype)


def build_R_screen(R, eps, kpts_int, kmesh):
    nkpts = int(np.prod(kmesh))
    eps_inv_sqrt = utils.matrix_power(eps, -0.5)
    k1_all = np.arange(nkpts)[:, None]
    k2_all = np.arange(nkpts)[None, :]
    q_int = (kpts_int[k2_all] - kpts_int[k1_all]) % kmesh
    q = np.ravel_multi_index(q_int.reshape(-1, 3).T, kmesh).reshape(nkpts, nkpts)
    eps_inv_sqrt_12 = torch.from_numpy(eps_inv_sqrt[q]).to(device=device, dtype=complex_dtype)
    R_screen = torch.einsum("klxab,klxy->klyab", R, eps_inv_sqrt_12)
    return R_screen


def build_Vovov(nocc, R):
    o, v = get_part(nocc)
    nkpts = R.shape[0]
    R0 = torch.einsum("kkxab->kxab", R)
    Rov = R0[:, :, o, v]
    Vovov = (4.0 / nkpts) * torch.einsum("kxia,lxjb->kliajb", Rov.conj(), Rov)
    return Vovov


def build_Woovv(nocc, R_screen):
    o, v = get_part(nocc)
    nkpts = R_screen.shape[0]
    Roo = R_screen[:, :, :, o, o]
    Rvv = R_screen[:, :, :, v, v]
    Woovv = -(1.0 / nkpts) * torch.einsum("klxij,klxab->klijab", Roo.conj(), Rvv)
    return Woovv


def build_kq_map(kmesh):
    nkpts = int(np.prod(kmesh))
    k = torch.arange(nkpts, dtype=torch.long)
    q = torch.arange(nkpts, dtype=torch.long)
    kq = utils.add_k(k[:, None], q[None, :], kmesh)
    return kq.to(dtype=torch.long)


def build_Vovov_q(nocc, R, kq_q):
    o, v = get_part(nocc)
    nkpts = R.shape[0]
    Rov = R[torch.arange(nkpts, device=R.device), kq_q][:, :, o, v]
    Vq = (4.0 / nkpts) * torch.einsum("kxia,lxjb->kliajb", Rov.conj(), Rov)
    return Vq


def build_Woovv_q(nocc, R_screen, kq_q):
    o, v = get_part(nocc)
    nkpts = R_screen.shape[0]
    Roo = R_screen[:, :, :, o, o]
    Rvv_q = R_screen[kq_q[:, None], kq_q[None, :]][:, :, :, v, v]
    Wq = -(1.0 / nkpts) * torch.einsum("klxij,klxab->klijab", Roo.conj(), Rvv_q)
    return Wq


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


def apply_oovv(H, x):
    return torch.einsum("klijab,ljb->kia", H, x)


def apply_ovov(H, x):
    return torch.einsum("kliajb,ljb->kia", H, x)


def apply_A(Vovov, Woovv, eia, x):
    y = eia * x
    y += apply_oovv(Woovv, x)
    y += 0.5 * apply_ovov(Vovov, x)
    return y


def make_tda_operator(Vovov, Woovv, eia):
    nkpts, _, nocc, nvir = Vovov.shape[:4]
    dim = nkpts * nocc * nvir
    it = 0

    def matvec(x):
        nonlocal it
        print('it', it, end='\r')
        it += 1
        x = torch.from_numpy(x.reshape(nkpts, nocc, nvir)).to(device=device, dtype=complex_dtype)
        y = apply_A(Vovov, Woovv, eia, x)
        return y.reshape(-1).detach().cpu().numpy()

    return scipy.sparse.linalg.LinearOperator((dim, dim), matvec=matvec, dtype=scipy_dtype)


def solve_tda(Vovov, Woovv, eia):
    op = make_tda_operator(Vovov, Woovv, eia)
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
parser.add_argument("-cuda", type=int, default=None)
parser.add_argument("-nroots", type=int, default=1)
parser.add_argument("-suffix", default=None)
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
suffix = args.suffix
use_Edft = args.use_Edft
unscreen = args.unscreen
TDA = True
klabel = system_common.get_klabel(kmesh)
data_dir = system_common.get_data_dir(system, basis, suffix=suffix)
dft_pkl = os.path.join(data_dir, f"DFT_{klabel}.pkl")
gdf_chk = os.path.join(data_dir, f"GDF_{klabel}.chk")
gw_path = os.path.join(data_dir, f"GWenergy_{klabel}.npy")
eps_path = os.path.join(data_dir, f"screening_eps_{klabel}.npy")

with open(dft_pkl, "rb") as f:
    mf = pickle.load(f)
if isinstance(mf.with_df, df.GDF):
    mf.with_df._cderi = gdf_chk

kpts = np.asarray(mf.kpts)
kpts_int = get_kpts_int(mf.cell, kpts, kmesh)
mo_energy = np.load(gw_path)
eps = np.load(eps_path)
mo_energy_qp = np.asarray(mf.mo_energy) if use_Edft else mo_energy
nocc = mf.cell.nelectron // 2
nkpts = len(kpts)

R = build_R(mf, kpts_int, kmesh)
R_screen = R if unscreen else build_R_screen(R, eps, kpts_int, kmesh)
kq = build_kq_map(kmesh).to(device=device)

print("kmesh", kmesh)
print("basis", basis)
print("cuda", args.cuda)
print("nroot", nroot)
print("use_Edft", use_Edft)
print("unscreen", unscreen)
print("indirect", args.indirect)
print("TDA", TDA)
print("nocc", nocc, "nmo", mo_energy_qp.shape[-1], "nkpts", nkpts)
print("SCF energy", mf.e_tot)

if args.indirect:
    q_indirect, e0_all = get_indirect_q(nocc, mo_energy_qp, kq)
    q_int = np.unravel_index(q_indirect, kmesh)
    kq_q = kq[:, q_indirect]
    Vovov = build_Vovov_q(nocc, R, kq_q)
    Woovv = build_Woovv_q(nocc, R_screen, kq_q)
    eia = get_eia_q(nocc, mo_energy_qp, kq_q)
    eia_gw = np.sort(eia.detach().cpu().numpy().reshape(-1))
    print("indirect q", q_indirect, "q_int", q_int)
    print("lowest QP excitation by q", e0_all)
    print()
    print("q", q_indirect, "q_int", q_int)
    print("QP excitation", eia_gw[:nroot])
    e, vec = solve_tda(Vovov, Woovv, eia)
    print()
    print("singlet", e[:nroot])
    print("binding", eia_gw[0] - e[0])
else:
    Vovov = build_Vovov(nocc, R)
    Woovv = build_Woovv(nocc, R_screen)
    eia = get_eia(nocc, mo_energy_qp)
    eia_gw = mo_energy_qp[:, None, nocc:] - mo_energy_qp[:, :nocc, None]
    eia_gw = np.sort(eia_gw.reshape(-1))
    print("QP excitation Q=0", eia_gw[:nroot])
    e, vec = solve_tda(Vovov, Woovv, eia)
    print()
    print("singlet Q=0", e[:nroot])
