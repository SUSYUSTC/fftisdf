import argparse
import os
import pickle
import time
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import torch
from pyscf.pbc import df
from pyscf.pbc.df import rsdf_builder

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
    C = torch.from_numpy(np.asarray(mf.mo_coeff)).to(device=device, dtype=complex_dtype)
    R_ao = torch.from_numpy(utils.get_gdf_tensor_compact(mf.with_df, kpts_int, kmesh, layout="k1k2")).to(device=device, dtype=complex_dtype)
    R = torch.einsum('pqxij,pia,qjb->pqxab', R_ao, C.conj(), C)
    return R


def get_q0_charge_vector(mf):
    auxcell = df.make_auxcell(mf.cell, mf.with_df.auxbasis)
    builder = rsdf_builder._RSGDFBuilder(mf.cell, auxcell, mf.kpts).build()
    builder.mesh = mf.with_df.mesh
    builder.linear_dep_threshold = mf.with_df.linear_dep_threshold
    j2c = builder.get_2c2e(np.zeros((1, 3)))[0]
    chol = scipy.linalg.cholesky(j2c, lower=True)
    charge = rsdf_builder._gaussian_int(auxcell)
    charge = charge / np.linalg.norm(charge)
    u = chol.T.conj() @ charge
    u = u / np.linalg.norm(u)
    return torch.from_numpy(u).to(device=device, dtype=complex_dtype)


def project_q0_charge_from_R(R, u):
    R_proj = R.clone()
    nkpts = R.shape[0]
    idx = torch.arange(nkpts, device=device)
    R0 = R_proj[idx, idx]
    coeff = torch.einsum("x,kxab->kab", u.conj(), R0)
    R_proj[idx, idx] -= torch.einsum("x,kab->kxab", u, coeff)
    before = torch.linalg.norm(R0).item()
    after = torch.linalg.norm(R_proj[idx, idx]).item()
    print("project q=0 charge from W R", before, after)
    return R_proj


def build_R_screen(R, eps_inv_ext, kpts_int, kmesh, nofc=False):
    nkpts = int(np.prod(kmesh))
    nao = R.shape[-1]
    if nofc:
        eps_inv = eps_inv_ext[:, 1:, 1:]
        R_ext = R
    else:
        eps_inv = eps_inv_ext
        R_ext = torch.zeros((nkpts, nkpts, R.shape[2] + 1, nao, nao), dtype=complex_dtype, device=device)
        R_ext[:, :, 1:] = R
        eye = torch.eye(nao, dtype=complex_dtype, device=device)
        idx = torch.arange(nkpts, device=device)
        R_ext[idx, idx, 0] = eye

    eps_inv_half = utils.matrix_power(eps_inv, 0.5)
    k1_all = np.arange(nkpts)[:, None]
    k2_all = np.arange(nkpts)[None, :]
    q_int = (kpts_int[k1_all] - kpts_int[k2_all]) % kmesh
    q = np.ravel_multi_index(q_int.reshape(-1, 3).T, kmesh).reshape(nkpts, nkpts)
    eps_inv_half_12 = torch.from_numpy(eps_inv_half[q]).to(device=device, dtype=complex_dtype)
    R_screen = torch.einsum("klyx,klxab->klyab", eps_inv_half_12, R_ext)
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
    idx = torch.argmin(eia)
    v0 = torch.zeros_like(eia.reshape((1, -1)))
    v0[0, idx] = 1.0
    op = make_tda_operator(Vovov, Woovv, eia)
    e, x = scipy.sparse.linalg.eigsh(op, k=nroot, which="SA", tol=eig_tol, v0=v0)
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
eig_tol = 1e-5

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
parser.add_argument("--nofc", action="store_true")
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
nofc = args.nofc
TDA = True
klabel = system_common.get_klabel(kmesh)
data_dir = system_common.get_data_dir(system, basis, suffix=suffix)
dft_pkl = os.path.join(data_dir, f"DFT_{klabel}.pkl")
gdf_chk = os.path.join(data_dir, f"GDF_{klabel}.chk")
gw_path = os.path.join(data_dir, f"GWenergy_{klabel}.npy")
eps_path = os.path.join(data_dir, f"screening_eps_ext_{klabel}.npy")

with open(dft_pkl, "rb") as f:
    mf = pickle.load(f)
if isinstance(mf.with_df, df.GDF):
    mf.with_df._cderi = gdf_chk

kpts = np.asarray(mf.kpts)
kpts_int = get_kpts_int(mf.cell, kpts, kmesh)
mo_energy = np.load(gw_path)
eps_inv_ext = np.load(eps_path)
mo_energy_qp = np.asarray(mf.mo_energy) if use_Edft else mo_energy
nocc = mf.cell.nelectron // 2
nvir = mf.cell.nao - nocc
nkpts = len(kpts)
project_q0_charge = system_common.load_section_setting(system, basis, "bse", "project_q0_charge", suffix=suffix, default=False)

t1 = time.time()
R = build_R(mf, kpts_int, kmesh)
t2 = time.time()
print('build R time', t2 - t1)
R_for_W = R
if project_q0_charge and not nofc:
    u_charge = get_q0_charge_vector(mf)
    R_for_W = project_q0_charge_from_R(R, u_charge)
R_screen = R_for_W if unscreen else build_R_screen(R_for_W, eps_inv_ext, kpts_int, kmesh, nofc=nofc)
t3 = time.time()
print('build R_screen time', t3 - t2)
kq = build_kq_map(kmesh).to(device=device)

print("kmesh", kmesh)
print("basis", basis)
print("cuda", args.cuda)
print("nroot", nroot)
print("use_Edft", use_Edft)
print("unscreen", unscreen)
print("nofc", nofc)
print("project_q0_charge", project_q0_charge)
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
    eia_gw_flatten = eia_gw.reshape(-1)
    idx = np.argsort(eia_gw_flatten)[:nroot]
    print("QP excitation Q=0", eia_gw_flatten[idx])
    idx_k, idx_i, idx_a = np.unravel_index(idx, eia_gw.shape)
    V_diag = Vovov[idx_k, idx_k, idx_i, idx_a, idx_i, idx_a].numpy()
    W_diag = Woovv[idx_k, idx_k, idx_i, idx_i, idx_a, idx_a].numpy()
    total_diag = (eia_gw_flatten[idx] + W_diag - 0.5 * V_diag).real
    print("diagonal corrected Q=0", total_diag)
    e_q0 = solve_tda(Vovov[0:1, 0:1], Woovv[0:1, 0:1], eia[0:1])[0]
    print("singlet solved at Q=0 only", e_q0[:nroot])
    e, vec = solve_tda(Vovov, Woovv, eia)
    print()
    print("singlet Q=0", e[:nroot])
