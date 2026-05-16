import sys
import pickle
import numpy as np
import torch
from pyscf.pbc import tools
from pyscf.lo import orth
import utils


def relative_cell_index(kmesh):
    i = torch.arange(int(np.prod(kmesh)))
    rel = utils.add_k(i[:, None], utils.negative_k(i[None, :], kmesh), kmesh)
    return rel.numpy()


def unitary_err(C1, C2):
    U = np.linalg.solve(C1, C2)
    err = np.linalg.norm(U @ U.swapaxes(-1, -2).conj() - np.eye(C1.shape[-1]))
    return err


def make_real_mo_block(A, S):
    D = A @ A.conj().T
    assert np.linalg.norm(D.imag) < 1e-6
    w, X = np.linalg.eigh(D.real)
    B = X[:, -A.shape[1]:] * np.sqrt(w[-A.shape[1]:])[None, :]
    U = A.conj().T @ S @ B
    assert np.linalg.norm(U @ U.conj().T - np.eye(A.shape[1])) < 1e-6
    return B


def orthonormalize_mo(A, S):
    G = A.conj().T @ S @ A
    w, X = np.linalg.eigh(G)
    assert w.min() > 1e-12
    G_isqrt = (X * (1.0 / np.sqrt(w))[None, :]) @ X.conj().T
    return A @ G_isqrt


def make_ao_orth(S):
    w, X = np.linalg.eigh(S)
    assert w.min() > 1e-12
    Xh = X.swapaxes(-1, -2).conj()
    S_mhalf = (X * (1.0 / np.sqrt(w))[:, None, :]) @ Xh
    S_half = (X * np.sqrt(w)[:, None, :]) @ Xh
    return S_mhalf, S_half


def ao_to_orth(C, S_half):
    return np.einsum("kab,kbi->kai", S_half, C)


def orth_to_ao(C, S_mhalf):
    return np.einsum("kab,kbi->kai", S_mhalf, C)


def make_C_adjoint(C):
    C = C.copy()
    for k in range(nk_tot):
        C[k] = orthonormalize_mo(C[k], S_k[k])
    C_init = C.copy()
    kidx = np.arange(nk_tot)
    for i, j in zip(kidx, utils.negative_k(kidx, kmesh).numpy()):
        if i == j:
            C[i, :, :nocc] = make_real_mo_block(C[i, :, :nocc], S_k[i])
            C[i, :, nocc:] = make_real_mo_block(C[i, :, nocc:], S_k[i])
            continue
        assert unitary_err(C[i].conj(), C[j]) < 1e-8
        C[j] = C[i].conj()
    U = np.einsum('kai,kbj,kab->kij', C_init.conj(), C, S_k)
    err = np.linalg.norm(U @ U.swapaxes(-1, -2).conj() - np.eye(nao))
    assert err < 1e-5
    return C


def torch_cayley(A):
    I = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    return torch.linalg.solve(I - 0.5 * A, I + 0.5 * A)


def torch_k2RT(C_k, force_real=None):
    eye = torch.eye(nk_tot, dtype=C_k.dtype, device=C_k.device)
    C_k_full = (C_k[:, None, :, :] * eye[:, :, None, None]).swapaxes(1, 2)
    C_RT = utils.k2R(C_k_full, axis=(0, 2), dual=(False, True), kmesh=kmesh)
    if force_real is None:
        force_real = real_gauge
    if force_real:
        return C_RT.real
    return C_RT


def torch_RT2k(C_RT):
    C_k_full = utils.R2k(C_RT, axis=(0, 2), dual=(False, True), kmesh=kmesh)
    kidx = torch.arange(nk_tot, device=C_RT.device)
    return C_k_full[kidx, :, kidx, :]


def torch_Uk_to_uRT(U_k):
    u_kk = torch.zeros((nk_tot, U_k.shape[1], nk_tot, U_k.shape[2]), dtype=U_k.dtype, device=U_k.device)
    kidx = torch.arange(nk_tot, device=U_k.device)
    u_kk[kidx, :, kidx, :] = U_k
    u_RT = utils.k2R(u_kk, axis=(0, 2), dual=(False, True), kmesh=kmesh)
    u_RT = u_RT.reshape(nk_tot * U_k.shape[1], nk_tot * U_k.shape[2])
    if real_gauge:
        return u_RT.real
    return u_RT


def torch_ordered_translation_error(C_RT, rel_t):
    err = torch.linalg.norm(C_RT - C_RT[rel_t, :, 0, :].swapaxes(1, 2))
    return err


def make_pop_ortho(mol, S_mhalf):
    C_k_orth = S_mhalf
    C_k_orth_t = torch.tensor(C_k_orth, dtype=torch.complex128)
    C_RT_orth = torch_k2RT(C_k_orth_t, force_real=True).numpy()
    C_orth = C_RT_orth.reshape(nk_tot * nao, nk_tot * nao)

    s = mol.pbc_intor("int1e_ovlp", hermi=1)
    csc = C_orth.T @ s @ orth.orth_ao(mol, "meta_lowdin", "ANO", s=s)
    atom_idx = np.arange(cell.natm)
    offsets = mol.offset_nr_by_atom()
    pop = np.empty((len(atom_idx), nk_tot * nao, nk_tot * nao))
    for x, i in enumerate(atom_idx):
        b0, b1, p0, p1 = offsets[i]
        pop[x] = csc[:, p0:p1] @ csc[:, p0:p1].T
    return pop


class KPointPM:
    def __init__(self, pop_ortho, C_k_init):
        self.nk_tot, self.nao, self.nmo = C_k_init.shape
        C_k_init_t = torch.tensor(C_k_init, dtype=torch.complex128)
        C_RT_init = torch_k2RT(C_k_init_t)
        C_flatten_init = C_RT_init.numpy().reshape(self.nk_tot * self.nao, self.nk_tot * self.nmo)
        proj = np.einsum("ui,xuv,vj->xij", C_flatten_init, pop_ortho, C_flatten_init, optimize=True)

        self.C_k_init = C_k_init_t
        if real_gauge:
            self.proj = torch.tensor(proj, dtype=torch.float64)
        else:
            self.proj = torch.tensor(proj, dtype=torch.complex128)
        self.proj_k = utils.R2k(self.proj.to(torch.complex128).reshape(-1, self.nk_tot, self.nmo, self.nk_tot, self.nmo), axis=(1, 3), dual=(True, True), kmesh=kmesh)
        self.pm_const = torch.tensor(float(np.real(self.nk_tot * np.einsum("xii->", proj))), dtype=torch.float64)
        self.neg_idx = utils.negative_k(np.arange(self.nk_tot), kmesh)
        self.exponent = 2

    @property
    def nparams(self):
        return 2 * self.nk_tot * self.nmo * self.nmo

    def get_A(self, A_flatten):
        A = A_flatten.reshape(2, self.nk_tot, self.nmo, self.nmo)
        A = A[0] + 1j * A[1]
        A = 0.5 * (A - A.conj().swapaxes(-1, -2))
        if real_gauge:
            A = 0.5 * (A + A[self.neg_idx].conj())
        return A

    def get_U_k(self, A_flatten):
        A = self.get_A(A_flatten)
        return torch_cayley(A)

    def pm_function(self, A_flatten):
        return self.pm_const - self.cost_function(A_flatten)

    def cost_function_R(self, A_flatten):
        U_k = self.get_U_k(A_flatten)
        u_RT = torch_Uk_to_uRT(U_k)
        q = torch.einsum("xij,ia,ja->xa", self.proj, u_RT.conj(), u_RT).real
        pm = self.nk_tot * torch.sum(q ** 2)
        return self.pm_const - pm
    
    #@profile
    def cost_function_k(self, A_flatten):
        U_k = self.get_U_k(A_flatten)
        q_k = torch.einsum('xpiqj,pia,qja->xpqa', self.proj_k, U_k, U_k)
        q_RT = utils.k2R(q_k, axis=(1, 2), dual=(True, True), kmesh=kmesh).real
        q = torch.einsum('xppa->xpa', q_RT).reshape((-1, self.nk_tot * self.nmo))
        pm = self.nk_tot * torch.sum(q ** 2)
        return self.pm_const - pm

    def cost_function(self, A_flatten):
        if use_k:
            return self.cost_function_k(A_flatten)
        else:
            return self.cost_function_R(A_flatten)

    def run(self, tol=1e-5, max_iter=200):
        shape = (2, self.nk_tot, self.nmo, self.nmo)
        #A0 = torch.zeros(shape, dtype=torch.float64)
        A0 = torch.randn(shape, dtype=torch.float64) * 1e-4
        A = torch.nn.Parameter(A0.reshape(-1))
        #opt = torch.optim.Adam([A], lr=5e-3, betas=(0.9, 0.99))
        #opt = torch.optim.Rprop([A], lr=1e-3, etas=(0.5, 1.2))
        opt = torch.optim.LBFGS([A], max_iter=5, history_size=10, line_search_fn="strong_wolfe")
        history = []

        for it in range(max_iter):
            def closure():
                opt.zero_grad()
                val = self.cost_function(A)
                val.backward()
                return val

            val = opt.step(closure)
            loss = self.cost_function(A).item()
            pm_value = self.pm_const.item() - loss
            history.append(loss)
            res = np.std(history[-5:]) / abs(np.mean(history[-5:])) if len(history) > 5 else np.inf
            print(f"  iter {it:4d}  loss {loss:.12e}  PM {pm_value:.12e}  res {res:.6e}")
            if res < tol:
                break

        with torch.no_grad():
            U_k = self.get_U_k(A)
            C_k = torch.einsum("kui,kij->kuj", self.C_k_init, U_k)
        return C_k.numpy()


def check_result(C_k_init, C_k_local):
    nmo = C_k_init.shape[-1]
    C_k_init = torch.tensor(C_k_init, dtype=torch.complex128)
    C_k_local = torch.tensor(C_k_local, dtype=torch.complex128)
    S_k_t = torch.tensor(np.asarray(S_k), dtype=torch.complex128)
    rel_t = torch.tensor(rel, dtype=torch.long)

    C_RT_local = torch_k2RT(C_k_local)
    trans_err = torch_ordered_translation_error(C_RT_local, rel_t)

    Sk_loc = torch.einsum("kui,kuv,kvj->kij", C_k_local.conj(), S_k_t, C_k_local)
    Sk_err = torch.max(torch.abs(Sk_loc - torch.eye(nmo, dtype=torch.complex128)[None, :, :]))

    U_k = torch.einsum("kui,kuv,kvj->kij", C_k_init.conj(), S_k_t, C_k_local)
    Uk_err = torch.max(torch.abs(torch.einsum("kji,kjl->kil", U_k.conj(), U_k) - torch.eye(nmo, dtype=torch.complex128)[None, :, :]))
    C_k_from_ref = torch.einsum("kui,kij->kuj", C_k_init, U_k)
    Ck_subspace_err = torch.max(torch.abs(C_k_from_ref - C_k_local))

    print(f"  translation error: {trans_err.item():.2e}")
    print(f"  max k-space overlap error: {Sk_err.item():.2e}")
    print(f"  max per-k U^H U error: {Uk_err.item():.2e}")
    print(f"  max C(k) U(k) - C_k_loc error: {Ck_subspace_err.item():.2e}")


kmesh = (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
basis = sys.argv[4]
ke_cutoff = 40.0
klabel = f"{kmesh[0]}x{kmesh[1]}x{kmesh[2]}"
real_gauge = True
use_k = True

scf_pkl = f"data/SCF_diamond_{klabel}_{basis}_ke{ke_cutoff}.pkl"
with open(scf_pkl, "rb") as f:
    mf = pickle.load(f)

cell = mf.cell
nocc = cell.nelectron // 2
nao = cell.nao_nr()
nk_tot = int(np.prod(kmesh))

print(f"kmesh = {klabel}, basis = {basis}")
print(f"nao = {nao}, nocc = {nocc}")

scell = tools.super_cell(cell, kmesh)
rel = relative_cell_index(kmesh)
S_k_ao = cell.pbc_intor("int1e_ovlp", kpts=mf.kpts)
S_mhalf, S_half = make_ao_orth(S_k_ao)
S_k = np.tile(np.eye(nao)[None, :, :], (nk_tot, 1, 1))
pop_ortho = make_pop_ortho(scell, S_mhalf)

C = np.array(mf.mo_coeff)
C = ao_to_orth(C, S_half)
C_adjoint = make_C_adjoint(C)

print("")
print("K-point PM occupied")
C_k_occ_init = C_adjoint[:, :, :nocc]
kpm_occ = KPointPM(pop_ortho, C_k_occ_init)

C_k_occ_local = kpm_occ.run()
check_result(C_k_occ_init, C_k_occ_local)

print("")
print("K-point PM virtual")
C_k_vir_init = C_adjoint[:, :, nocc:]
kpm_vir = KPointPM(pop_ortho, C_k_vir_init)

C_k_vir_local = kpm_vir.run()
check_result(C_k_vir_init, C_k_vir_local)

C_k_occ_local = orth_to_ao(C_k_occ_local, S_mhalf)
C_k_vir_local = orth_to_ao(C_k_vir_local, S_mhalf)

data = {
    "occ": C_k_occ_local,
    "vir": C_k_vir_local,
}
save_path = f"data/Clocal_kPM_diamond_{klabel}_{basis}_ke{ke_cutoff}.npy"
np.save(save_path, data)
print("")
print("Localized k-space orbitals")
print(f"path: {save_path}")
