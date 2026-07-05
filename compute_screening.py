import sys
import os
import pickle
import numpy as np
from pyscf import lib
import utils
import system_common


system = sys.argv[1]
kmesh = (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
klabel = system_common.get_klabel(kmesh)
basis = sys.argv[5]

data_dir = system_common.get_data_dir(system, basis)
gdf_chk = os.path.join(data_dir, f"GDF_{klabel}.chk")
dft_pkl = os.path.join(data_dir, f"DFT_{klabel}.pkl")
gw_path = os.path.join(data_dir, f"GWenergy_{klabel}.npy")
screening_path = os.path.join(data_dir, f"screening_eps_{klabel}.npy")


def get_kpts_int(cell, kpts, kmesh):
    kpts_int = np.round(cell.get_scaled_kpts(kpts) * kmesh).astype(int) % kmesh
    assert utils.is_k_ordered(kpts_int, kmesh)
    return kpts_int


def get_kpt_map(kpts_int):
    return {tuple(k): i for i, k in enumerate(kpts_int)}


def load_cderi_block(cderi, kpts, ki, kj, nao):
    Luv = cderi.load(kpts[ki], kpts[kj])
    if Luv.shape[-1] == nao * (nao + 1) // 2:
        Luv = lib.unpack_tril(Luv)
    return Luv.reshape(-1, nao, nao)


def get_Lia_block(cderi, kpts, mo_coeff, ki, kj, nocc):
    nao = mo_coeff.shape[1]
    Luv = load_cderi_block(cderi, kpts, ki, kj, nao)
    Co = mo_coeff[ki, :, :nocc]
    Cv = mo_coeff[kj, :, nocc:]
    Lia = np.einsum("Puv,ui,va->Pia", Luv, Co.conj(), Cv, optimize=True)
    return Lia


with open(dft_pkl, "rb") as f:
    mf = pickle.load(f)
mf.with_df._cderi = gdf_chk
gw_energy = np.load(gw_path)

kpts = np.asarray(mf.kpts)
kpts_int = get_kpts_int(mf.cell, kpts, kmesh)
kpt_map = get_kpt_map(kpts_int)

nkpts = len(kpts)
mo_coeff = np.asarray(mf.mo_coeff)
mo_energy = np.asarray(gw_energy)
nocc = int(round(np.sum(mf.mo_occ[0]) / 2.0))
cderi = mf.with_df.cderi_array()
naux = cderi.load(kpts[0], kpts[0]).shape[0]

eps = np.zeros((nkpts, naux, naux), dtype=np.complex128)
for kq in range(nkpts):
    q_int = kpts_int[kq]
    X = np.zeros((naux, naux), dtype=np.complex128)
    for ki in range(nkpts):
        kj_int = (kpts_int[ki] - q_int) % kmesh
        kj = kpt_map[tuple(kj_int)]

        Lia = get_Lia_block(cderi, kpts, mo_coeff, ki, kj, nocc)
        eia_inv = 1.0 / (mo_energy[ki, :nocc, None] - mo_energy[kj, None, nocc:])
        X += (4.0 / nkpts) * np.einsum("Pia,ia,Qia->PQ", Lia, eia_inv, Lia.conj(), optimize=True)

    X = 0.5 * (X + X.conj().T)
    eps[kq] = np.eye(naux) - X
    print("kq", kq, "q_int", q_int, "eps norm", np.linalg.norm(eps[kq]), flush=True)
np.save(screening_path, eps)
print("Saved screening eps =", screening_path, flush=True)
