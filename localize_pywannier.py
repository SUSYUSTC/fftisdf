import sys
import os
import pickle
import types
import numpy as np
from pyscf.pbc.tools import pywannier90
from pyscf.pbc.tools.k2gamma import get_phase


def cell_labels(i, kmesh):
    i = np.asarray(i)
    kmesh = np.asarray(kmesh)
    return np.stack([i // (kmesh[1] * kmesh[2]), (i % (kmesh[1] * kmesh[2])) // kmesh[2], i % kmesh[2]], axis=-1)


def make_kmf_part(mf, p0, p1):
    kmf = types.SimpleNamespace()
    kmf.kpts = mf.kpts
    kmf.mo_energy_kpts = [np.asarray(e)[p0:p1] for e in mf.mo_energy_kpts]
    kmf.mo_coeff_kpts = [np.asarray(c)[:, p0:p1] for c in mf.mo_coeff_kpts]
    kmf.mo_energy = kmf.mo_energy_kpts
    kmf.mo_coeff = kmf.mo_coeff_kpts
    return kmf


def run_w90_part(mf, kmesh, p0, p1, name):
    num_wann = p1 - p0
    kmf = make_kmf_part(mf, p0, p1)
    w90 = pywannier90.W90(kmf, mf.cell, kmesh, num_wann)
    w90.use_bloch_phases = True
    print("")
    print(f"pyWannier90 {name}")
    print(f"  bands {p0}:{p1}")
    w90.kernel()
    print(f"  spread {w90.spread}")
    print(f"  wann_spreads {w90.wann_spreads}")
    print(f"  wann_centres")
    print(w90.wann_centres)

    C_k = []
    for k in range(w90.num_kpts_loc):
        mo = w90.mo_coeff_kpts[k][:, w90.band_included_list]
        win = w90.lwindow[k]
        C_opt = mo[:, win] @ w90.U_matrix_opt[k][:, win].T
        C_tilde = C_opt @ w90.U_matrix[k].T
        C_k.append(C_tilde)
    return np.asarray(C_k)


kmesh = np.array([int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])])
basis = sys.argv[4]

ke_cutoff = 40.0
klabel = f"{kmesh[0]}x{kmesh[1]}x{kmesh[2]}"
scf_pkl = f"data_GDF/SCF_diamond_{klabel}_{basis}_ke{ke_cutoff}.pkl"
if not os.path.exists(scf_pkl):
    scf_pkl = f"data/SCF_diamond_{klabel}_{basis}_ke{ke_cutoff}.pkl"

with open(scf_pkl, "rb") as f:
    mf = pickle.load(f)

cell = mf.cell
nocc = cell.nelectron // 2
nao = cell.nao_nr()
nk_tot = int(np.prod(kmesh))
_, phase = get_phase(cell, mf.kpts, kmesh)

print(f"kmesh = {klabel}, basis = {basis}")
print(f"nao = {nao}, nocc = {nocc}")

C_occ = run_w90_part(mf, kmesh, 0, nocc, "occ")
C_vir = run_w90_part(mf, kmesh, nocc, nao, "vir")

S_k = cell.pbc_intor("int1e_ovlp", kpts=mf.kpts)
for name, C in [("occ", C_occ), ("vir", C_vir)]:
    nmo = C.shape[-1]
    ovlp = np.einsum("kui,kuv,kvj->kij", C.conj(), S_k, C)
    err = np.max(np.abs(ovlp - np.eye(nmo)[None, :, :]))
    print("")
    print(f"{name} max S-orth error {err:.3e}")

data = {
    "occ": C_occ,
    "vir": C_vir,
}
save_path = f"data/Clocal_pywannier_diamond_{klabel}_{basis}_ke{ke_cutoff}.npy"
np.save(save_path, data)
print("")
print("pyWannier90 localized k-space orbitals")
print(f"path: {save_path}")
