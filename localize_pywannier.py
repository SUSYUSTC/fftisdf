import argparse
import os
import pickle
import types
import numpy as np
import pyscf.pbc.tools.pywannier90 as pywannier90

import system_common


def make_kmf_part(mf, p0, p1):
    kmf = types.SimpleNamespace()
    kmf.kpts = mf.kpts
    kmf.mo_energy_kpts = [np.asarray(e)[p0:p1] for e in mf.mo_energy]
    kmf.mo_coeff_kpts = [np.asarray(c)[:, p0:p1] for c in mf.mo_coeff]
    kmf.mo_occ_kpts = [np.asarray(o)[p0:p1] for o in mf.mo_occ]
    kmf.mo_energy = kmf.mo_energy_kpts
    kmf.mo_coeff = kmf.mo_coeff_kpts
    kmf.mo_occ = kmf.mo_occ_kpts
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
    print(f"  wann_spreads")
    print(w90.wann_spreads)
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


def check_s_orthonormal(cell, kpts, C, name):
    S_k = cell.pbc_intor("int1e_ovlp", kpts=kpts)
    nmo = C.shape[-1]
    ovlp = np.einsum("kui,kuv,kvj->kij", C.conj(), S_k, C, optimize=True)
    err = np.max(np.abs(ovlp - np.eye(nmo)[None, :, :]))
    print(f"{name} max S-orth error {err:.3e}")


parser = argparse.ArgumentParser()
parser.add_argument("system")
parser.add_argument("kx", type=int)
parser.add_argument("ky", type=int)
parser.add_argument("kz", type=int)
parser.add_argument("basis")
parser.add_argument("-suffix", default=None)
args = parser.parse_args()

system = args.system
kmesh = (args.kx, args.ky, args.kz)
basis = args.basis
suffix = args.suffix
klabel = system_common.get_klabel(kmesh)
data_dir = system_common.get_data_dir(system, basis, suffix=suffix)
dft_pkl = os.path.join(data_dir, f"DFT_{klabel}.pkl")
save_path = os.path.join(data_dir, f"Clocal_pywannier_{klabel}.npy")

with open(dft_pkl, "rb") as f:
    mf = pickle.load(f)

os.chdir(data_dir)

cell = mf.cell
C = np.asarray(mf.mo_coeff)
occ = np.asarray(mf.mo_occ)
nocc_all = np.count_nonzero(occ > 0, axis=1)
nocc = int(nocc_all[0])
nmo = C.shape[-1]

print(f"system = {system}")
print(f"kmesh = {klabel}")
print(f"basis = {basis}")
print(f"suffix = {suffix}")
print(f"data_dir = {data_dir}")
print(f"nao = {cell.nao_nr()}, nmo = {nmo}, nocc = {nocc}, nvir = {nmo - nocc}")
print(f"nocc by k = {nocc_all}")

C_occ = run_w90_part(mf, kmesh, 0, nocc, "occ")
C_vir = run_w90_part(mf, kmesh, nocc, nmo, "vir")

print("")
check_s_orthonormal(cell, mf.kpts, C_occ, "occ")
check_s_orthonormal(cell, mf.kpts, C_vir, "vir")

data = {
    "occ": C_occ,
    "vir": C_vir,
}
np.save(save_path, data)
print("")
print("pyWannier90 localized k-space orbitals")
print(f"path: {save_path}")
