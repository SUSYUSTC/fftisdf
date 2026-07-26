import os
import pickle
import argparse

import numpy as np
import scipy.linalg
import torch
from pyscf.pbc import df
import libsymm
import system_common


def get_symmetry_error(symm, A):
    A = torch.from_numpy(np.asarray(A)).to(dtype=torch.complex128)
    A_avg = libsymm.symmetrize_ao_operator(symm, A)
    return (torch.linalg.norm(A_avg - A) / torch.linalg.norm(A)).detach().cpu().numpy()


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
gdf_chk = os.path.join(data_dir, f"GDF_{klabel}.chk")
mf_pkl = os.path.join(data_dir, f"DFT_{klabel}.pkl")
out_pkl = os.path.join(data_dir, f"DFT_{klabel}_symm.pkl")

with open(mf_pkl, "rb") as f:
    mf = pickle.load(f)

cell = mf.cell
kpts = np.asarray(mf.kpts)
if getattr(mf, "with_df", None) is None:
    mf.with_df = df.GDF(cell, kpts)
if getattr(mf.with_df, "_cderi", None) is None:
    mf.with_df._cderi = gdf_chk

dm0 = np.asarray(mf.make_rdm1())
fock0 = np.asarray(mf.get_fock(dm=dm0))
S = np.asarray(mf.get_ovlp())

symm = libsymm.PBCSymmetry(cell, kmesh, kpts)

fock0_t = torch.from_numpy(fock0).to(dtype=torch.complex128)
fock_sym = libsymm.symmetrize_ao_operator(symm, fock0_t).detach().cpu().numpy()
fock_change = np.linalg.norm(fock_sym - fock0) / np.linalg.norm(fock0)
fock_symm_err0 = fock_change

mo_energy_sym = []
mo_coeff_sym = []
for k in range(len(kpts)):
    e, c = scipy.linalg.eigh(fock_sym[k], S[k])
    mo_energy_sym.append(e)
    mo_coeff_sym.append(c)
mo_energy_sym = np.asarray(mo_energy_sym)
mo_coeff_sym = np.asarray(mo_coeff_sym)
mo_occ_sym = np.asarray(mf.get_occ(mo_energy_kpts=mo_energy_sym, mo_coeff_kpts=mo_coeff_sym))

dm1 = np.asarray(mf.make_rdm1(mo_coeff_sym, mo_occ_sym))
fock1 = np.asarray(mf.get_fock(dm=dm1))
dm_change = np.linalg.norm(dm1 - dm0) / np.linalg.norm(dm0)
dm_symm_err0 = get_symmetry_error(symm, dm0)
dm_symm_err1 = get_symmetry_error(symm, dm1)
fock_symm_err1 = get_symmetry_error(symm, fock1)

mf.mo_energy = mo_energy_sym
mf.mo_coeff = mo_coeff_sym
mf.mo_occ = mo_occ_sym

with open(out_pkl, "wb") as f:
    pickle.dump(mf, f)

print("input mf =", mf_pkl)
print("output mf =", out_pkl)
print("gdf chk =", gdf_chk)
print("system =", system)
print("kmesh =", kmesh)
print("basis =", basis)
print("suffix =", suffix)
print("rel_fro Fock difference =", fock_change)
print("rel_fro DM difference =", dm_change)
print("old Fock symm error =", fock_symm_err0)
print("new Fock symm error =", fock_symm_err1)
print("old DM symm error =", dm_symm_err0)
print("new DM symm error =", dm_symm_err1)
