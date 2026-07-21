import os
import pickle
import argparse

import numpy as np
import scipy.linalg
import pyscf
pyscf.scf.hf.remove_overlap_zero_eigenvalue = False
from pyscf.pbc import scf, df
from pyscf.pbc.symm import symmetry
import system_common


def build_symmetry_kpts(cell, kmesh):
    cell_sym = cell.copy()
    cell_sym.space_group_symmetry = True
    cell_sym.symmorphic = False
    cell_sym.build(False, False)
    kpts = cell_sym.make_kpts(
        kmesh,
        space_group_symmetry=True,
        time_reversal_symmetry=False,
    )
    return cell_sym, kpts


def build_kpt_map(kpts_scaled0, kpts_scaled1):
    perm = np.empty(len(kpts_scaled0), dtype=np.int64)
    kpts_scaled1 = np.mod(kpts_scaled1, 1.0)
    for i, kpt in enumerate(np.mod(kpts_scaled0, 1.0)):
        diff = kpts_scaled1 - kpt[None, :]
        diff -= np.rint(diff)
        err = np.abs(diff).max(axis=1)
        perm[i] = np.argmin(err)
        assert err[perm[i]] < 1e-10
    return perm


def symmetrize_dm(cell, kpts, dm):
    nkpts = len(dm)
    dm_avg = np.zeros_like(dm)
    count = np.zeros(nkpts, dtype=np.int64)
    for k in range(nkpts):
        for iop, op in enumerate(kpts.ops):
            kp = kpts.k2opk[k, iop]
            if kp < 0:
                continue
            dm_avg[kp] += symmetry.transform_dm(
                cell,
                kpts.kpts_scaled[k],
                dm[k],
                op,
                kpts.Dmats[iop],
            )
            count[kp] += 1
    dm_avg /= count[:, None, None]
    return dm_avg


def symmetrize_fock(cell, kpts, fock):
    nkpts = len(fock)
    fock_avg = np.zeros_like(fock)
    count = np.zeros(nkpts, dtype=np.int64)
    for k in range(nkpts):
        for iop, op in enumerate(kpts.ops):
            kp = kpts.k2opk[k, iop]
            if kp < 0:
                continue
            fock_avg[kp] += symmetry.transform_1e_operator(
                cell,
                kpts.kpts_scaled[k],
                fock[k],
                op,
                kpts.Dmats[iop],
            )
            count[kp] += 1
    fock_avg /= count[:, None, None]
    return fock_avg


def get_dm_symmetry_error(cell, kmesh, kpts, dm):
    cell_sym, kpts_symm = build_symmetry_kpts(cell, kmesh)
    perm_sym_to_mf = build_kpt_map(kpts_symm.kpts_scaled, cell.get_scaled_kpts(kpts))
    perm_mf_to_sym = np.argsort(perm_sym_to_mf)
    dm_sym_order = np.asarray(dm)[perm_sym_to_mf]
    dm_ibz = dm_sym_order[kpts_symm.ibz2bz]
    dm_avg = kpts_symm.transform_dm(dm_ibz)
    dm_avg_mf = dm_avg[perm_mf_to_sym]
    return np.linalg.norm(dm_avg_mf - dm) / np.linalg.norm(dm)


parser = argparse.ArgumentParser()
parser.add_argument("system")
parser.add_argument("kx", type=int)
parser.add_argument("ky", type=int)
parser.add_argument("kz", type=int)
parser.add_argument("basis")
parser.add_argument("-suffix", default=None)
parser.add_argument("--symm", action="store_true")
args = parser.parse_args()

system = args.system
kmesh = (args.kx, args.ky, args.kz)
klabel = system_common.get_klabel(kmesh)
basis = args.basis
suffix = args.suffix
do_symm = args.symm
cell = system_common.make_cell(system, basis, verbose=0, suffix=suffix)
print(cell.mesh)
print('overlap min eigval', np.linalg.eigvalsh(cell.pbc_intor("int1e_ovlp")).min())
xc = system_common.load_xc(system, basis, suffix=suffix)
exxdiv = system_common.load_setting(system, basis, "exxdiv", suffix=suffix)
conv_tol = system_common.load_setting(system, basis, "conv_tol", suffix=suffix)

kpts = cell.make_kpts(kmesh)
kpts_int = np.round(cell.get_scaled_kpts(kpts) * kmesh).astype(int) % kmesh

S = cell.pbc_intor("int1e_ovlp", kpts=kpts)
print('min S eigval', np.linalg.eigvalsh(S).min())

data_dir = system_common.get_data_dir(system, basis, suffix=suffix)
dft_pkl = os.path.join(data_dir, f"DFT_{klabel}.pkl")
gdf_chk = os.path.join(data_dir, f"GDF_{klabel}.chk")


def make_gdf():
    with_df = df.GDF(cell, kpts)
    with_df.verbose = 0
    if os.path.exists(gdf_chk):
        print("Loading GDF cache =", gdf_chk, flush=True)
        with_df._cderi = gdf_chk
    else:
        print("Building GDF cache =", gdf_chk, flush=True)
        with_df._cderi_to_save = gdf_chk
        with_df.build()
        with_df._cderi = gdf_chk
        with_df._cderi_to_save = None
    return with_df


mf = scf.KRKS(cell, kpts, exxdiv=exxdiv, xc=xc).density_fit(with_df=make_gdf())
mf.verbose = 4
if conv_tol is not None:
    mf.conv_tol = conv_tol
print("XC =", xc, flush=True)
print("exxdiv =", exxdiv, flush=True)
print("conv_tol =", mf.conv_tol, flush=True)
print("GDF chkfile =", gdf_chk, flush=True)
print("Running SCF ...", flush=True)
mf.kernel()
print('min mo energy', np.asarray(mf.mo_energy).min())
print('max mo energy', np.asarray(mf.mo_energy).max())

if do_symm:
    dm0 = np.asarray(mf.make_rdm1())
    fock0 = np.asarray(mf.get_fock(dm=dm0))
    S = np.asarray(mf.get_ovlp())

    cell_sym, kpts_symm = build_symmetry_kpts(cell, kmesh)
    perm_sym_to_mf = build_kpt_map(kpts_symm.kpts_scaled, cell.get_scaled_kpts(kpts))
    perm_mf_to_sym = np.argsort(perm_sym_to_mf)

    fock_sym = symmetrize_fock(cell, kpts_symm, fock0[perm_sym_to_mf])[perm_mf_to_sym]
    fock_change = np.linalg.norm(fock_sym - fock0) / np.linalg.norm(fock0)
    fock_symm_err0 = np.linalg.norm(
        symmetrize_fock(cell, kpts_symm, fock0[perm_sym_to_mf])[perm_mf_to_sym] - fock0
    ) / np.linalg.norm(fock0)

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
    dm_symm_err0 = get_dm_symmetry_error(cell, kmesh, kpts, dm0)
    dm_symm_err1 = get_dm_symmetry_error(cell, kmesh, kpts, dm1)
    fock_symm_err1 = np.linalg.norm(
        symmetrize_fock(cell, kpts_symm, fock1[perm_sym_to_mf])[perm_mf_to_sym] - fock1
    ) / np.linalg.norm(fock1)

    mf.mo_energy = mo_energy_sym
    mf.mo_coeff = mo_coeff_sym
    mf.mo_occ = mo_occ_sym

    print("symmetrized final Fock and rediagonalized orbitals", flush=True)
    print("rel_fro Fock difference =", fock_change, flush=True)
    print("rel_fro DM difference =", dm_change, flush=True)
    print("old Fock symm error =", fock_symm_err0, flush=True)
    print("new Fock symm error =", fock_symm_err1, flush=True)
    print("old DM symm error =", dm_symm_err0, flush=True)
    print("new DM symm error =", dm_symm_err1, flush=True)

with open(dft_pkl, "wb") as f:
    pickle.dump(mf, f)
print("Saved DFT pickle =", dft_pkl, flush=True)
