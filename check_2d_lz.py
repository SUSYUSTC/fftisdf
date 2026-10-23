import argparse
import os
import tempfile
import time

import numpy as np
import pyscf
import pyscf.pbc.gto
import pyscf.pbc.dft
import pyscf.pbc.df

import system_common


parser = argparse.ArgumentParser()
parser.add_argument("system")
parser.add_argument("basis")
parser.add_argument("-suffix", default=None)
parser.add_argument("--inf-vacuum", action="store_true")
parser.add_argument("-Lz", type=float, nargs="+", default=[8, 10, 12, 15, 18, 20, 30, 50])
args = parser.parse_args()

system = args.system
basis = args.basis
suffix = args.suffix
data_dir = system_common.get_data_dir(system, basis, suffix=suffix)
atom, lattice = system_common.load_structure(system, basis, suffix=suffix)
settings = system_common.load_settings(system, basis, suffix=suffix)
pseudo = settings["pseudo_potential"]
xc = settings["xc"]
basis_input = system_common.load_basis(atom, basis, pseudo)
symbols = [x[0] for x in atom]
coords0 = np.asarray([x[1] for x in atom])
coords0[:, 2] -= coords0[:, 2].mean()
kmesh = (1, 1, 1)
low_dim_ft_type = "inf_vacuum" if args.inf_vacuum else None
mode = "inf_vacuum" if args.inf_vacuum else "finite_vacuum"
out_dir = tempfile.mkdtemp(prefix=f"{system}_2d_lz_")
rows = []
DM_all = []
e_all = []

print("system", system)
print("basis", basis)
print("output", out_dir)
print("PySCF", pyscf.__version__, "threads", pyscf.lib.num_threads())
print("low_dim_ft_type", low_dim_ft_type)

for Lz in args.Lz:
    begin = time.perf_counter()
    a = lattice.copy()
    a[2] = [0, 0, Lz]
    coords = coords0.copy()
    coords[:, 2] += Lz / 2

    cell = pyscf.pbc.gto.Cell()
    cell.unit = "A"
    cell.a = a
    cell.atom = list(zip(symbols, coords))
    cell.dimension = 2
    cell.low_dim_ft_type = low_dim_ft_type
    cell.basis = basis_input
    cell.pseudo = pseudo
    cell.precision = 1e-8
    cell.max_memory = 4000
    cell.verbose = 0
    cell.build()

    kpts = cell.make_kpts(kmesh)
    mf = pyscf.pbc.dft.KRKS(cell, kpts, xc=xc, exxdiv=None).density_fit()
    mf.with_df = pyscf.pbc.df.GDF(cell, kpts)
    mf.with_df._cderi_to_save = os.path.join(out_dir, f"GDF_Lz{Lz:g}.chk")
    mf.max_memory = 4000
    mf.conv_tol = 1e-8
    mf.max_cycle = 100
    mf.init_guess = "atom"

    print("START", Lz, "mesh", cell.mesh, flush=True)
    mf.kernel()
    e = np.asarray(mf.mo_energy)
    DM = np.asarray(mf.make_rdm1())
    nocc = cell.nelectron // 2
    gap = e[:, nocc].min() - e[:, nocc - 1].max()
    seconds = time.perf_counter() - begin
    rows.append([Lz, int(mf.converged), mf.e_tot, gap, seconds])
    DM_all.append(DM)
    e_all.append(e)
    print("RESULT Lz=%g converged=%s E=%.12f gap_Ha=%.12f time=%.2f" %
          (Lz, mf.converged, mf.e_tot, gap, seconds), flush=True)

rows = np.asarray(rows)
DM_ref = DM_all[-1]
e_ref = e_all[-1]
dm_rel = np.asarray([np.linalg.norm(DM - DM_ref) / np.linalg.norm(DM_ref) for DM in DM_all])
e_rel = np.asarray([np.linalg.norm(e - e_ref) / np.linalg.norm(e_ref) for e in e_all])

print("SUMMARY relative to largest Lz")
print("Lz converged E_Ha gap_Ha dE_mHa dgap_mHa DM_rel mo_energy_rel seconds")
for row, dm_err, e_err in zip(rows, dm_rel, e_rel):
    print("%g %d %.12f %.12f %+.6f %+.6f %.6e %.6e %.2f" %
          (*row[:4], 1000 * (row[2] - rows[-1, 2]), 1000 * (row[3] - rows[-1, 3]),
           dm_err, e_err, row[-1]))

path = os.path.join(data_dir, f"lz_convergence_{mode}.npz")
np.savez(path, Lz=rows[:, 0], converged=rows[:, 1].astype(bool), energy=rows[:, 2],
         gap=rows[:, 3], seconds=rows[:, 4], dm_rel=dm_rel, mo_energy_rel=e_rel)
print("Saved", path)
