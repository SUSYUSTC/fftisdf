import sys
import os
import pickle

import numpy as np
from pyscf.pbc import gto, scf, df

kmesh = (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
klabel = f"{kmesh[0]}x{kmesh[1]}x{kmesh[2]}"
basis = sys.argv[4]

a = 1.7834
lv = np.ones((3, 3)) * a
lv -= np.diag([a, a, a])
atom = [("C", [0.00000, 0.00000, 0.00000])]
atom += [("C", [0.5 * a, 0.5 * a, 0.5 * a])]

cell = gto.Cell()
cell.unit = "A"
cell.atom = atom
cell.a = lv
cell.basis = basis
cell.pseudo = "gth-pbe"
cell.ke_cutoff = 40.0
cell.verbose = 0
cell.build()

kpts = cell.make_kpts(kmesh)
kpts_int = np.round(cell.get_scaled_kpts(kpts) * kmesh).astype(int) % kmesh

S = cell.pbc_intor("int1e_ovlp", kpts=kpts)
print('min S eigval', np.linalg.eigvalsh(S).min())

pbe_pkl = f"data_GDF/PBE_diamond_{klabel}_{cell.basis}_ke{cell.ke_cutoff}.pkl"
gdf_chk = f"data_GDF/GDF_diamond_{klabel}_{cell.basis}.chk"


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


mf = scf.KRKS(cell, kpts, exxdiv='ewald', xc='PBE')
mf.conv_tol = 1e-6
mf.max_cycle = 50
mf.verbose = 4
print("GDF chkfile =", gdf_chk, flush=True)
mf.with_df = make_gdf()
print("Running SCF ...", flush=True)
mf.kernel()

with open(pbe_pkl, "wb") as f:
    pickle.dump(mf, f)
print("Saved PBE pickle =", pbe_pkl, flush=True)
