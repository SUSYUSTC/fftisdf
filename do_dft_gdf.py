import sys
import os
import pickle

import numpy as np
from pyscf.pbc import scf, df
import system_common

system = sys.argv[1]
kmesh = (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
klabel = system_common.get_klabel(kmesh)
basis = sys.argv[5]
cell = system_common.make_cell(system, basis, verbose=0)
print(cell.mesh)
xc = system_common.load_xc(system, basis)

kpts = cell.make_kpts(kmesh)
kpts_int = np.round(cell.get_scaled_kpts(kpts) * kmesh).astype(int) % kmesh

S = cell.pbc_intor("int1e_ovlp", kpts=kpts)
print('min S eigval', np.linalg.eigvalsh(S).min())

data_dir = system_common.get_data_dir(system, basis)
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


mf = scf.KRKS(cell, kpts, exxdiv='ewald', xc=xc)
mf.verbose = 4
print("XC =", xc, flush=True)
print("GDF chkfile =", gdf_chk, flush=True)
mf.with_df = make_gdf()
print("Running SCF ...", flush=True)
mf.kernel()

with open(dft_pkl, "wb") as f:
    pickle.dump(mf, f)
print("Saved DFT pickle =", dft_pkl, flush=True)
