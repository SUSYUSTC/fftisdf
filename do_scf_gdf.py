import sys
import os
import pickle
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

import numpy as np
from pyscf.pbc import scf, df
import system_common

system = sys.argv[1]
kmesh = (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
basis = sys.argv[5]
cell = system_common.make_cell(system, basis, verbose=0)

klabel = system_common.get_klabel(kmesh)
kpts = cell.make_kpts(kmesh)
kpts_int = np.round(cell.get_scaled_kpts(kpts) * kmesh).astype(int) % kmesh

S = cell.pbc_intor("int1e_ovlp", kpts=kpts)
print('min S eigval', np.linalg.eigvalsh(S).min())

data_dir = system_common.get_data_dir(system, basis)
scf_pkl = os.path.join(data_dir, f"SCF_{klabel}.pkl")
gdf_chk = system_common.resolve_path(
    system, basis,
    f"GDF_{klabel}.chk",
    f"GDF_{system}_{klabel}_{basis}.chk",
)


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


mf = scf.KHF(cell, kpts, exxdiv='ewald')
mf.conv_tol = 1e-6
mf.max_cycle = 50
mf.verbose = 4
print("GDF chkfile =", gdf_chk, flush=True)
mf.with_df = make_gdf()
print("Running SCF ...", flush=True)
mf.kernel()

with open(scf_pkl, "wb") as f:
    pickle.dump(mf, f)
print("Saved SCF pickle =", scf_pkl, flush=True)
