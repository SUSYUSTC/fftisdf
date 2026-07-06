import sys
import os
import pickle
import numpy as np
from fcdmft.gw.pbc.krgw_ac import KRGWAC
import threadpoolctl
import system_common
threadpoolctl.threadpool_limits(1)

system = sys.argv[1]
kmesh = (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
klabel = system_common.get_klabel(kmesh)
basis = sys.argv[5]

data_dir = system_common.get_data_dir(system, basis)
dft_pkl = os.path.join(data_dir, f"DFT_{klabel}.pkl")
gdf_chk = os.path.join(data_dir, f"GDF_{klabel}.chk")
with open(dft_pkl, "rb") as f:
    mf = pickle.load(f)
    mf.with_df._cderi = gdf_chk

gw_path = os.path.join(data_dir, f"GWenergy_{klabel}.npy")

fc = True
gw = KRGWAC(mf)
gw.fc = fc
gw.verbose = 5
gw.kernel()

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
if rank == 0:
    np.save(gw_path, gw.mo_energy)
