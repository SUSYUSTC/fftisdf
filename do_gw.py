import sys
import pickle
import numpy as np
from fcdmft.gw.pbc.krgw_ac import KRGWAC
import threadpoolctl
threadpoolctl.threadpool_limits(1)

kmesh = (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
klabel = f"{kmesh[0]}x{kmesh[1]}x{kmesh[2]}"
basis = sys.argv[4]
ke_cutoff = 40.0

pbe_pkl = f"data_GDF/PBE_diamond_{klabel}_{basis}_ke{ke_cutoff}.pkl"
gdf_chk = f"data_GDF/GDF_diamond_{klabel}_{basis}.chk"
with open(pbe_pkl, "rb") as f:
    mf = pickle.load(f)
    mf.with_df._cderi = gdf_chk

gw_path = f'data_GDF/GWenergy_diamond_{klabel}_{basis}.npy'

fc = False
gw = KRGWAC(mf)
gw.fc = fc
gw.verbose = 5
gw.kernel()

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
if rank == 0:
    np.save(gw_path, gw.mo_energy)
