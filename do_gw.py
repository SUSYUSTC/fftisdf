import os
import pickle
import argparse
import numpy as np
from fcdmft.gw.pbc.krgw_ac import KRGWAC
import pyscf
import system_common
import threadpoolctl
threadpoolctl.threadpool_limits(1)

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
klabel = system_common.get_klabel(kmesh)
basis = args.basis
suffix = args.suffix

data_dir = system_common.get_data_dir(system, basis, suffix=suffix)
dft_pkl = os.path.join(data_dir, f"DFT_{klabel}.pkl")
gdf_chk = os.path.join(data_dir, f"GDF_{klabel}.chk")
gw_cache_dir = os.path.join(data_dir, f"GWcache_{klabel}")
with open(dft_pkl, "rb") as f:
    mf = pickle.load(f)
    mf.with_df._cderi = gdf_chk
    mf.max_memory = pyscf.lib.parameters.MAX_MEMORY

gw_energy_path = os.path.join(data_dir, f"GWenergy_{klabel}.npy")
gw_pkl_path = os.path.join(data_dir, f"GW_{klabel}.pkl")

fc = system_common.load_section_setting(system, basis, "gw", "fc", suffix=suffix, default=True)
use_pyscf_vk = system_common.load_section_setting(system, basis, "gw", "use_pyscf_vk", suffix=suffix, default=False)


gw = KRGWAC(mf)
gw.fc = fc
gw.verbose = 5
gw.writefile = 1
gw.load_vxc = True
gw.use_pyscf_vk = use_pyscf_vk
os.makedirs(gw_cache_dir, exist_ok=True)
cwd = os.getcwd()
os.chdir(gw_cache_dir)
gw.kernel()
os.chdir(cwd)

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
if rank == 0:
    np.save(gw_energy_path, gw.mo_energy)
    pickle.dump(gw, open(gw_pkl_path, "wb"))
