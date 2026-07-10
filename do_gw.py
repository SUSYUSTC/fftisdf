import os
import pickle
import argparse
import numpy as np
from fcdmft.gw.pbc.krgw_ac import KRGWAC
from fcdmft.gw.pbc import kbse
import threadpoolctl
import system_common
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

gw_path = os.path.join(data_dir, f"GWenergy_{klabel}.npy")
screening_path = os.path.join(data_dir, f"screening_eps_{klabel}.npy")
head_path = os.path.join(data_dir, f"bse_head_{klabel}.npy")

fc = True
gw = KRGWAC(mf)
gw.fc = fc
gw.verbose = 5
gw.writefile = 1
os.makedirs(gw_cache_dir, exist_ok=True)
cwd = os.getcwd()
os.chdir(gw_cache_dir)
gw.kernel()
os.chdir(cwd)

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
if rank == 0:
    np.save(gw_path, gw.mo_energy)
    transfer = kbse._make_transfer_map(mf.cell, mf.kpts)
    eps_inv = kbse._build_screened_coulomb(
        gw,
        gw.mo_energy,
        gw.mo_coeff,
        transfer,
        gamma_coulomb_cutoff=1.0e-5,
    )
    kpts_int = np.round(mf.cell.get_scaled_kpts(mf.kpts) * np.asarray(kmesh)).astype(int) % np.asarray(kmesh)
    kpt_map = {tuple(k): i for i, k in enumerate(kpts_int)}
    negative = np.array([kpt_map[tuple((-k) % np.asarray(kmesh))] for k in kpts_int], dtype=np.int64)
    eps_inv = np.asarray(eps_inv)[negative]
    bse = kbse.KBSE(gw)
    gamma = kbse._gamma_head_strength(bse)
    np.save(screening_path, np.asarray(eps_inv))
    np.save(head_path, np.asarray(gamma))
