import os
import pickle
import argparse

import h5py
import numpy as np
import pyscf
from functools import reduce

import system_common


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
vxc_path = os.path.join(gw_cache_dir, "vxc.h5")

with open(dft_pkl, "rb") as f:
    mf = pickle.load(f)
    mf.with_df._cderi = gdf_chk
    mf.max_memory = pyscf.lib.parameters.MAX_MEMORY

nkpts = len(mf.kpts)
nmo = mf.mo_coeff[0].shape[1]
mo_coeff = np.asarray(mf.mo_coeff)

with pyscf.lib.temporary_env(mf, verbose=0), pyscf.lib.temporary_env(mf.mol, verbose=0), pyscf.lib.temporary_env(mf.with_df, verbose=0):
    dm = mf.make_rdm1()
    v_mf_ao = mf.get_veff() - mf.get_j(dm_kpts=dm)

v_mf = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)
for k in range(nkpts):
    v_mf[k] = reduce(np.matmul, (mo_coeff[k].T.conj(), v_mf_ao[k], mo_coeff[k]))

os.makedirs(gw_cache_dir, exist_ok=True)
with h5py.File(vxc_path, "w") as f:
    f["v_mf"] = np.asarray(v_mf)

print("Saved vxc cache =", vxc_path, flush=True)
