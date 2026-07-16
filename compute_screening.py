import os
import pickle
import argparse
import numpy as np
from fcdmft.gw.pbc import kbse
from pyscf.pbc import df

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
gw_pkl_path = os.path.join(data_dir, f"GW_{klabel}.pkl")
screening_path = os.path.join(data_dir, f"screening_eps_ext_{klabel}.npy")

with open(dft_pkl, "rb") as f:
    mf = pickle.load(f)
if isinstance(mf.with_df, df.GDF):
    mf.with_df._cderi = gdf_chk

with open(gw_pkl_path, "rb") as f:
    gw = pickle.load(f)

gw._scf = mf
gw.kmf = mf
gw.with_df = mf.with_df
gw.cell = mf.cell
gw.kpts = mf.kpts

bse = kbse.KBSE(gw)
bse.qkpt = 0
bse.TDA = True
bse.coulomb_correction = True
bse.dielectric_correction = True
bse.dielectric_wing = True
bse.q0_directions = np.vstack([np.eye(3), -np.eye(3)])
bse.q0_step = 1e-3
bse.build_finite_size()

naux = bse.eps_inv[0].shape[0]
nkpts = len(mf.kpts)
eps_inv_ext = np.zeros((nkpts, naux + 1, naux + 1), dtype=np.complex128)
eps_inv_ext[:, 1:, 1:] = np.asarray(bse.eps_inv)
eps_inv_ext[0, 1:, 1:] = np.asarray(bse.eps_inv_body_q0)
eps_inv_ext[0, 0, 0] = nkpts * bse.coulomb_head * bse.eps_inv_head
eps_inv_ext[0, 1:, 0] = nkpts * bse.coulomb_wing * bse.eps_inv_wing
eps_inv_ext[0, 0, 1:] = nkpts * bse.coulomb_wing * bse.eps_inv_wing.conj()
np.save(screening_path, eps_inv_ext)

print("Saved screening eps =", screening_path)
