import argparse
import os
import pickle
from pyscf.pbc import df
from pyscf.pbc import mp

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
basis = args.basis
suffix = args.suffix
klabel = system_common.get_klabel(kmesh)
data_dir = system_common.get_data_dir(system, basis, suffix=suffix)
dft_pkl = os.path.join(data_dir, f"DFT_{klabel}.pkl")
gdf_chk = os.path.join(data_dir, f"GDF_{klabel}.chk")

with open(dft_pkl, "rb") as f:
    mf = pickle.load(f)

if isinstance(mf.with_df, df.GDF):
    mf.with_df._cderi = gdf_chk

print("kmesh", kmesh)
print("basis", basis)
print("suffix", suffix)
print("nocc", mf.cell.nelectron // 2, "nmo", mf.cell.nao, "nkpts", len(mf.kpts))
print("SCF energy", mf.e_tot)
print("GDF", gdf_chk)
print("naux", mf.with_df.get_naoaux())

pt = mp.KMP2(mf)
pt.verbose = 0
emp2, _ = pt.kernel(with_t2=False)

print("PySCF DF MP2 energy  = %.16e" % emp2)
