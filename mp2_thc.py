import argparse
import os
import pickle
import h5py
from pyscf.pbc import mp

import fft
import system_common


parser = argparse.ArgumentParser()
parser.add_argument("system")
parser.add_argument("kx", type=int)
parser.add_argument("ky", type=int)
parser.add_argument("kz", type=int)
parser.add_argument("basis")
parser.add_argument("-suffix", default=None)
ov_group = parser.add_mutually_exclusive_group(required=True)
ov_group.add_argument("-ov_ref", type=int, default=None)
ov_group.add_argument("-ov_opt", default=None)
args = parser.parse_args()

system = args.system
kmesh = (args.kx, args.ky, args.kz)
basis = args.basis
suffix = args.suffix
klabel = system_common.get_klabel(kmesh)
data_dir = system_common.get_data_dir(system, basis, suffix=suffix)
dft_pkl = os.path.join(data_dir, f"DFT_{klabel}.pkl")

if args.ov_ref is not None:
    ov_chk = os.path.join(data_dir, f"ISDFov_bareGDF_{klabel}_c{args.ov_ref}.chk")
else:
    ov_chk = os.path.join(data_dir, f"ISDFov_opt_bareGDF_{klabel}_{args.ov_opt}.chk")

with open(dft_pkl, "rb") as f:
    mf = pickle.load(f)

with h5py.File(ov_chk, "r") as f:
    nth = f["inpv_kpt"].shape[1]

mf_isdf = mf.copy()
mf_isdf.with_df = fft.ISDF(mf.cell, mf.kpts)
mf_isdf.with_df._isdf = ov_chk
mf_isdf.with_df.build()

print("kmesh", kmesh)
print("basis", basis)
print("suffix", suffix)
print("ov_ref", args.ov_ref)
print("ov_opt", args.ov_opt)
print("nocc", mf.cell.nelectron // 2, "nmo", mf.cell.nao, "nkpts", len(mf.kpts))
print("SCF energy", mf.e_tot)
print("OV THC", ov_chk)
print("nth", nth)

pt = mp.KMP2(mf_isdf)
pt.verbose = 0
emp2, _ = pt.kernel(with_t2=False)

print("PySCF THC MP2 energy = %.16e" % emp2)
