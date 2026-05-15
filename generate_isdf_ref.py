import sys
import pickle
from pyscf.pbc import df
#import signal
#signal.signal(signal.SIGINT, signal.SIG_DFL)

import numpy as np

import fft
fft.isdf.CHOLESKY_MAX_SIZE = 12000
import utils

nk = int(sys.argv[1])
basis = "gth-dzvp"
ke_cutoff = 40.0
kmesh = np.array([nk, nk, nk])
klabel = f"{kmesh[0]}x{kmesh[1]}x{kmesh[2]}"

scf_pkl = f"data/SCF_diamond_{klabel}_{basis}_ke{ke_cutoff}.pkl"
with open(scf_pkl, "rb") as f:
    mf = pickle.load(f)
if isinstance(mf.with_df, df.GDF):
    gdf_chk = f"data/GDF_diamond_{klabel}_{basis}.chk"
    mf.with_df._cderi = gdf_chk

cell = mf.cell
kpts = cell.make_kpts(kmesh)
kpts_int = np.round(cell.get_scaled_kpts(kpts) * kmesh).astype(int) % kmesh
assert utils.is_k_ordered(kpts_int, kmesh)

C = np.asarray(mf.mo_coeff)
nkpts, nao, nmo = C.shape
nocc = cell.nelectron // 2
o = slice(None, nocc)
v = slice(nocc, None)
Cocc = np.array(C[:, :, o], order="C", copy=True)
Cvir = np.array(C[:, :, v], order="C", copy=True)
nvir = Cvir.shape[2]

cisdf = 20
compare_with_exact = False

print("")
print("cell.ke_cutoff =", cell.ke_cutoff)
print("cell.mesh =", cell.mesh)
print("kmesh =", kmesh)
print("nao =", nao)
print("nocc =", nocc)
print("nvir =", nvir)
print("cisdf =", cisdf)
print("")

chkfile = f"data/ISDF_diamond_{klabel}_{cell.basis}_c{cisdf}_ref.chk"

isdf_ov = fft.ISDF(cell, kpts, ov=(Cocc, Cvir))
#isdf_ov._isdf = chkfile
isdf_ov._isdf_to_save = chkfile
isdf_ov.verbose = 10
isdf_ov.build(cisdf=cisdf)

C_ovvo = [Cocc, Cvir, Cvir, Cocc]
if compare_with_exact:
    eri_fft = mf.with_df.ao2mo_7d(C_ovvo)
    eri_ref = isdf_ov.ao2mo_7d(C_ovvo)
    print("rel error of ERI =", np.linalg.norm(eri_fft - eri_ref) / np.linalg.norm(eri_fft))

for nsamples in [16, 64, 256]:
    rel_err, rel_std = utils.stochastic_eri_diff(mf.with_df, isdf_ov, C_ovvo, nsamples)
    print(f"rel_err = {rel_err:.2e}, rel_std = {rel_std:.2e}")
