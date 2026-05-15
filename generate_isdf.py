import sys
import pickle
from pyscf.pbc import df
#import signal
#signal.signal(signal.SIGINT, signal.SIG_DFL)

import numpy as np

import fft
import utils
fft.isdf.CHOLESKY_MAX_SIZE = 12000

nk = int(sys.argv[1])
basis = "gth-dzvp"
ke_cutoff = 40.0
kmesh = np.array([nk, nk, nk])
klabel = f"{kmesh[0]}x{kmesh[1]}x{kmesh[2]}"

scf_pkl = f"data/SCF_diamond_{klabel}_{basis}_ke{ke_cutoff}.pkl"
with open(scf_pkl, "rb") as f:
    mf = pickle.load(f)
if isinstance(mf.with_df, df.GDF):
    gdf_chk = f"data/GDF_diamond_{klabel}_{basis}_ke{ke_cutoff}.chk"
    mf.with_df._cderi = gdf_chk

cell = mf.cell
kpts = cell.make_kpts(kmesh)
kpts_int = np.round(cell.get_scaled_kpts(kpts) * kmesh).astype(int) % kmesh
assert utils.is_k_ordered(kpts_int, kmesh)

# you can run the following code to check pyscf MP2, but somehow it's very slow
#from pyscf.pbc.mp import kmp2
#mp = kmp2.KMP2(mf)
#emp2_pyscf, t2_pyscf = mp.kernel(with_t2=False)
#print("PySCF KMP2: E = %16.8e" % emp2_pyscf)

C = np.asarray(mf.mo_coeff)
nkpts, nao, nmo = C.shape
nocc = cell.nelectron // 2
o = slice(None, nocc)
v = slice(nocc, None)
Cocc = np.array(C[:, :, o], order="C", copy=True)
Cvir = np.array(C[:, :, v], order="C", copy=True)
nvir = Cvir.shape[2]

cisdf = int(sys.argv[2])
reg = 1e-8
compare_with_exact = False
save_isdf = True

print("")
print("cell.ke_cutoff =", cell.ke_cutoff)
print("cell.mesh =", cell.mesh)
print("kmesh =", kmesh)
print("nao =", nao)
print("nocc =", nocc)
print("nvir =", nvir)
print("cisdf =", cisdf)
print("")

# technically the program optimizes (ov|ov), which is different from (ov|vo) (this is what appears in LCC, etc), but numerically they fits equally well, not sure is it a coincidence.
isdf_ov = fft.ISDF(cell, kpts, ov=(Cocc, Cvir))
if save_isdf:
    isdf_ov._isdf_to_save = f"data/ISDF_diamond_{klabel}_{cell.basis}_c{cisdf}.chk"
    print('ISDF(OV) will be saved to', isdf_ov._isdf_to_save)
isdf_ov.verbose = 10
isdf_ov.build(cisdf=cisdf, reg=reg)

print('Xo norm', np.linalg.svdvals(isdf_ov.inpv_kpt @ Cocc).max())
print('Xv norm', np.linalg.svdvals(isdf_ov.inpv_kpt @ Cvir).max())
print('W  norm', np.abs(np.linalg.eigvalsh(isdf_ov.coul_kpt)).max() / nkpts)

if compare_with_exact:
    with_df = mf.with_df
    C_ovov = [Cocc, Cvir, Cocc, Cvir]
    C_ovvo = [Cocc, Cvir, Cvir, Cocc]
    eri_ovov_7d_fft = with_df.ao2mo_7d(C_ovov, kpts=kpts)
    eri_ovvo_7d_fft = with_df.ao2mo_7d(C_ovvo, kpts=kpts)
    norm_ovov_7d_fft = np.linalg.norm(eri_ovov_7d_fft)
    norm_ovvo_7d_fft = np.linalg.norm(eri_ovvo_7d_fft)
    eri_ovov_7d_isdf = isdf_ov.ao2mo_7d(C_ovov, kpts=kpts)
    eri_ovvo_7d_isdf = isdf_ov.ao2mo_7d(C_ovvo, kpts=kpts)
    diff_ovov = np.linalg.norm(eri_ovov_7d_isdf - eri_ovov_7d_fft) / norm_ovov_7d_fft
    diff_ovvo = np.linalg.norm(eri_ovvo_7d_isdf - eri_ovvo_7d_fft) / norm_ovvo_7d_fft
    emp2_fft = utils.mp2_from_ovov_7d(eri_ovov_7d_fft, mf.mo_energy, nocc, kpts_int, kmesh)
    emp2_isdf = utils.mp2_from_ovov_7d(eri_ovov_7d_isdf, mf.mo_energy, nocc, kpts_int, kmesh)
    print()
    print("FFTDF MP2:            E = %16.8e" % emp2_fft, flush=True)
    print(f"ISDF(OV, {str(reg):6s}) MP2: E = %16.8e" % emp2_isdf, flush=True)
    print(f"ovov ||ISDF(OV, {str(reg):6s}) - FFTDF|| / ||FFTDF|| = %16.8e" % diff_ovov, flush=True)
    print(f"ovvo ||ISDF(OV, {str(reg):6s}) - FFTDF|| / ||FFTDF|| = %16.8e" % diff_ovvo, flush=True)
    print(f'ISDF(OV, {str(reg):6s}) Coulomb norm', np.abs(np.linalg.eigvalsh(isdf_ov.coul_kpt)[1]).max() / nkpts)
else:
    isdf_ov_ref = fft.ISDF(cell, kpts, ov=(Cocc, Cvir))
    isdf_ov_ref._isdf = f"data/ISDF_diamond_{klabel}_{cell.basis}_c20_ref.chk"
    isdf_ov_ref.verbose = 0
    isdf_ov_ref.build()
    print("Utility comparison against cisdf=20 reference:", flush=True)
    rel_error = utils.compare_two_isdf(isdf_ov_ref, isdf_ov, Cocc, Cvir, kmesh)
    print(f"ovov ||ISDF(OV, {str(reg):6s}) - ref|| / ||ref|| = %16.8e" % rel_error, flush=True)
