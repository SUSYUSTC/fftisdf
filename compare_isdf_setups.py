import sys
import pickle
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

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

cisdf = 10.0
compare_with_fft = False

print("")
print("cell.ke_cutoff =", cell.ke_cutoff)
print("cell.mesh =", cell.mesh)
print("kmesh =", kmesh)
print("nao =", nao)
print("nocc =", nocc)
print("nvir =", nvir)
print("cisdf =", cisdf)
print("")

C_ovov = [Cocc, Cvir, Cocc, Cvir]
C_ovvo = [Cocc, Cvir, Cvir, Cocc]
fftdf = mf.with_df

if compare_with_fft:
    eri_ovov_7d_fft = fftdf.ao2mo_7d(C_ovov, kpts=kpts)
    eri_ovvo_7d_fft = fftdf.ao2mo_7d(C_ovvo, kpts=kpts)
    norm_ovov_7d_fft = np.linalg.norm(eri_ovov_7d_fft)
    print("ovov ||FFTDF|| = %16.8e" % norm_ovov_7d_fft, flush=True)
    norm_ovvo_7d_fft = np.linalg.norm(eri_ovvo_7d_fft)
    print("ovvo ||FFTDF|| = %16.8e" % norm_ovvo_7d_fft, flush=True)
    emp2_fft = utils.mp2_from_ovov_7d(eri_ovov_7d_fft, mf.mo_energy, nocc, kpts_int, kmesh)
    print("FFTDF            MP2: E = %16.8e" % emp2_fft, flush=True)
    print()

    isdf_ao = fft.ISDF(cell, kpts)
    isdf_ao.verbose = 0
    isdf_ao.build(cisdf=cisdf)
    eri_ovov_7d_isdf_ao = isdf_ao.ao2mo_7d(C_ovov, kpts=kpts)
    eri_ovvo_7d_isdf_ao = isdf_ao.ao2mo_7d(C_ovvo, kpts=kpts)
    diff_isdf_ao_ovov = np.linalg.norm(eri_ovov_7d_isdf_ao - eri_ovov_7d_fft) / norm_ovov_7d_fft
    diff_isdf_ao_ovvo = np.linalg.norm(eri_ovvo_7d_isdf_ao - eri_ovvo_7d_fft) / norm_ovvo_7d_fft
    emp2_isdf_ao = utils.mp2_from_ovov_7d(eri_ovov_7d_isdf_ao, mf.mo_energy, nocc, kpts_int, kmesh)
    print('Xo norm', np.linalg.svd(isdf_ao.inpv_kpt @ Cocc)[1].max())
    print('Xv norm', np.linalg.svd(isdf_ao.inpv_kpt @ Cvir)[1].max())
    print()
    print("ISDF(AO)         MP2: E = %16.8e" % emp2_isdf_ao, flush=True)
    print("ovov ||ISDF(AO) - FFTDF|| / ||FFTDF|| = %16.8e" % diff_isdf_ao_ovov, flush=True)
    print("ovvo ||ISDF(AO) - FFTDF|| / ||FFTDF|| = %16.8e" % diff_isdf_ao_ovvo, flush=True)
    print('ISDF(AO) Coulomb norm', np.linalg.svd(isdf_ao.coul_kpt)[1].max() / nkpts)
    print()

    isdf_ov = fft.ISDF(cell, kpts, ov=(Cocc, Cvir))
    isdf_ov.verbose = 0
    for reg in [0, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]:
        isdf_ov.build(cisdf=cisdf, reg=reg)
        eri_ovov_7d_isdf_ov = isdf_ov.ao2mo_7d(C_ovov, kpts=kpts)
        eri_ovvo_7d_isdf_ov = isdf_ov.ao2mo_7d(C_ovvo, kpts=kpts)
        diff_isdf_ov_ovov = np.linalg.norm(eri_ovov_7d_isdf_ov - eri_ovov_7d_fft) / norm_ovov_7d_fft
        diff_isdf_ov_ovvo = np.linalg.norm(eri_ovvo_7d_isdf_ov - eri_ovvo_7d_fft) / norm_ovvo_7d_fft
        emp2_isdf_ov = utils.mp2_from_ovov_7d(eri_ovov_7d_isdf_ov, mf.mo_energy, nocc, kpts_int, kmesh)
        print(f"ISDF(OV, {str(reg):6s}) MP2: E = %16.8e" % emp2_isdf_ov, flush=True)
        print(f"ovov ||ISDF(OV, {str(reg):6s}) - FFTDF|| / ||FFTDF|| = %16.8e" % diff_isdf_ov_ovov, flush=True)
        print(f"ovvo ||ISDF(OV, {str(reg):6s}) - FFTDF|| / ||FFTDF|| = %16.8e" % diff_isdf_ov_ovvo, flush=True)
        print(f'ISDF(OV, {str(reg):6s}) Coulomb norm', np.abs(np.linalg.eigvalsh(isdf_ov.coul_kpt)[1]).max() / nkpts)
        print()
else:
    isdf_ref = fft.ISDF(cell, kpts, ov=(Cocc, Cvir))
    isdf_ref._isdf = f"data/ISDF_diamond_{klabel}_{cell.basis}_c20_ref.chk"
    isdf_ref.verbose = 0
    isdf_ref.build()

    isdf_ao = fft.ISDF(cell, kpts)
    isdf_ao.verbose = 0
    isdf_ao.build(cisdf=cisdf)
    diff_isdf_ao = utils.compare_two_isdf(isdf_ref, isdf_ao, kmesh, C1=Cocc, C2=Cvir)
    print('Xo norm', np.linalg.svd(isdf_ao.inpv_kpt @ Cocc)[1].max())
    print('Xv norm', np.linalg.svd(isdf_ao.inpv_kpt @ Cvir)[1].max())
    print()
    print("ovvo ||ISDF(AO) - ref|| / ||ref|| = %16.8e" % diff_isdf_ao, flush=True)
    print('ISDF(AO) Coulomb norm', np.linalg.svd(isdf_ao.coul_kpt)[1].max() / nkpts)
    print()

    isdf_ov = fft.ISDF(cell, kpts, ov=(Cocc, Cvir))
    isdf_ov.verbose = 0
    for reg in [0, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]:
        isdf_ov.build(cisdf=cisdf, reg=reg)
        if reg == 0:
            print('Xo norm', np.linalg.svd(isdf_ov.inpv_kpt @ Cocc)[1].max())
            print('Xv norm', np.linalg.svd(isdf_ov.inpv_kpt @ Cvir)[1].max())
        diff_isdf_ov = utils.compare_two_isdf(isdf_ref, isdf_ov, kmesh, C1=Cocc, C2=Cvir)
        print(f"ovvo ||ISDF(OV, {str(reg):6s}) - ref|| / ||ref|| = %16.8e" % diff_isdf_ov, flush=True)
        print(f'ISDF(OV, {str(reg):6s}) Coulomb norm', np.abs(np.linalg.eigvalsh(isdf_ov.coul_kpt)[1]).max() / nkpts)
        print()
