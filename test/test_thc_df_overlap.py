import os
import pickle

import numpy as np
import torch

import fft
import utils


def print_complex(name, z):
    print("%-12s = %.16e%+.16ej" % (name, z.real, z.imag))


def print_reldiff(name, a, b):
    print("%-12s = %.16e" % (name, abs(a - b) / abs(b)))


def rel_diff(a, b):
    return abs(a - b) / abs(b)


def make_thc(cell, kpts, X_ao, W):
    df_thc = fft.ISDF(cell, kpts=kpts)
    df_thc._inpv_kpt = X_ao
    df_thc._coul_kpt = W
    return df_thc


rng = np.random.default_rng(12)

basis = "gth-szv"
kmesh = (1, 3, 3)
klabel = f"{kmesh[0]}x{kmesh[1]}x{kmesh[2]}"

test_data_dir = os.path.join(os.path.dirname(__file__), "data_thc_overlap")
scf_pkl = os.path.join(test_data_dir, f"DFT_diamond_{klabel}_{basis}.pkl")
with open(scf_pkl, "rb") as f:
    mf = pickle.load(f)

gdf_chk = os.path.join(test_data_dir, f"GDF_diamond_{klabel}_{basis}.chk")
mf.with_df._cderi = gdf_chk

cell = mf.cell
kpts = cell.make_kpts(kmesh)
kpts_int = np.round(cell.get_scaled_kpts(kpts) * kmesh).astype(int) % kmesh
assert utils.is_k_ordered(kpts_int, kmesh)

C = np.asarray(mf.mo_coeff)
nkpts, nao, nmo = C.shape
nocc = cell.nelectron // 2
Cocc = np.array(C[:, :, :nocc], order="C", copy=True)
Cvir = np.array(C[:, :, nocc:], order="C", copy=True)
C_ovvo = [Cocc, Cvir, Cvir, Cocc]

nip_A = 5
nip_B = 6
X_ao_A = rng.normal(size=(nkpts, nip_A, nao)) + 1j * rng.normal(size=(nkpts, nip_A, nao))
X_ao_B = rng.normal(size=(nkpts, nip_B, nao)) + 1j * rng.normal(size=(nkpts, nip_B, nao))
W_A = rng.normal(size=(nkpts, nip_A, nip_A)) + 1j * rng.normal(size=(nkpts, nip_A, nip_A))
W_B = rng.normal(size=(nkpts, nip_B, nip_B)) + 1j * rng.normal(size=(nkpts, nip_B, nip_B))

X_A = X_ao_A @ C
X_B = X_ao_B @ C
Xo_A = X_ao_A @ Cocc
Xv_A = X_ao_A @ Cvir
Xo_B = X_ao_B @ Cocc
Xv_B = X_ao_B @ Cvir

X_A_t = torch.from_numpy(X_A)
X_B_t = torch.from_numpy(X_B)
Xo_A_t = torch.from_numpy(Xo_A)
Xv_A_t = torch.from_numpy(Xv_A)
Xo_B_t = torch.from_numpy(Xo_B)
Xv_B_t = torch.from_numpy(Xv_B)
W_A_t = torch.from_numpy(W_A)
W_B_t = torch.from_numpy(W_B)

df_thc_A = make_thc(cell, kpts, X_ao_A, W_A)
df_thc_B = make_thc(cell, kpts, X_ao_B, W_B)

R = utils.get_gdf_tensor_compact(mf.with_df, kpts_int, kmesh, C, C, layout="k1q")
R_ov = utils.get_gdf_tensor_compact(mf.with_df, kpts_int, kmesh, Cocc, Cvir, layout="k1q")
R_t = torch.from_numpy(R)
R_ov_t = torch.from_numpy(R_ov)

R_full = utils.gdf_compact_to_full(R, kpts_int, kmesh, layout="k1q")
neg = utils.negative_k(np.arange(nkpts), kmesh).numpy()
diff = np.linalg.norm(R_full.swapaxes(0, 1).swapaxes(-1, -2) - R_full[:, :, neg].conj())
print('R Hermitian error', diff)

eri_df = mf.with_df.ao2mo_7d(C)
eri_df_ovvo = mf.with_df.ao2mo_7d(C_ovvo)
df_norm2 = np.vdot(eri_df, eri_df).real
df_norm2_ovvo = np.vdot(eri_df_ovvo, eri_df_ovvo).real
fast_df_norm2 = utils.df_inner_from_mo(R_t, kmesh).detach().cpu().numpy().real
fast_df_norm2_ovvo = utils.df_inner_from_mo(R_ov_t, kmesh).detach().cpu().numpy().real

print("cell.basis =", cell.basis)
print("kmesh      =", kmesh)
print("nao        =", nao)
print("nmo        =", nmo)
print("nip_A      =", nip_A)
print("nip_B      =", nip_B)


print("")
print("THC-THC full")
eri_A = df_thc_A.ao2mo_7d(C)
eri_B = df_thc_B.ao2mo_7d(C)
ref_ab = np.vdot(eri_A, eri_B)
ref_error2 = np.linalg.norm(eri_A - eri_B) ** 2
fast_ab = utils.thc_inner_from_mo(X_A_t, W_A_t, X_B_t, W_B_t, kmesh).detach().cpu().numpy()
fast_error2 = utils.thc_error2_from_mo(X_A_t, W_A_t, X_B_t, W_B_t, kmesh).detach().cpu().numpy()
ref_norm2 = utils.thc_inner_from_mo(X_A_t, W_A_t, X_A_t, W_A_t, kmesh).real
W_fit, fit_error2 = utils.thc_solve_w_error2_from_mo(X_A_t, W_A_t, X_B_t, kmesh, ref_norm2)
df_fit = make_thc(cell, kpts, X_ao_B, W_fit.detach().cpu().numpy())
fit_error2_ref = np.linalg.norm(eri_A - df_fit.ao2mo_7d(C)) ** 2
print_complex("ref_ab", ref_ab)
print_complex("fast_ab", fast_ab)
ab_reldiff = rel_diff(fast_ab, ref_ab)
err_reldiff = rel_diff(fast_error2, ref_error2)
fit_reldiff = rel_diff(fit_error2.detach().cpu().numpy(), fit_error2_ref)
print("%-12s = %.16e" % ("ab_reldiff", ab_reldiff))
print("%-12s = %.16e" % ("err_reldiff", err_reldiff))
print("%-12s = %.16e" % ("fit_reldiff", fit_reldiff))
assert ab_reldiff < 1e-8
assert err_reldiff < 1e-8
assert fit_reldiff < 1e-8


print("")
print("THC-THC ovvo")
eri_A = df_thc_A.ao2mo_7d(C_ovvo)
eri_B = df_thc_B.ao2mo_7d(C_ovvo)
ref_ab = np.vdot(eri_A, eri_B)
ref_error2 = np.linalg.norm(eri_A - eri_B) ** 2
fast_ab = utils.thc_ovvo_inner_from_mo(
    Xo_A_t, Xv_A_t, W_A_t, Xo_B_t, Xv_B_t, W_B_t, kmesh
).detach().cpu().numpy()
fast_error2 = utils.thc_ovvo_error2_from_mo(
    Xo_A_t, Xv_A_t, W_A_t, Xo_B_t, Xv_B_t, W_B_t, kmesh
).detach().cpu().numpy()
ref_norm2 = utils.thc_ovvo_inner_from_mo(
    Xo_A_t, Xv_A_t, W_A_t, Xo_A_t, Xv_A_t, W_A_t, kmesh
).real
W_fit, fit_error2 = utils.thc_ovvo_solve_w_error2_from_mo(
    Xo_A_t, Xv_A_t, W_A_t, Xo_B_t, Xv_B_t, kmesh, ref_norm2
)
df_fit = make_thc(cell, kpts, X_ao_B, W_fit.detach().cpu().numpy())
fit_error2_ref = np.linalg.norm(eri_A - df_fit.ao2mo_7d(C_ovvo)) ** 2
print_complex("ref_ab", ref_ab)
print_complex("fast_ab", fast_ab)
ab_reldiff = rel_diff(fast_ab, ref_ab)
err_reldiff = rel_diff(fast_error2, ref_error2)
fit_reldiff = rel_diff(fit_error2.detach().cpu().numpy(), fit_error2_ref)
print("%-12s = %.16e" % ("ab_reldiff", ab_reldiff))
print("%-12s = %.16e" % ("err_reldiff", err_reldiff))
print("%-12s = %.16e" % ("fit_reldiff", fit_reldiff))
assert ab_reldiff < 1e-8
assert err_reldiff < 1e-8
assert fit_reldiff < 1e-8


print("")
print("THC-DF full")
eri_A = df_thc_A.ao2mo_7d(C)
ref_ab = np.vdot(eri_A, eri_df)
ref_error2 = np.linalg.norm(eri_A - eri_df) ** 2
fast_ab = utils.thc_df_inner_from_mo(X_A_t, W_A_t, R_t, kmesh).detach().cpu().numpy()
fast_error2 = utils.thc_df_error2_from_mo(
    X_A_t, W_A_t, R_t, kmesh, torch.tensor(fast_df_norm2, dtype=X_A_t.real.dtype)
).detach().cpu().numpy()
W_fit, fit_error2 = utils.thc_df_solve_w_error2_from_mo(
    X_A_t, R_t, kmesh, torch.tensor(fast_df_norm2, dtype=X_A_t.real.dtype)
)
df_fit = make_thc(cell, kpts, X_ao_A, W_fit.detach().cpu().numpy())
fit_error2_ref = np.linalg.norm(df_fit.ao2mo_7d(C) - eri_df) ** 2
print_complex("ref_ab", ref_ab)
print_complex("fast_ab", fast_ab)
ab_reldiff = rel_diff(fast_ab, ref_ab)
bb_reldiff = rel_diff(fast_df_norm2, df_norm2)
err_reldiff = rel_diff(fast_error2, ref_error2)
fit_reldiff = rel_diff(fit_error2.detach().cpu().numpy(), fit_error2_ref)
print("%-12s = %.16e" % ("ab_reldiff", ab_reldiff))
print("%-12s = %.16e" % ("bb_reldiff", bb_reldiff))
print("%-12s = %.16e" % ("err_reldiff", err_reldiff))
print("%-12s = %.16e" % ("fit_reldiff", fit_reldiff))
assert ab_reldiff < 1e-8
assert bb_reldiff < 1e-8
assert err_reldiff < 1e-8
assert fit_reldiff < 1e-8


print("")
print("THC-DF ovvo")
eri_A = df_thc_A.ao2mo_7d(C_ovvo)
ref_ab = np.vdot(eri_A, eri_df_ovvo)
ref_error2 = np.linalg.norm(eri_A - eri_df_ovvo) ** 2
fast_ab = utils.thc_df_ov_inner_from_mo(
    Xo_A_t, Xv_A_t, W_A_t, R_ov_t, kmesh
).detach().cpu().numpy()
fast_error2 = utils.thc_df_ov_error2_from_mo(
    Xo_A_t, Xv_A_t, W_A_t, R_ov_t, kmesh,
    torch.tensor(fast_df_norm2_ovvo, dtype=X_A_t.real.dtype),
).detach().cpu().numpy()
W_fit, fit_error2 = utils.thc_df_ov_solve_w_error2_from_mo(
    Xo_A_t, Xv_A_t, R_ov_t, kmesh,
    torch.tensor(fast_df_norm2_ovvo, dtype=X_A_t.real.dtype),
)
df_fit = make_thc(cell, kpts, X_ao_A, W_fit.detach().cpu().numpy())
fit_error2_ref = np.linalg.norm(df_fit.ao2mo_7d(C_ovvo) - eri_df_ovvo) ** 2
print_complex("ref_ab", ref_ab)
print_complex("fast_ab", fast_ab)
ab_reldiff = rel_diff(fast_ab, ref_ab)
bb_reldiff = rel_diff(fast_df_norm2_ovvo, df_norm2_ovvo)
err_reldiff = rel_diff(fast_error2, ref_error2)
fit_reldiff = rel_diff(fit_error2.detach().cpu().numpy(), fit_error2_ref)
print("%-12s = %.16e" % ("ab_reldiff", ab_reldiff))
print("%-12s = %.16e" % ("bb_reldiff", bb_reldiff))
print("%-12s = %.16e" % ("err_reldiff", err_reldiff))
print("%-12s = %.16e" % ("fit_reldiff", fit_reldiff))
assert ab_reldiff < 1e-8
assert bb_reldiff < 1e-8
assert err_reldiff < 1e-8
assert fit_reldiff < 1e-8


def test_thc_df_overlap():
    pass
