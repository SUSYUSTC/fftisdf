import os
import pickle

import numpy as np
import torch

import libsymm
import utils


def rel_fro(A, B):
    return torch.linalg.norm(A - B) / torch.linalg.norm(B)


kmesh = (2, 2, 2)
data_dir = os.path.join(os.path.dirname(__file__), "data", "diamond_2x2x2_gth-szv")

with open(os.path.join(data_dir, "DFT.pkl"), "rb") as f:
    mf = pickle.load(f)
mf.with_df._cderi = os.path.join(data_dir, "GDF.chk")

cell = mf.cell
kpts = cell.make_kpts(kmesh)
kpts_int = np.round(cell.get_scaled_kpts(kpts) * kmesh).astype(int) % kmesh
assert utils.is_k_ordered(kpts_int, kmesh)

cell_isdf = cell.copy()
cell_isdf.mesh = (4, 4, 4)
coords = cell_isdf.gen_uniform_grids(cell_isdf.mesh)
ix_sel = np.arange(len(coords), dtype=np.int64)
symm = libsymm.PBCSymmetry(cell_isdf, kmesh, kpts, dtype=torch.complex128)
perm, phase = libsymm.build_isdf_grid_transform(symm, coords, ix_sel, mesh=cell_isdf.mesh)

rng = torch.Generator()
rng.manual_seed(12)
nkpts = len(kpts)
nI = len(ix_sel)
nao = cell.nao_nr()

assert symm.nops == 48
assert nI == 64


def random_complex(shape):
    A = torch.randn(shape, generator=rng, dtype=torch.float64)
    return A + 1j * torch.randn(shape, generator=rng, dtype=torch.float64)


def test_fast_symmetrizers_match_slow():
    X = random_complex((nkpts, nI, nao))
    W = random_complex((nkpts, nI, nI))

    X_slow = libsymm.symmetrize_isdf_X(symm, X, perm, phase)
    X_fast = libsymm.symmetrize_isdf_X_fast(symm, X, perm, phase)
    W_slow = libsymm.symmetrize_isdf_W(symm, W, perm, phase)
    W_fast = libsymm.symmetrize_isdf_W_fast(symm, W, perm, phase)

    assert rel_fro(X_fast, X_slow) < 1e-12
    assert rel_fro(W_fast, W_slow) < 1e-12


def test_symmetrizers_are_projectors():
    dm = torch.from_numpy(np.asarray(mf.make_rdm1())).to(dtype=torch.complex128)
    dm_sym = libsymm.symmetrize_ao_operator(symm, dm)
    X_sym = libsymm.symmetrize_isdf_X_fast(
        symm, random_complex((nkpts, nI, nao)), perm, phase,
    )
    W_sym = libsymm.symmetrize_isdf_W_fast(
        symm, random_complex((nkpts, nI, nI)), perm, phase,
    )

    assert rel_fro(libsymm.symmetrize_ao_operator(symm, dm_sym), dm_sym) < 1e-12
    assert rel_fro(libsymm.symmetrize_isdf_X_fast(symm, X_sym, perm, phase), X_sym) < 1e-12
    assert rel_fro(libsymm.symmetrize_isdf_W_fast(symm, W_sym, perm, phase), W_sym) < 1e-12


def test_ovvo_solve_intermediates_are_symmetric():
    X = libsymm.symmetrize_isdf_X_fast(
        symm, random_complex((nkpts, nI, nao)), perm, phase,
    )
    C = np.asarray(mf.mo_coeff)
    nocc = cell.nelectron // 2
    Cocc = np.array(C[:, :, :nocc], order="C", copy=True)
    Cvir = np.array(C[:, :, nocc:], order="C", copy=True)
    Cocc_t = torch.from_numpy(Cocc).to(dtype=torch.complex128)
    Cvir_t = torch.from_numpy(Cvir).to(dtype=torch.complex128)
    Xo = torch.einsum("kIa,kai->kIi", X, Cocc_t)
    Xv = torch.einsum("kIa,kab->kIb", X, Cvir_t)

    A = utils.thc_ovvo_build_Lbar(Xo, Xv, Xo, Xv, kmesh)
    R = utils.get_gdf_tensor_compact(mf.with_df, kpts_int, kmesh, Cocc, Cvir, layout="k1q")
    R = torch.from_numpy(R).to(dtype=torch.complex128)
    B = utils.thc_df_ov_rhs_from_mo(Xo, Xv, R, kmesh)

    A_sym = libsymm.symmetrize_isdf_W_fast(symm, A, perm, phase)
    B_sym = libsymm.symmetrize_isdf_W_fast(symm, B, perm, phase)
    assert rel_fro(A_sym, A) < 1e-10
    assert rel_fro(B_sym, B) < 1e-4
