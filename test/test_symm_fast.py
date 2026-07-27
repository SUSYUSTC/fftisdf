import glob
import pickle

import torch

import libsymm


def rel_fro(a, b):
    return torch.linalg.norm(a - b) / torch.linalg.norm(b)


kmesh = (2, 2, 2)

dft_files = sorted(glob.glob("data_diamond_*/DFT_2x2x2_symm.pkl"))
dft_pkl = dft_files[0]

with open(dft_pkl, "rb") as f:
    mf = pickle.load(f)

cell = mf.cell
kpts = cell.make_kpts(kmesh)
cell_isdf = cell.copy()
cell_isdf.mesh = (4, 4, 4)
coords = cell_isdf.gen_uniform_grids(cell_isdf.mesh)
ix_sel = torch.arange(len(coords), dtype=torch.long).numpy()

symm = libsymm.PBCSymmetry(cell_isdf, kmesh, kpts, dtype=torch.complex128)
perm, phase = libsymm.build_isdf_grid_transform(symm, coords, ix_sel, mesh=cell_isdf.mesh)

rng = torch.Generator()
rng.manual_seed(12)
nkpts = len(kpts)
nI = len(ix_sel)
nao = cell.nao_nr()
X = torch.randn((nkpts, nI, nao), generator=rng, dtype=torch.float64)
X = X + 1j * torch.randn((nkpts, nI, nao), generator=rng, dtype=torch.float64)
W = torch.randn((nkpts, nI, nI), generator=rng, dtype=torch.float64)
W = W + 1j * torch.randn((nkpts, nI, nI), generator=rng, dtype=torch.float64)

X_slow = libsymm.symmetrize_isdf_X(symm, X, perm, phase)
X_fast = libsymm.symmetrize_isdf_X_fast(symm, X, perm, phase)
W_slow = libsymm.symmetrize_isdf_W(symm, W, perm, phase)
W_fast = libsymm.symmetrize_isdf_W_fast(symm, W, perm, phase)

X_err = rel_fro(X_fast, X_slow)
W_err = rel_fro(W_fast, W_slow)

print("X fast/slow rel_fro = %.16e" % X_err.item())
print("W fast/slow rel_fro = %.16e" % W_err.item())


def test_symm_fast_matches_slow():
    assert X_err < 1e-12
    assert W_err < 1e-12
