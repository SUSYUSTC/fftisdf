import torch

import utils


nkpts = 4
nocc = 2
nvir = 3
naux = 5
dtype = torch.complex128

torch.manual_seed(12)

Xo_ref = torch.randn(nkpts, naux, nocc, dtype=dtype)
Xo_ref = Xo_ref + 1j * torch.randn(nkpts, naux, nocc, dtype=dtype)
Xv_ref = torch.randn(nkpts, naux, nvir, dtype=dtype)
Xv_ref = Xv_ref + 1j * torch.randn(nkpts, naux, nvir, dtype=dtype)
W_ref = torch.randn(nkpts, naux, naux, dtype=dtype)
W_ref = W_ref + 1j * torch.randn(nkpts, naux, naux, dtype=dtype)

Xo = Xo_ref + 0.2 * torch.randn(nkpts, naux, nocc, dtype=dtype)
Xo = Xo + 0.2j * torch.randn(nkpts, naux, nocc, dtype=dtype)
Xv = Xv_ref + 0.2 * torch.randn(nkpts, naux, nvir, dtype=dtype)
Xv = Xv + 0.2j * torch.randn(nkpts, naux, nvir, dtype=dtype)

L_ref = utils.thc_ovvo_build_Lbar(Xo_ref, Xv_ref, Xo, Xv, (nkpts, 1, 1))
L = utils.thc_ovvo_build_Lbar(Xo, Xv, Xo, Xv, (nkpts, 1, 1))

W_svd = []
W_oinv_PSD = []
for s in range(nkpts):
    rhs = L_ref[s].T @ W_ref[s].conj() @ L_ref[s].conj()
    rhs = rhs.conj()
    W_svd.append(utils.torch_lstsq(L[s], rhs, reg=None))
    W_oinv_PSD.append(utils.torch_lstsq_oinv_PSD(L[s], rhs, reg=None))

W_svd = torch.stack(W_svd)
W_oinv_PSD = torch.stack(W_oinv_PSD)

diff = torch.linalg.norm(W_oinv_PSD - W_svd)
norm = torch.linalg.norm(W_svd)
print("unregularized oinv_PSD vs svd")
print("  abs diff =", f"{diff.item():.16e}")
print("  rel diff =", f"{(diff / norm).item():.16e}")

ref_norm2 = utils.thc_ovvo_inner_from_mo(
    Xo_ref, Xv_ref, W_ref, Xo_ref, Xv_ref, W_ref, (nkpts, 1, 1)
).real

print()
print("reg scan")
print("  reg                 rel_error              ||W||")
rel_errors = []
w_norms = []
for reg in [None, 1e-8, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]:
    W = []
    for s in range(nkpts):
        rhs = L_ref[s].T @ W_ref[s].conj() @ L_ref[s].conj()
        rhs = rhs.conj()
        W.append(utils.torch_lstsq_oinv_PSD(L[s], rhs, reg=reg))
    W = torch.stack(W)

    error2 = utils.thc_ovvo_error2_from_mo(
        Xo_ref, Xv_ref, W_ref, Xo, Xv, W, (nkpts, 1, 1)
    )
    rel_error = torch.sqrt(error2 / ref_norm2)
    w_norm = torch.linalg.norm(W)
    rel_errors.append(rel_error)
    w_norms.append(w_norm)
    reg_label = "None" if reg is None else f"{reg:.1e}"
    print(f"  {reg_label:>8s}   {rel_error.item():.16e}   {w_norm.item():.16e}")


def test_lstsq_oinv_matches_svd():
    assert diff / norm < 1e-10


def test_regularization_reduces_middle_norm():
    assert all(w_norms[i + 1] <= w_norms[i] for i in range(len(w_norms) - 1))
    assert rel_errors[-1] > rel_errors[0]
