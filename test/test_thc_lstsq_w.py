import numpy as np
import torch

import utils


torch.manual_seed(7)
dtype = torch.complex128

kmesh = np.array([2, 2, 1])
kpts_int = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
    ]
)
nkpts = len(kpts_int)
naux = 5
nocc = 2
nvir = 3

Xo_A = torch.randn(nkpts, naux, nocc, dtype=dtype)
Xv_A = torch.randn(nkpts, naux, nvir, dtype=dtype)
W_A = torch.randn(nkpts, naux, naux, dtype=dtype)

Xo_B = torch.randn(nkpts, naux, nocc, dtype=dtype)
Xv_B = torch.randn(nkpts, naux, nvir, dtype=dtype)

for use_fast in [False, True]:
    if use_fast:
        utils.enable_fast_fft()
    else:
        utils.disable_fast_fft()

    ref_norm2 = utils.thc_ovvo_inner_from_mo(
        Xo_A, Xv_A, W_A,
        Xo_A, Xv_A, W_A,
        kmesh,
    ).real
    W_B_opt, error2_combined = utils.thc_ovvo_solve_w_error2_from_mo(
        Xo_A, Xv_A, W_A,
        Xo_B, Xv_B,
        kmesh,
        ref_norm2,
        reg=None,
    )
    W_B_test = W_B_opt.detach().clone().requires_grad_(True)

    error2_explicit = utils.thc_ovvo_error2_from_mo(
        Xo_A, Xv_A, W_A,
        Xo_B, Xv_B, W_B_test,
        kmesh,
    )
    error2_explicit.backward()

    grad_norm = torch.linalg.norm(W_B_test.grad)
    grad_max = W_B_test.grad.abs().max()
    error_abs_diff = abs(error2_combined - error2_explicit.detach())
    error_rel_diff = error_abs_diff / abs(error2_explicit.detach())

    print("use_fast_fft   = %s" % use_fast)
    print("error2_combined = %.16e" % error2_combined.item())
    print("error2_explicit = %.16e" % error2_explicit.item())
    print("error_abs_diff  = %.16e" % error_abs_diff.item())
    print("error_rel_diff  = %.16e" % error_rel_diff.item())
    print("grad_norm       = %.16e" % grad_norm.item())
    print("grad_max        = %.16e" % grad_max.item())
    print("")
    assert error_rel_diff < 1e-12
    assert grad_norm < 1e-10
    assert grad_max < 1e-10


def test_thc_lstsq_w():
    pass
