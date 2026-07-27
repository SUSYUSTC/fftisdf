import sys

import numpy as np
import torch

import fft
import fft.isdf_ao2mo
import utils

use_gpu = False
device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

rng = np.random.default_rng(12)

kmesh = np.array([3, 3, 3])
kpts_int = np.stack(np.meshgrid(*map(range, kmesh), indexing="ij")).reshape(3, -1).T.copy()
nkpts = len(kpts_int)
kpt_map = {tuple(k): i for i, k in enumerate(kpts_int)}

kconserv2 = np.zeros((nkpts, nkpts), dtype=int)
kconserv3 = np.zeros((nkpts, nkpts, nkpts), dtype=int)
for k1 in range(nkpts):
    for k2 in range(nkpts):
        #q_int = (kpts_int[k1] - kpts_int[k2]) % kmesh
        q_int = (kpts_int[k2] - kpts_int[k1]) % kmesh
        kconserv2[k1, k2] = kpt_map[tuple(q_int)]
        for k3 in range(nkpts):
            k4_int = (kpts_int[k1] - kpts_int[k2] + kpts_int[k3]) % kmesh
            kconserv3[k1, k2, k3] = kpt_map[tuple(k4_int)]

naux = 7
nocc = 2
nvir = 3
nao = nocc + nvir

Xo_A = rng.normal(size=(nkpts, naux, nocc)) + 1j * rng.normal(size=(nkpts, naux, nocc))
Xv_A = rng.normal(size=(nkpts, naux, nvir)) + 1j * rng.normal(size=(nkpts, naux, nvir))
W_A = rng.normal(size=(nkpts, naux, naux)) + 1j * rng.normal(size=(nkpts, naux, naux))

Xo_B = rng.normal(size=(nkpts, naux, nocc)) + 1j * rng.normal(size=(nkpts, naux, nocc))
Xv_B = rng.normal(size=(nkpts, naux, nvir)) + 1j * rng.normal(size=(nkpts, naux, nvir))
W_B = rng.normal(size=(nkpts, naux, naux)) + 1j * rng.normal(size=(nkpts, naux, naux))

X_A = np.concatenate([Xo_A, Xv_A], axis=2)
X_B = np.concatenate([Xo_B, Xv_B], axis=2)
Cocc = np.eye(nao, nocc, dtype=np.complex128)
Cvir = np.eye(nao, dtype=np.complex128)[:, nocc:]
C_ovvo = [
    np.tile(Cocc[None], (nkpts, 1, 1)),
    np.tile(Cvir[None], (nkpts, 1, 1)),
    np.tile(Cvir[None], (nkpts, 1, 1)),
    np.tile(Cocc[None], (nkpts, 1, 1)),
]
kpts = np.zeros((nkpts, 3))


class FakeCell:
    def nao_nr(self):
        return nao


class FakeISDF:
    pass


df_A = FakeISDF()
df_A.cell = FakeCell()
df_A.kpts = kpts
df_A.kconserv2 = kconserv2
df_A.kconserv3 = kconserv3
df_A.inpv_kpt = X_A
df_A.coul_kpt = W_A
df_A.verbose = 0
df_A.stdout = sys.stdout

df_B = FakeISDF()
df_B.cell = FakeCell()
df_B.kpts = kpts
df_B.kconserv2 = kconserv2
df_B.kconserv3 = kconserv3
df_B.inpv_kpt = X_B
df_B.coul_kpt = W_B
df_B.verbose = 0
df_B.stdout = sys.stdout

get_phase_factor = fft.isdf_ao2mo.get_phase_factor
fft.isdf_ao2mo.get_phase_factor = lambda cell, kpts: np.eye(len(kpts))
eri_A = fft.isdf_ao2mo.ao2mo_7d(df_A, C_ovvo, kpts=kpts)
eri_B = fft.isdf_ao2mo.ao2mo_7d(df_B, C_ovvo, kpts=kpts)
fft.isdf_ao2mo.get_phase_factor = get_phase_factor

ref_error2 = np.linalg.norm(eri_A - eri_B) ** 2
Xo_A_t = torch.from_numpy(Xo_A).to(device)
Xv_A_t = torch.from_numpy(Xv_A).to(device)
W_A_t = torch.from_numpy(W_A).to(device)
Xo_B_t = torch.from_numpy(Xo_B).to(device)
Xv_B_t = torch.from_numpy(Xv_B).to(device)
W_B_t = torch.from_numpy(W_B).to(device)
ref_ab = np.vdot(eri_A, eri_B)

for use_fast in [False, True]:
    if use_fast:
        utils.enable_fast_fft()
    else:
        utils.disable_fast_fft()

    fast_error2 = utils.thc_ovvo_error2_from_mo(
        Xo_A_t, Xv_A_t, W_A_t,
        Xo_B_t, Xv_B_t, W_B_t,
        kmesh,
    )
    fast_ab = utils.thc_ovvo_inner_from_mo(
        Xo_A_t, Xv_A_t, W_A_t,
        Xo_B_t, Xv_B_t, W_B_t,
        kmesh,
    )

    fast_error2_np = fast_error2.detach().cpu().numpy()
    fast_ab_np = fast_ab.detach().cpu().numpy()
    abs_diff = abs(fast_error2_np - ref_error2)
    rel_diff = abs_diff / abs(ref_error2)
    ab_reldiff = abs(fast_ab_np - ref_ab) / abs(ref_ab)

    print("use_fast_fft =", use_fast)
    print("ref_error2   = %.16e" % ref_error2)
    print("fast_error2  = %.16e" % fast_error2_np)
    print("abs_diff     = %.16e" % abs_diff)
    print("rel_diff     = %.16e" % rel_diff)
    print("ab_reldiff   = %.16e" % ab_reldiff)
    print("torch device =", device)
    print("")
    assert rel_diff < 1e-8
    assert ab_reldiff < 1e-8


def test_thc_ovvo_error():
    pass
