import functools
import numpy as np
import torch
import builtins
from inspect import signature, Parameter
from pyscf import lib

has_profile = hasattr(builtins, 'profile')
if not has_profile:
    profile = lambda x: x

df_eig = False
disable_pm_sort = False
enable_profile = False
use_fast_fft = False


def enable_fast_fft():
    global use_fast_fft
    use_fast_fft = True


def disable_fast_fft():
    global use_fast_fft
    use_fast_fft = False


def synchronize():
    if enable_profile:
        torch.cuda.synchronize()


def modify_signature(func, args_to_add, args_to_delete, **kwargs):
    original_sig = signature(func)
    new_params = [value for value in original_sig.parameters.values() if value.name not in args_to_delete]
    for arg in args_to_add:
        new_params.append(Parameter(arg, Parameter.POSITIONAL_ONLY))
    for key, value in kwargs.items():
        new_params.append(Parameter(key, Parameter.POSITIONAL_OR_KEYWORD, default=value))
    order = [Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY, Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD]
    params_reordered = sorted(new_params, key=lambda p: order.index(p.kind))
    new_sig = original_sig.replace(parameters=params_reordered)

    def decorator(func_input):
        func_input.__signature__ = new_sig
        return func_input

    return decorator


def maybe_profile(func):
    func_profile = profile(func)

    @modify_signature(func, [], [])
    def newfunc(*args, **kwargs):
        if enable_profile:
            return func_profile(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return newfunc

    
def wrap(func, **fixed_kwargs):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        kwargs.update(fixed_kwargs)
        return func(*args, **kwargs)
    return wrapped


def ravel_multi_index(indices, shape):
    strides = shape2stride(shape)
    return sum(i * s for i, s in tuple(zip(indices, strides))[:-1]) + indices[-1]


def matrix_operation(A, func):
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = func(eigvals)
    return eigvecs @ np.diag(eigvals) @ eigvecs.conj().T


def make_df_eig():
    global df_eig
    import pyscf.df
    import pyscf.pbc.df
    if df_eig:
        return

    def _eig_decompose(dev, j2c, lindep=pyscf.df.incore.LINEAR_DEP_THR):
        return matrix_operation(j2c, lambda x: 1 / np.sqrt(x))

    def eigenvalue_decomposed_metric(self, j2c):
        result = matrix_operation(j2c, lambda x: 1 / np.sqrt(x))
        return result, None, 'ED'

    pyscf.df.incore.cholesky_eri = wrap(pyscf.df.incore.cholesky_eri, decompose_j2c='eig')
    pyscf.df.incore._eig_decompose = _eig_decompose
    pyscf.pbc.df.rsdf_builder._RSGDFBuilder.j2c_eig_always = True
    pyscf.pbc.df.rsdf_builder._RSGDFBuilder.eigenvalue_decomposed_metric = eigenvalue_decomposed_metric
    df_eig = True


def make_disable_pm_sort():
    global disable_pm_sort
    import pyscf.tools.mo_mapping
    if disable_pm_sort:
        return

    pyscf.tools.mo_mapping.mo_1to1map = lambda u: np.arange(u.shape[1])
    disable_pm_sort = True


def shape2stride(shape):
    stride = [1]
    for dim in shape[1:][::-1]:
        stride.append(stride[-1] * dim)
    return tuple(stride[::-1])


def unravel_index(indices, shape):
    strides = shape2stride(shape)
    # the last stride is always 1
    n = len(shape)
    result = torch.empty(indices.shape + (n,), dtype=indices.dtype, device=indices.device)
    for i, s in enumerate(strides):
        result[..., i] = indices // s
        indices = indices - result[..., i] * s
    return tuple(result[..., i] for i in range(n))


def get_one_phase(n):
    j = torch.arange(n).to(torch.float64)
    phase = torch.exp((2j * torch.pi / n) * (j[:, None] * j[None, :])) / torch.sqrt(torch.tensor(n, dtype=torch.float64))
    return phase


def add_k(idx1, idx2, kmesh):
    if not isinstance(idx1, torch.Tensor):
        idx1 = torch.tensor(idx1)
    if not isinstance(idx2, torch.Tensor):
        idx2 = torch.tensor(idx2)
    idx1 = unravel_index(idx1, kmesh)
    idx2 = unravel_index(idx2, kmesh)
    idx_sum = tuple((i1 + i2) % n for i1, i2, n in zip(idx1, idx2, kmesh))
    idx = ravel_multi_index(idx_sum, kmesh)
    return idx


def negative_k(idx, kmesh):
    if not isinstance(idx, torch.Tensor):
        idx = torch.tensor(idx)
    idx = unravel_index(idx, kmesh)
    idx_neg = tuple((-i) % n for i, n in zip(idx, kmesh))
    idx = ravel_multi_index(idx_neg, kmesh)
    return idx


def fourier_transform_3d(tensor, *, axis, kmesh, inverse):
    nkpts = tensor.shape[axis]
    assert int(np.prod(kmesh)) == nkpts
    tensor = tensor.movedim(axis, 0)
    tensor = tensor.reshape(kmesh + tensor.shape[1:])
    if use_fast_fft:
        if inverse:
            tensor = torch.fft.ifftn(tensor, dim=tuple(range(len(kmesh))), norm='ortho')
        else:
            tensor = torch.fft.fftn(tensor, dim=tuple(range(len(kmesh))), norm='ortho')
    else:
        phase_all = [get_one_phase(n).to(dtype=tensor.dtype, device=tensor.device) for n in kmesh]
        if inverse:
            phase_all = [phase.conj() for phase in phase_all]
        tensor = torch.einsum('ijk...,ai,bj,ck->abc...', tensor, *phase_all)
    tensor = tensor.reshape((nkpts, ) + tensor.shape[3:])
    tensor = tensor.movedim(0, axis)
    return tensor


def k2R(tensor, *, axis, dual, kmesh):
    for ax, is_dual in zip(axis, dual):
        tensor = fourier_transform_3d(tensor, axis=ax, kmesh=kmesh, inverse=is_dual)
    return tensor


def R2k(tensor, *, axis, dual, kmesh):
    return k2R(tensor, axis=axis, dual=tuple([not d for d in dual]), kmesh=kmesh)


@maybe_profile
def c_loss_F_norm(logc, X_t, W_t):
    X2 = X_t.abs()**2
    a = X2.sum(dim=tuple(i for i in range(X_t.ndim) if i != 1)).to(logc.dtype)

    W2 = W_t.abs()**2
    b = W2.sum(dim=tuple(i for i in range(W_t.ndim) if i not in (W_t.ndim - 2, W_t.ndim - 1))).to(logc.dtype)

    c2 = torch.exp(2.0 * logc)
    c4_inv = torch.exp(-4.0 * logc)
    sx2 = torch.dot(a, c2)
    sw2 = (b * c4_inv[:, None] * c4_inv[None, :]).sum()
    return sx2 * sx2 * torch.sqrt(sw2)


@maybe_profile
def c_loss_2_norm(logc, X_t, W_t):
    c = torch.exp(logc)
    X = X_t * c[None, :, None]
    W = W_t / (c[None, :, None]**2 * c[None, None, :]**2)
    s_X = torch.linalg.svdvals(X).max()
    s_W = torch.linalg.eigvalsh(W).abs().max()
    return s_X**4 * s_W


@maybe_profile
def thc_ovvo_build_Lbar(Xo_A, Xv_A, Xo_B, Xv_B, kmesh):
    nkpts, naux_A = Xo_A.shape[:2]
    sqrt_nkpts = np.sqrt(nkpts).item()
    naux_B = Xo_B.shape[1]
    kmesh = tuple(int(x) for x in kmesh)
    assert nkpts == int(np.prod(kmesh))

    O = torch.einsum("pIi,pKi->pIK", Xo_A, Xo_B.conj())
    V = torch.einsum("qIa,qKa->qIK", Xv_A.conj(), Xv_B)

    O_fft = fourier_transform_3d(O, axis=0, kmesh=kmesh, inverse=False)
    V_fft = fourier_transform_3d(V, axis=0, kmesh=kmesh, inverse=False)

    negative = negative_k(torch.arange(nkpts), kmesh)
    Lbar = O_fft[negative] * V_fft
    Lbar = fourier_transform_3d(Lbar, axis=0, kmesh=kmesh, inverse=True)

    return Lbar.reshape(nkpts, naux_A, naux_B) * sqrt_nkpts


@maybe_profile
def torch_lstsq(a, b, tol=1e-10, reg=None):
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    r = s[None, :] * s[:, None]
    m = torch.abs(r) > tol * tol

    t = u.conj().T @ b @ u
    if reg is None:
        t = torch.where(m, t / r, torch.zeros_like(t))
    else:
        t = torch.where(m, t * r / (r * r + reg * reg), torch.zeros_like(t))

    v = vh.conj().T
    return v @ t @ vh


@maybe_profile
def torch_lstsq_oinv(a, b, reg=None):
    dtype = a.dtype
    a = a.to(dtype=torch.complex128)
    b = b.to(dtype=torch.complex128)
    synchronize()
    if a.ndim == 3:
        if reg is None:
            x = torch.linalg.solve(a, b)
            result = torch.linalg.solve(a.transpose(-1, -2), x.transpose(-1, -2)).transpose(-1, -2)
        else:
            eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
            ah = a.conj().transpose(-1, -2)
            o = ah @ a + reg**2 * eye
            rhs = ah @ b @ a
            synchronize()
            x = torch.linalg.solve(o, rhs)
            synchronize()
            result = torch.linalg.solve(o.transpose(-1, -2), x.transpose(-1, -2)).transpose(-1, -2)
            synchronize()
    else:
        if reg is None:
            result = torch.linalg.solve(a.T, torch.linalg.solve(a, b).T).T
        else:
            eye = torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
            ah = a.conj().T
            o = ah @ a + reg**2 * eye
            rhs = ah @ b @ a
            result = torch.linalg.solve(o.T, torch.linalg.solve(o, rhs).T).T
    return result.to(dtype=dtype)


@maybe_profile
def torch_lstsq_oinv_PSD(a, b, reg=None):
    dtype = a.dtype
    a = a.to(dtype=torch.complex128)
    b = b.to(dtype=torch.complex128)
    if reg is not None:
        a = a + reg * torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    synchronize()
    if a.ndim == 3:
        x = torch.linalg.solve(a, b)
        result = torch.linalg.solve(a.transpose(-1, -2), x.transpose(-1, -2)).transpose(-1, -2)
    else:
        result = torch.linalg.solve(a.T, torch.linalg.solve(a, b).T).T
    synchronize()
    return result.to(dtype=dtype)


@maybe_profile
def thc_ovvo_solve_w_from_mo(Xo_ref, Xv_ref, W_ref, Xo, Xv, kmesh, reg=None):
    L_ref = thc_ovvo_build_Lbar(Xo_ref, Xv_ref, Xo, Xv, kmesh)
    L = thc_ovvo_build_Lbar(Xo, Xv, Xo, Xv, kmesh)
    rhs = (L_ref.transpose(-1, -2) @ W_ref.conj() @ L_ref.conj()).conj()
    return torch_lstsq_oinv_PSD(L, rhs, reg=reg)


@maybe_profile
def thc_ovvo_solve_w_error2_from_mo(Xo_ref, Xv_ref, W_ref, Xo, Xv, kmesh, ref_norm2, reg=None):
    L_ref = thc_ovvo_build_Lbar(Xo_ref, Xv_ref, Xo, Xv, kmesh)
    synchronize()
    L = thc_ovvo_build_Lbar(Xo, Xv, Xo, Xv, kmesh)
    synchronize()
    rhs = (L_ref.transpose(-1, -2) @ W_ref.conj() @ L_ref.conj()).conj()
    synchronize()
    W = torch_lstsq_oinv_PSD(L, rhs, reg=reg)
    synchronize()

    ab = ((W.conj().transpose(-1, -2) @ rhs).diagonal(dim1=-1, dim2=-2).sum()).real
    synchronize()
    bb = ((W.conj().transpose(-1, -2) @ L @ W @ L.conj().transpose(-1, -2)).diagonal(dim1=-1, dim2=-2).sum()).real
    synchronize()
    error2 = (ref_norm2 + bb - 2.0 * ab).real
    return W, error2


@maybe_profile
def thc_ovvo_inner_from_mo(Xo_A, Xv_A, W_A, Xo_B, Xv_B, W_B, kmesh):
    Lbar = thc_ovvo_build_Lbar(Xo_A, Xv_A, Xo_B, Xv_B, kmesh)
    return (W_A.conj().transpose(-1, -2) @ Lbar @ W_B @ Lbar.conj().transpose(-1, -2)).diagonal(dim1=-1, dim2=-2).sum()


@maybe_profile
def thc_ovvo_error2_from_mo(Xo_A, Xv_A, W_A, Xo_B, Xv_B, W_B, kmesh):
    aa = thc_ovvo_inner_from_mo(Xo_A, Xv_A, W_A, Xo_A, Xv_A, W_A, kmesh)
    bb = thc_ovvo_inner_from_mo(Xo_B, Xv_B, W_B, Xo_B, Xv_B, W_B, kmesh)
    ab = thc_ovvo_inner_from_mo(Xo_A, Xv_A, W_A, Xo_B, Xv_B, W_B, kmesh)
    return (aa + bb - 2.0 * ab.real).real


def is_k_ordered(kpts_int, kmesh):
    A = np.moveaxis(np.stack(np.meshgrid(*[np.arange(n) for n in kmesh], indexing='ij'), axis=0), 0, -1)
    B = kpts_int.reshape(tuple(kmesh) + (len(kmesh),))
    return np.all(A == B)


def eri7d_to_full(tensor_7d, kpts_int, kmesh):
    nkpts, _, _, n1, n2, n3, n4 = tensor_7d.shape
    out = np.zeros((nkpts * n1, nkpts * n2, nkpts * n3, nkpts * n4), dtype=tensor_7d.dtype)
    kpt_map = {tuple(k): i for i, k in enumerate(kpts_int)}
    for k1 in range(nkpts):
        for k2 in range(nkpts):
            for k3 in range(nkpts):
                k4_int = (kpts_int[k1] - kpts_int[k2] + kpts_int[k3]) % kmesh
                k4 = kpt_map[tuple(k4_int)]
                s1 = slice(k1 * n1, (k1 + 1) * n1)
                s2 = slice(k2 * n2, (k2 + 1) * n2)
                s3 = slice(k3 * n3, (k3 + 1) * n3)
                s4 = slice(k4 * n4, (k4 + 1) * n4)
                out[s1, s2, s3, s4] = tensor_7d[k1, k2, k3].copy()
    return out


def mp2_from_ovov_7d(eri_ovov_7d, mo_energy, nocc, kpts_int, kmesh):
    nkpts = eri_ovov_7d.shape[0]
    nvir = eri_ovov_7d.shape[4]
    mo_energy = np.asarray(mo_energy)
    e_occ = mo_energy[:, :nocc]
    e_vir = mo_energy[:, nocc:nocc+nvir]
    kpt_map = {tuple(k): i for i, k in enumerate(kpts_int)}

    emp2 = 0.0
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb_int = (kpts_int[ki] - kpts_int[ka] + kpts_int[kj]) % kmesh
                kb = kpt_map[tuple(kb_int)]

                eia = e_occ[ki, :, None] - e_vir[ka]
                ejb = e_occ[kj, :, None] - e_vir[kb]
                eijab = eia[:, None, :, None] + ejb[None, :, None, :]

                gijab = eri_ovov_7d[ki, ka, kj].transpose(0, 2, 1, 3) / nkpts
                gijba = eri_ovov_7d[ki, kb, kj].transpose(0, 2, 1, 3) / nkpts

                t2 = np.conj(gijab / eijab)
                edi = np.einsum("ijab,ijab", t2, gijab, optimize=True).real
                exi = -np.einsum("ijab,ijba", t2, gijba, optimize=True).real
                emp2 += 2.0 * edi + exi

    emp2 /= nkpts
    return emp2


def mp2_from_ovov_full(eri_ovov_full, mo_energy, nocc):
    mo_energy = np.asarray(mo_energy)
    nkpts = mo_energy.shape[0]
    nvir = eri_ovov_full.shape[1] // nkpts

    e_occ = mo_energy[:, :nocc].reshape(nkpts * nocc)
    e_vir = mo_energy[:, nocc:nocc+nvir].reshape(nkpts * nvir)

    eijab = (
        e_occ[:, None, None, None]
        + e_occ[None, :, None, None]
        - e_vir[None, None, :, None]
        - e_vir[None, None, None, :]
    )

    gijab = eri_ovov_full.swapaxes(1, 2) / nkpts
    t2 = np.conj(gijab / eijab)

    emp2 = (
        2.0 * np.einsum("ijab,ijab", t2, gijab, optimize=True)
        - np.einsum("ijab,ijba", t2, gijab, optimize=True)
    ).real / nkpts
    return emp2


def compare_two_isdf(isdf_ref, isdf, Cocc, Cvir, kmesh):
    X_ref = torch.from_numpy(np.asarray(isdf_ref.inpv_kpt)).to(dtype=torch.complex128)
    W_ref = torch.from_numpy(np.asarray(isdf_ref.coul_kpt)).to(dtype=torch.complex128)
    X = torch.from_numpy(np.asarray(isdf.inpv_kpt)).to(dtype=torch.complex128)
    W = torch.from_numpy(np.asarray(isdf.coul_kpt)).to(dtype=torch.complex128)
    Cocc_t = torch.from_numpy(np.asarray(Cocc)).to(dtype=torch.complex128)
    Cvir_t = torch.from_numpy(np.asarray(Cvir)).to(dtype=torch.complex128)

    Xo_ref, Xv_ref = X_ref @ Cocc_t, X_ref @ Cvir_t
    Xo, Xv = X @ Cocc_t, X @ Cvir_t
    ref_norm2 = thc_ovvo_inner_from_mo(
        Xo_ref, Xv_ref, W_ref,
        Xo_ref, Xv_ref, W_ref,
        kmesh,
    ).real
    error2 = thc_ovvo_error2_from_mo(
        Xo_ref, Xv_ref, W_ref,
        Xo, Xv, W,
        kmesh,
    ).real
    return torch.sqrt(error2 / ref_norm2).item()


def stochastic_eri_diff(df1, df2, mo_coeff_kpts, nsamples):
    from pyscf.pbc import tools as pbctools

    kpts = np.asarray(df1.kpts)
    assert np.allclose(kpts, np.asarray(df2.kpts))

    kmesh = pbctools.k2gamma.kpts_to_kmesh(df1.cell, kpts - kpts[0])
    kmesh = np.asarray(kmesh, dtype=int)
    kpts_int = np.round(df1.cell.get_scaled_kpts(kpts) * kmesh).astype(int) % kmesh
    assert is_k_ordered(kpts_int, kmesh)

    if isinstance(mo_coeff_kpts, np.ndarray) and mo_coeff_kpts.ndim == 3:
        mo_coeff_kpts = [mo_coeff_kpts, ] * 4
    else:
        mo_coeff_kpts = list(mo_coeff_kpts)
    assert len(mo_coeff_kpts) == 4

    kpt_map = {tuple(k): i for i, k in enumerate(kpts_int)}
    nkpts = len(kpts)
    rng = np.random.default_rng()

    diff2 = np.empty(nsamples, dtype=np.float64)
    ref2 = np.empty(nsamples, dtype=np.float64)
    import tqdm
    for isample in tqdm.tqdm(range(nsamples)):
        k1, k2, k3 = rng.integers(0, nkpts, size=3)
        k4_int = (kpts_int[k1] - kpts_int[k2] + kpts_int[k3]) % kmesh
        k4 = kpt_map[tuple(k4_int)]
        kpts4 = kpts[[k1, k2, k3, k4]]
        mo_coeff_sample = []
        for mo, ki in zip(mo_coeff_kpts, (k1, k2, k3, k4)):
            mo_coeff_sample.append(mo[ki])

        eri1 = df1.ao2mo(mo_coeff_sample, kpts=kpts4).flatten()
        eri2 = df2.ao2mo(mo_coeff_sample, kpts=kpts4).flatten()
        diff2[isample] = np.linalg.norm(eri1 - eri2)**2
        ref2[isample] = np.linalg.norm(eri1)**2

    diff_mean = diff2.mean()
    ref_mean = ref2.mean()
    rel_diff = float(np.sqrt(diff_mean / ref_mean))

    if nsamples > 1:
        cov = np.cov(np.vstack((diff2, ref2)), ddof=1) / nsamples
        drel_ddiff = 0.5 / np.sqrt(diff_mean * ref_mean)
        drel_dref = -0.5 * np.sqrt(diff_mean) / (ref_mean ** 1.5)
        rel_var = (
            drel_ddiff * drel_ddiff * cov[0, 0]
            + drel_dref * drel_dref * cov[1, 1]
            + 2.0 * drel_ddiff * drel_dref * cov[0, 1]
        )
        rel_std = float(np.sqrt(max(rel_var, 0.0)))
    else:
        rel_std = 0.0

    return rel_diff, rel_std


def get_full_gdf_tensor(gdf, kpts_int, kmesh, C1=None, C2=None, progressbar=False):
    kpts = np.asarray(gdf.kpts)
    nkpts = len(kpts)
    nao = gdf.cell.nao_nr()
    kpt_map = {tuple(k): i for i, k in enumerate(kpts_int)}
    cderi = gdf.cderi_array()
    naux = cderi.load(kpts[0], kpts[0]).shape[0]
    if C1 is None:
        n1 = nao
    else:
        n1 = C1.shape[2]
    if C2 is None:
        n2 = nao
    else:
        n2 = C2.shape[2]
    out = np.zeros((nkpts * n1, nkpts * n2, nkpts * naux), dtype=np.complex128)

    kij = [(ki, kj) for ki in range(nkpts) for kj in range(nkpts)]
    if progressbar:
        import tqdm
        kij = tqdm.tqdm(kij, desc="Loading GDF tensor blocks", unit="block")
    for ki, kj in kij:
        si = slice(ki * n1, (ki + 1) * n1)
        sj = slice(kj * n2, (kj + 1) * n2)
        block = cderi.load(kpts[ki], kpts[kj])
        if block.shape[1] == nao * (nao + 1) // 2:
            block = lib.unpack_tril(block)
        block = block.reshape(naux, nao, nao).transpose(1, 2, 0)
        if C1 is not None:
            block = np.einsum("pi,pqL->iqL", C1[ki].conj(), block, optimize=True)
        if C2 is not None:
            block = np.einsum("qj,iqL->ijL", C2[kj], block, optimize=True)
        q_int = (kpts_int[kj] - kpts_int[ki]) % kmesh
        q = kpt_map[tuple(q_int)]
        sq = slice(q * naux, (q + 1) * naux)
        out[si, sj, sq] = block
    return out


def get_gdf_eri_eigvalsh(gdf, kpts_int, kmesh, C1=None, C2=None, progressbar=False):
    kpts = np.asarray(gdf.kpts)
    nkpts = len(kpts)
    nao = gdf.cell.nao_nr()
    kpt_map = {tuple(k): i for i, k in enumerate(kpts_int)}
    cderi = gdf.cderi_array()
    naux = cderi.load(kpts[0], kpts[0]).shape[0]
    if C1 is None:
        n1 = nao
    else:
        n1 = C1.shape[2]
    if C2 is None:
        n2 = nao
    else:
        n2 = C2.shape[2]
    metric = np.zeros((nkpts, naux, naux), dtype=np.complex128)

    kij = [(ki, kj) for ki in range(nkpts) for kj in range(nkpts)]
    if progressbar:
        import tqdm
        kij = tqdm.tqdm(kij, desc="Accumulating GDF metric blocks", unit="block")
    for ki, kj in kij:
        block = cderi.load(kpts[ki], kpts[kj])
        if block.shape[1] == nao * (nao + 1) // 2:
            block = lib.unpack_tril(block)
        block = block.reshape(naux, nao, nao).transpose(1, 2, 0)
        if C1 is not None:
            block = np.einsum("pi,pqL->iqL", C1[ki].conj(), block, optimize=True)
        if C2 is not None:
            block = np.einsum("qj,iqL->ijL", C2[kj], block, optimize=True)
        q_int = (kpts_int[kj] - kpts_int[ki]) % kmesh
        q = kpt_map[tuple(q_int)]
        block = block.reshape(n1 * n2, naux)
        metric[q] += block.conj().T @ block

    return np.linalg.eigvalsh(metric)


def borrow_ij_pairs(dim, q, bout, bin_all):
    pairs = []
    for bin_ in bin_all:
        for i in range(dim):
            for j in range(dim):
                t = j - i - bin_
                q_ = t % dim
                bout_ = 1 if t < 0 else 0
                if q_ == q and (bout is None or bout_ == bout):
                    pairs.append((i, j))
    return pairs


def _mpo_borrow_block_norm_site(T, isite, nsite):
    if T.ndim == 3 and isite == 0:
        dim, _, right = T.shape
        T = T.reshape(1, dim, dim, right)
    elif T.ndim == 3 and isite == nsite - 1:
        left, dim, _ = T.shape
        T = T.reshape(left, dim, dim, 1)

    left, dim, _, right = T.shape

    if isite == 0:
        bin_all = [0]
    else:
        bin_all = [0, 1]

    if isite == nsite - 1:
        bout_all = [None]
    else:
        bout_all = [0, 1]

    norms = []
    for q in range(dim):
        for bout in bout_all:
            pairs = borrow_ij_pairs(dim, q, bout, bin_all)
            if pairs:
                blocks = []
                for i, j in pairs:
                    blocks.append(T[:, i, j, :].reshape(left, right))
                M = np.concatenate(blocks, axis=0)
                norms.append(np.linalg.svd(M, compute_uv=False)[0])
            else:
                norms.append(0.0)
    return max(norms), norms


def mpo_borrow_block_norm(tensors, return_blocks=False):
    nsite = len(tensors)
    all_norms = []
    all_block_norms = []
    for isite, T in enumerate(tensors):
        norm, block_norms = _mpo_borrow_block_norm_site(T, isite, nsite)
        all_norms.append(norm)
        all_block_norms.append(block_norms)
    if return_blocks:
        return all_norms, all_block_norms
    return all_norms
