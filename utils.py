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
    return np.einsum("...ai,...i,...bi->...ab", eigvecs, eigvals, eigvecs.conj(), optimize=True)


def matrix_power(A, power):
    return matrix_operation(A, lambda x: x ** power)


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


def _get_kpt_map(kpts_int):
    return {tuple(k): i for i, k in enumerate(kpts_int)}


def _lookup_kpts(kpts_int_query, kpts_int):
    kpt_map = _get_kpt_map(kpts_int)
    idx = np.empty(kpts_int_query.shape[:-1], dtype=np.int64)
    for i in np.ndindex(idx.shape):
        idx[i] = kpt_map[tuple(kpts_int_query[i])]
    return idx


def _get_gdf_layout_indices(kpts_int, kmesh, layout):
    nkpts = len(kpts_int)
    idx0 = np.arange(nkpts)[:, None]
    idx1 = np.arange(nkpts)[None, :]
    if layout == "k1q":
        k1 = np.broadcast_to(idx0, (nkpts, nkpts))
        q = np.broadcast_to(idx1, (nkpts, nkpts))
        k2_int = (kpts_int[k1] + kpts_int[q]) % kmesh
        k2 = _lookup_kpts(k2_int, kpts_int)
    elif layout == "k1k2":
        k1 = np.broadcast_to(idx0, (nkpts, nkpts))
        k2 = np.broadcast_to(idx1, (nkpts, nkpts))
        q_int = (kpts_int[k2] - kpts_int[k1]) % kmesh
        q = _lookup_kpts(q_int, kpts_int)
    elif layout == "k2q":
        k2 = np.broadcast_to(idx0, (nkpts, nkpts))
        q = np.broadcast_to(idx1, (nkpts, nkpts))
        k1_int = (kpts_int[k2] - kpts_int[q]) % kmesh
        k1 = _lookup_kpts(k1_int, kpts_int)
    else:
        raise ValueError("layout should be k1q, k1k2, or k2q")
    return k1, k2, q


def get_gdf_tensor_compact(gdf, kpts_int, kmesh, C1=None, C2=None, *, layout, progressbar=False):
    kpts = np.asarray(gdf.kpts)
    nkpts = len(kpts)
    nao = gdf.cell.nao_nr()
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
    R = np.empty((nkpts, nkpts, naux, n1, n2), dtype=np.complex128)

    k1_all, k2_all, q_all = _get_gdf_layout_indices(kpts_int, kmesh, layout)
    pairs = [(i, j) for i in range(nkpts) for j in range(nkpts)]
    if progressbar:
        import tqdm
        pairs = tqdm.tqdm(pairs, desc=f"Loading GDF tensor R[{layout}]", unit="block")
    for i, j in pairs:
        k1 = k1_all[i, j]
        k2 = k2_all[i, j]
        block = cderi.load(kpts[k1], kpts[k2])
        if block.shape[1] == nao * (nao + 1) // 2:
            block = lib.unpack_tril(block)
        block = block.reshape(naux, nao, nao)
        if C1 is not None:
            block = C1[k1].conj().T @ block
        if C2 is not None:
            block = block @ C2[k2]
        R[i, j] = block
    return R


def gdf_compact_to_full(tensor_in, kpts_int, kmesh, *, layout):
    nkpts = tensor_in.shape[0]
    assert tensor_in.shape[1] == nkpts
    k1_all, k2_all, q_all = _get_gdf_layout_indices(kpts_int, kmesh, layout)
    tensor = np.zeros((nkpts, nkpts, nkpts) + tensor_in.shape[2:], dtype=tensor_in.dtype)
    tensor[k1_all, k2_all, q_all] = tensor_in
    return tensor


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
    kmesh = tuple(kmesh)
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
def thc_ovvo_build_Lbar(Xo_A, Xv_A, Xo_B, Xv_B, kmesh):
    # contract four W and get a (q, I, j) object
    nkpts, naux_A = Xo_A.shape[:2]
    sqrt_nkpts = np.sqrt(nkpts).item()
    naux_B = Xo_B.shape[1]
    assert nkpts == int(np.prod(kmesh))

    O = torch.einsum("kIi,kJi->kIJ", Xo_A, Xo_B.conj())
    V = torch.einsum("kIa,kJa->kIJ", Xv_A.conj(), Xv_B)

    O_fft = fourier_transform_3d(O, axis=0, kmesh=kmesh, inverse=False)
    V_fft = fourier_transform_3d(V, axis=0, kmesh=kmesh, inverse=False)

    negative = negative_k(torch.arange(nkpts), kmesh)
    Lbar = O_fft[negative] * V_fft
    Lbar = fourier_transform_3d(Lbar, axis=0, kmesh=kmesh, inverse=True)

    return Lbar.reshape(nkpts, naux_A, naux_B) * sqrt_nkpts


@maybe_profile
def thc_build_Lbar(X_A, X_B, kmesh):
    nkpts, naux_A = X_A.shape[:2]
    sqrt_nkpts = np.sqrt(nkpts).item()
    naux_B = X_B.shape[1]
    assert nkpts == int(np.prod(kmesh))

    O = torch.einsum("kIa,kJa->kIJ", X_A, X_B.conj())
    O_fft = fourier_transform_3d(O, axis=0, kmesh=kmesh, inverse=False)

    negative = negative_k(torch.arange(nkpts, device=X_A.device), kmesh)
    Lbar = (O_fft[negative].abs()**2).to(dtype=O_fft.dtype)
    Lbar = fourier_transform_3d(Lbar, axis=0, kmesh=kmesh, inverse=True)

    return Lbar.reshape(nkpts, naux_A, naux_B) * sqrt_nkpts


def thc_inner_from_L(W_A, L, W_B, by_q=False):
    value = (W_A.conj().transpose(-1, -2) @ L @ W_B @ L.conj().transpose(-1, -2)).diagonal(dim1=-1, dim2=-2).sum(axis=-1)
    if by_q:
        return value
    return value.sum()


@maybe_profile
def thc_ovvo_inner_from_mo(Xo_A, Xv_A, W_A, Xo_B, Xv_B, W_B, kmesh, by_q=False):
    Lbar = thc_ovvo_build_Lbar(Xo_A, Xv_A, Xo_B, Xv_B, kmesh)
    return thc_inner_from_L(W_A, Lbar, W_B, by_q=by_q)


@maybe_profile
def thc_lrdf_ov_build_Lbar(Xo, Xv, D, kmesh):
    """Build the mixed Gram tensor for THC pair factors and D[k,r,i,a]."""
    nkpts = int(np.prod(kmesh))
    sqrt_nkpts = np.sqrt(nkpts).item()
    assert Xo.shape[0] == Xv.shape[0] == D.shape[0] == nkpts

    A = torch.einsum("kIi,kria->kIra", Xo, D.conj())
    A_fft = fourier_transform_3d(A, axis=0, kmesh=kmesh, inverse=False)
    Xv_fft = fourier_transform_3d(Xv.conj(), axis=0, kmesh=kmesh, inverse=False)

    negative = negative_k(torch.arange(nkpts, device=Xo.device), kmesh)
    Lbar_fft = torch.einsum("qIra,qIa->qIr", A_fft[negative], Xv_fft)
    Lbar = fourier_transform_3d(Lbar_fft, axis=0, kmesh=kmesh, inverse=True)
    return Lbar * sqrt_nkpts


@maybe_profile
def thc_lrdf_ov_inner_from_mo(Xo, Xv, W, D, G, kmesh, by_q=False):
    Lbar = thc_lrdf_ov_build_Lbar(Xo, Xv, D, kmesh)
    return thc_inner_from_L(W, Lbar, G, by_q=by_q)


@maybe_profile
def lrdf_ov_build_Lbar(D_A, D_B, kmesh):
    assert D_A.shape[0] == D_B.shape[0] == int(np.prod(kmesh))
    return torch.einsum("kria,ksia->rs", D_A, D_B.conj())


@maybe_profile
def lrdf_ov_inner_from_mo(D_A, G_A, D_B, G_B, kmesh, by_q=False):
    Lbar = lrdf_ov_build_Lbar(D_A, D_B, kmesh)
    return thc_inner_from_L(G_A, Lbar, G_B, by_q=by_q)


@maybe_profile
def thc_thclrdf_ov_solve_w_intermediate_from_mo(Xo_ref, Xv_ref, W_ref, Xo, Xv, D, G, kmesh, reg=None):
    L = thc_ovvo_build_Lbar(Xo, Xv, Xo, Xv, kmesh)
    L_ref = thc_ovvo_build_Lbar(Xo_ref, Xv_ref, Xo, Xv, kmesh)
    rhs_W = L_ref.conj().transpose(-1, -2) @ W_ref @ L_ref

    C_mix = thc_lrdf_ov_build_Lbar(Xo, Xv, D, kmesh)
    L_D = lrdf_ov_build_Lbar(D, D, kmesh)
    C_ref = thc_lrdf_ov_build_Lbar(Xo_ref, Xv_ref, D, kmesh)
    rhs_G = C_ref.conj().transpose(-1, -2) @ W_ref @ C_ref

    rhs_W_res = rhs_W - C_mix @ G @ C_mix.conj().transpose(-1, -2)
    W = torch_lstsq_oinv_PSD(L, rhs_W_res, reg=reg)
    W = (W + W.conj().transpose(-1, -2)) / 2.0
    return W, L, rhs_W, L_D, C_mix, rhs_G


@maybe_profile
def thc_thclrdf_ov_solve_w_from_mo(Xo_ref, Xv_ref, W_ref, Xo, Xv, D, G, kmesh, reg=None):
    W, L, rhs_W, L_D, C_mix, rhs_G = thc_thclrdf_ov_solve_w_intermediate_from_mo(
        Xo_ref, Xv_ref, W_ref, Xo, Xv, D, G, kmesh, reg=reg,
    )
    return W


@maybe_profile
def thc_thclrdf_ov_solve_w_error2_from_mo(Xo_ref, Xv_ref, W_ref, Xo, Xv, D, G, kmesh, ref_norm2, reg=None):
    W, L, rhs_W, L_D, C_mix, rhs_G = thc_thclrdf_ov_solve_w_intermediate_from_mo(
        Xo_ref, Xv_ref, W_ref, Xo, Xv, D, G, kmesh, reg=reg,
    )
    norm2_thc = thc_inner_from_L(W, L, W).real
    norm2_lrdf = thc_inner_from_L(G, L_D, G).real
    cross_thc_lrdf = thc_inner_from_L(W, C_mix, G).real
    cross_ref_thc = (W.conj().transpose(-1, -2) @ rhs_W).diagonal(dim1=-1, dim2=-2).sum().real
    cross_ref_lrdf = (G.conj().transpose(-1, -2) @ rhs_G).diagonal(dim1=-1, dim2=-2).sum().real
    error2 = ref_norm2 + norm2_thc + norm2_lrdf + 2.0 * cross_thc_lrdf - 2.0 * cross_ref_thc - 2.0 * cross_ref_lrdf
    return W, error2


@maybe_profile
def thc_ovvo_error2_from_mo(Xo_A, Xv_A, W_A, Xo_B, Xv_B, W_B, kmesh, by_q=False):
    aa = thc_ovvo_inner_from_mo(Xo_A, Xv_A, W_A, Xo_A, Xv_A, W_A, kmesh, by_q=by_q)
    bb = thc_ovvo_inner_from_mo(Xo_B, Xv_B, W_B, Xo_B, Xv_B, W_B, kmesh, by_q=by_q)
    ab = thc_ovvo_inner_from_mo(Xo_A, Xv_A, W_A, Xo_B, Xv_B, W_B, kmesh, by_q=by_q)
    return (aa + bb - 2.0 * ab.real).real


@maybe_profile
def thc_ovvo_solve_w_intermediate_from_mo(Xo_ref, Xv_ref, W_ref, Xo, Xv, kmesh, reg=None):
    L_mix = thc_ovvo_build_Lbar(Xo_ref, Xv_ref, Xo, Xv, kmesh)
    L = thc_ovvo_build_Lbar(Xo, Xv, Xo, Xv, kmesh)
    rhs = L_mix.conj().transpose(-1, -2) @ W_ref @ L_mix
    W = torch_lstsq_oinv_PSD(L, rhs, reg=reg)
    return W, L, rhs


def thc_solve_w_error2_from_intermediate(W, L, rhs, ref_norm2):
    ab = ((W.conj().transpose(-1, -2) @ rhs).diagonal(dim1=-1, dim2=-2).sum()).real
    bb = thc_inner_from_L(W, L, W).real
    return (ref_norm2 + bb - 2.0 * ab).real


@maybe_profile
def thc_ovvo_solve_w_from_mo(Xo_ref, Xv_ref, W_ref, Xo, Xv, kmesh, reg=None):
    W, L, rhs = thc_ovvo_solve_w_intermediate_from_mo(Xo_ref, Xv_ref, W_ref, Xo, Xv, kmesh, reg=reg)
    return W


@maybe_profile
def thc_ovvo_solve_w_error2_from_mo(Xo_ref, Xv_ref, W_ref, Xo, Xv, kmesh, ref_norm2, reg=None):
    W, L, rhs = thc_ovvo_solve_w_intermediate_from_mo(Xo_ref, Xv_ref, W_ref, Xo, Xv, kmesh, reg=reg)
    error2 = thc_solve_w_error2_from_intermediate(W, L, rhs, ref_norm2)
    return W, error2


def thc_inner_from_mo(X_A, W_A, X_B, W_B, kmesh, by_q=False):
    Lbar = thc_build_Lbar(X_A, X_B, kmesh)
    return thc_inner_from_L(W_A, Lbar, W_B, by_q=by_q)


@maybe_profile
def thc_error2_from_mo(X_A, W_A, X_B, W_B, kmesh, by_q=False):
    aa = thc_inner_from_mo(X_A, W_A, X_A, W_A, kmesh, by_q=by_q)
    bb = thc_inner_from_mo(X_B, W_B, X_B, W_B, kmesh, by_q=by_q)
    ab = thc_inner_from_mo(X_A, W_A, X_B, W_B, kmesh, by_q=by_q)
    return (aa + bb - 2.0 * ab.real).real


@maybe_profile
def thc_solve_w_intermediate_from_mo(X_ref, W_ref, X, kmesh, reg=None):
    L_mix = thc_build_Lbar(X_ref, X, kmesh)
    L = thc_build_Lbar(X, X, kmesh)
    rhs = L_mix.conj().transpose(-1, -2) @ W_ref @ L_mix
    W = torch_lstsq_oinv_PSD(L, rhs, reg=reg)
    return W, L, rhs


@maybe_profile
def thc_solve_w_from_mo(X_ref, W_ref, X, kmesh, reg=None):
    W, L, rhs = thc_solve_w_intermediate_from_mo(X_ref, W_ref, X, kmesh, reg=reg)
    return W


@maybe_profile
def thc_solve_w_error2_from_mo(X_ref, W_ref, X, kmesh, ref_norm2, reg=None):
    W, L, rhs = thc_solve_w_intermediate_from_mo(X_ref, W_ref, X, kmesh, reg=reg)
    error2 = thc_solve_w_error2_from_intermediate(W, L, rhs, ref_norm2)
    return W, error2


@maybe_profile
def thc_df_ov_build_A_from_mo(Xo, Xv, R, kmesh):
    nkpts = int(np.prod(kmesh))
    k = torch.arange(nkpts, device=Xo.device)
    A = torch.empty((nkpts, Xo.shape[1], R.shape[2]), dtype=R.dtype, device=R.device)
    for q in range(nkpts):
        kq = add_k(k, q, kmesh).to(device=Xo.device)
        A[q] = torch.einsum("kIi,kIa,kxia->Ix", Xo, Xv[kq].conj(), R[:, q])
    return A


@maybe_profile
def thc_df_ov_rhs_from_mo(Xo, Xv, R, kmesh):
    A = thc_df_ov_build_A_from_mo(Xo, Xv, R, kmesh)
    B = torch.einsum("qIx,qJx->qIJ", A, A.conj())
    return B


@maybe_profile
def thc_df_ov_inner_from_mo(Xo, Xv, W, R, kmesh, by_q=False):
    B = thc_df_ov_rhs_from_mo(Xo, Xv, R, kmesh)
    value = torch.einsum("qIJ,qIJ->q", W.conj(), B)
    if by_q:
        return value
    return value.sum()


@maybe_profile
def thc_df_inner_from_mo(X, W, R, kmesh, by_q=False):
    return thc_df_ov_inner_from_mo(X, X, W, R, kmesh, by_q=by_q)


@maybe_profile
def thc_df_ov_error2_from_mo(Xo, Xv, W, R, kmesh, df_norm2, by_q=False):
    aa = thc_ovvo_inner_from_mo(Xo, Xv, W, Xo, Xv, W, kmesh, by_q=by_q).real
    ab = thc_df_ov_inner_from_mo(Xo, Xv, W, R, kmesh, by_q=by_q).real
    return (aa + df_norm2 - 2.0 * ab).real


@maybe_profile
def thc_df_error2_from_mo(X, W, R, kmesh, df_norm2, by_q=False):
    return thc_df_ov_error2_from_mo(X, X, W, R, kmesh, df_norm2, by_q=by_q)


@maybe_profile
def thc_df_ov_solve_w_intermediate_from_mo(Xo, Xv, R, kmesh, reg=None):
    A = thc_ovvo_build_Lbar(Xo, Xv, Xo, Xv, kmesh)
    B = thc_df_ov_rhs_from_mo(Xo, Xv, R, kmesh)
    W = torch_lstsq_oinv_PSD(A, B, reg=reg)
    return W, A, B


@maybe_profile
def thc_df_ov_solve_w_from_mo(Xo, Xv, R, kmesh, reg=None):
    W, A, B = thc_df_ov_solve_w_intermediate_from_mo(Xo, Xv, R, kmesh, reg=reg)
    return W


@maybe_profile
def thc_df_solve_w_intermediate_from_mo(X, R, kmesh, reg=None):
    A = thc_build_Lbar(X, X, kmesh)
    B = thc_df_ov_rhs_from_mo(X, X, R, kmesh)
    W = torch_lstsq_oinv_PSD(A, B, reg=reg)
    return W, A, B


@maybe_profile
def thc_df_solve_w_from_mo(X, R, kmesh, reg=None):
    W, A, B = thc_df_solve_w_intermediate_from_mo(X, R, kmesh, reg=reg)
    return W


@maybe_profile
def thc_df_ov_solve_w_error2_from_mo(Xo, Xv, R, kmesh, df_norm2, reg=None):
    W, A, B = thc_df_ov_solve_w_intermediate_from_mo(Xo, Xv, R, kmesh, reg=reg)
    error2 = thc_solve_w_error2_from_intermediate(W, A, B, df_norm2)
    return W, error2


@maybe_profile
def thc_df_solve_w_error2_from_mo(X, R, kmesh, df_norm2, reg=None):
    W, A, B = thc_df_solve_w_intermediate_from_mo(X, R, kmesh, reg=reg)
    error2 = thc_solve_w_error2_from_intermediate(W, A, B, df_norm2)
    return W, error2


@maybe_profile
def df_inner_from_mo(R, kmesh, by_q=False):
    nkpts = int(np.prod(kmesh))
    result = torch.zeros((nkpts,), dtype=R.dtype, device=R.device)
    for q in range(nkpts):
        M = torch.einsum("kxab,kyab->xy", R[:, q].conj(), R[:, q])
        result[q] = torch.einsum("xy,xy->", M, M.conj())
    if by_q:
        return result
    return result.sum()


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


def compare_two_isdf(isdf_ref, isdf, kmesh, C1=None, C2=None, by_q=False, abs=False):
    X_ref = torch.from_numpy(np.asarray(isdf_ref.inpv_kpt)).to(dtype=torch.complex128)
    W_ref = torch.from_numpy(np.asarray(isdf_ref.coul_kpt)).to(dtype=torch.complex128)
    X = torch.from_numpy(np.asarray(isdf.inpv_kpt)).to(dtype=torch.complex128)
    W = torch.from_numpy(np.asarray(isdf.coul_kpt)).to(dtype=torch.complex128)

    if C1 is None:
        X1_ref = X_ref
        X1 = X
    else:
        C1_t = torch.from_numpy(np.asarray(C1)).to(dtype=torch.complex128)
        X1_ref = X_ref @ C1_t
        X1 = X @ C1_t

    if C2 is None:
        X2_ref = X_ref
        X2 = X
    else:
        C2_t = torch.from_numpy(np.asarray(C2)).to(dtype=torch.complex128)
        X2_ref = X_ref @ C2_t
        X2 = X @ C2_t

    ref_norm2 = thc_ovvo_inner_from_mo(
        X1_ref, X2_ref, W_ref,
        X1_ref, X2_ref, W_ref,
        kmesh,
        by_q=by_q,
    ).real
    error2 = thc_ovvo_error2_from_mo(
        X1_ref, X2_ref, W_ref,
        X1, X2, W,
        kmesh,
        by_q=by_q,
    ).real
    ref_norm = torch.sqrt(ref_norm2).detach().cpu().numpy()
    error = torch.sqrt(error2).detach().cpu().numpy()
    if abs:
        return error, ref_norm
    else:
        return error / ref_norm
