import numpy as np
import torch

from pyscf.pbc.symm import Symmetry
import pyscf.pbc.symm.symmetry


# Utilities for symmetry-preserving ISDF point selection.
#
# The central object is the uniform real-space grid used by ISDF. A space-group
# operation maps a grid point to another grid point, possibly after translating
# by a lattice vector. For interpolation points to preserve symmetry, whenever
# one grid point is selected, every grid point in its point-group orbit should
# be selected as well.


def frac_to_mesh_idx(frac, mesh):
    # Convert fractional coordinates to integer labels on a uniform mesh.
    # The modulo removes lattice translations, so points in different periodic
    # images but at the same unit-cell grid position get the same label.
    frac = np.mod(frac, 1.0)
    idx_float = frac * mesh[None, :]
    idx = np.rint(idx_float).astype(np.int64) % mesh[None, :]
    resid = np.abs(idx_float - np.rint(idx_float))
    return idx, resid.max()


def build_grid_perm(idx0, idx1):
    # Build the permutation p such that grid point i is mapped to p[i].
    # idx0 is the original mesh labeling, idx1 is the transformed labeling.
    idx_map = {tuple(idx): i for i, idx in enumerate(idx0)}
    perm = np.array([idx_map[tuple(idx)] for idx in idx1], dtype=np.int64)
    return perm


def finite_grid_transform(cell, coords, op, mesh):
    # Space-group operation on a finite uniform grid:
    # r -> R r + t = r_perm + L, with L a lattice vector.
    mesh = np.asarray(mesh, dtype=np.int64)
    coords = np.asarray(coords)
    a_inv = np.linalg.inv(cell.lattice_vectors())
    frac0 = coords @ a_inv
    frac0_mod = np.mod(frac0, 1.0)
    idx0, resid0 = frac_to_mesh_idx(frac0_mod, mesh)

    frac1 = frac0_mod @ op.rot.T + op.trans[None, :]
    idx1, resid1 = frac_to_mesh_idx(frac1, mesh)
    perm = build_grid_perm(idx0, idx1)

    coords_g = (frac0 @ op.rot.T + op.trans[None, :]) @ cell.lattice_vectors()
    L_frac = (coords_g - coords[perm]) @ a_inv
    return perm, L_frac, max(resid0, resid1)


def build_symmetry_kpts(cell, kmesh, time_reversal=False):
    # Build PySCF's full-BZ k-point object with space-group metadata. The
    # object contains both the symmetry operations and the map k -> gk.
    cell_sym = cell.copy()
    cell_sym.space_group_symmetry = True
    cell_sym.symmorphic = False
    cell_sym.build(False, False)
    kpts = cell_sym.make_kpts(
        kmesh,
        space_group_symmetry=True,
        time_reversal_symmetry=time_reversal,
    )
    return cell_sym, kpts


def ao_rotation(cell, kpt_scaled, A, op, Dmats):
    # AO representation U_g(k) of a space-group operation. PySCF includes the
    # atom permutation, angular-momentum rotation, and Bloch phase associated
    # with atoms crossing unit-cell boundaries.
    return pyscf.pbc.symm.symmetry._get_rotation_mat(
        cell,
        kpt_scaled,
        A,
        op,
        Dmats,
    )


def selected_grid_transform(cell, coords, ix_sel, kpts, iop, mesh=None):
    # For selected ISDF points, build the finite-grid map I -> gI and the
    # lattice translation L_I defined by g r_I = r_{gI} + L_I.
    if mesh is None:
        mesh = cell.mesh
    ix_sel = np.asarray(ix_sel, dtype=np.int64)
    op = kpts.ops[iop]
    perm_grid, L_frac, resid = finite_grid_transform(cell, coords, op, mesh)
    assert resid < 1e-10
    ix_to_I = {int(ix): I for I, ix in enumerate(ix_sel)}
    perm_I = np.empty(len(ix_sel), dtype=np.int64)
    for I, ix in enumerate(ix_sel):
        ix1 = int(perm_grid[ix])
        assert ix1 in ix_to_I
        perm_I[I] = ix_to_I[ix1]
    L_frac_I = L_frac[ix_sel]
    return perm_I, L_frac_I


def valid_isdf_op_ids(cell, coords, ix_sel, kpts, mesh=None, tol=1e-6):
    # A crystal/k-point symmetry is usable for ISDF only if it also acts as an
    # exact permutation of the chosen finite grid and closes the selected points.
    if mesh is None:
        mesh = cell.mesh
    ix_sel = np.asarray(ix_sel, dtype=np.int64)
    ix_set = set(int(ix) for ix in ix_sel)
    op_ids = []
    for iop, op in enumerate(kpts.ops):
        if np.any(kpts.k2opk[:, iop] < 0):
            continue
        perm_grid, _L_frac, resid = finite_grid_transform(cell, coords, op, mesh)
        if resid >= tol:
            continue
        if all(int(perm_grid[ix]) in ix_set for ix in ix_sel):
            op_ids.append(iop)
    return np.asarray(op_ids, dtype=np.int64)


class PBCSymmetry:
    # Grid-independent PBC symmetry data. This object knows how a symmetry
    # operation acts on k labels and AO indices, but it deliberately does not
    # store any ISDF-grid information such as selected points or grid phases.
    def __init__(self, cell, kmesh, kpts, dtype=torch.complex128, device="cpu"):
        self.cell = cell
        self.dtype = dtype
        self.device = device

        _cell_sym, kpts_symm = build_symmetry_kpts(cell, kmesh)
        self.kpts = kpts_symm
        kpts_scaled = cell.get_scaled_kpts(kpts)
        assert np.max(np.abs(kpts_symm.kpts_scaled - kpts_scaled)) < 1e-10

        op_ids = np.where(np.all(kpts_symm.k2opk >= 0, axis=0))[0]
        self.op_ids = op_ids
        self.kmap = torch.from_numpy(kpts_symm.k2opk[:, op_ids]).to(dtype=torch.long, device=device)

        # U[iop, k] is the AO rotation matrix for operation iop at k.  PySCF's
        # helper includes atom permutations, angular momentum rotations, and the
        # Bloch phase from atoms crossing unit-cell boundaries.
        U = []
        nao = cell.nao_nr()
        for iop in op_ids:
            op = kpts_symm.ops[iop]
            Dmats = kpts_symm.Dmats[iop]
            U_i = []
            for kpt_scaled in kpts_symm.kpts_scaled:
                U_i.append(ao_rotation(cell, kpt_scaled, np.empty((nao, nao)), op, Dmats))
            U.append(U_i)

        self.U = torch.from_numpy(np.asarray(U)).to(dtype=dtype, device=device)

    @property
    def nops(self):
        return len(self.op_ids)

    def apply_matrix(self, T, M, axis):
        # out[..., a_new, ...] = sum_a_old M[a_new, a_old] T[..., a_old, ...]
        T = torch.moveaxis(T, axis, 0)
        T = torch.einsum("ab,b...->a...", M, T)
        return torch.moveaxis(T, 0, axis)

    def transform_k(self, T, iop, axis=0):
        T = torch.moveaxis(T, axis, 0)
        out = torch.empty_like(T)
        for k, kp in enumerate(self.kmap[:, iop]):
            out[kp] = T[k]
        return torch.moveaxis(out, 0, axis)

    def transform_ao(self, T, iop, axis, kaxis=0, dagger=False):
        axis = axis % T.ndim
        kaxis = kaxis % T.ndim
        out = torch.empty_like(T)
        for k in range(T.shape[kaxis]):
            U = self.U[iop, k]
            if dagger:
                U = U.conj()
            idx = [slice(None)] * T.ndim
            idx[kaxis] = k
            out[tuple(idx)] = self.apply_matrix(T[tuple(idx)], U, axis if axis < kaxis else axis - 1)
        return out


def build_isdf_grid_transform(symm, coords, ix_sel, mesh=None):
    # ISDF-specific add-on to PBCSymmetry. For selected points r_I, construct
    # perm and phase from
    #
    #     g r_I = r_{perm[I]} + L_I,
    #     phase[k, I] = exp(i 2*pi k . L_I).
    #
    # The assertion means the selected grid must support every operation stored
    # in symm. If later we want partial finite-grid symmetry, this is the place
    # to return an ISDF-specific op list instead.
    if mesh is None:
        mesh = symm.cell.mesh
    coords = np.asarray(coords)
    ix_sel = np.asarray(ix_sel, dtype=np.int64)

    op_ids = valid_isdf_op_ids(symm.cell, coords, ix_sel, symm.kpts, mesh=mesh)
    assert np.array_equal(op_ids, symm.op_ids)

    perm = []
    phase = []
    for iop in symm.op_ids:
        perm_I, L_frac_I = selected_grid_transform(symm.cell, coords, ix_sel, symm.kpts, iop, mesh=mesh)
        perm.append(perm_I)

        kmap = symm.kpts.k2opk[:, iop]
        phase_i = []
        for k in range(len(symm.kpts.kpts_scaled)):
            kp = kmap[k]
            phase_i.append(np.exp(1j * 2.0 * np.pi * (L_frac_I @ symm.kpts.kpts_scaled[kp])))
        phase.append(phase_i)

    perm = torch.from_numpy(np.asarray(perm)).to(dtype=torch.long, device=symm.device)
    phase = torch.from_numpy(np.asarray(phase)).to(dtype=symm.dtype, device=symm.device)
    return perm, phase


def transform_grid(T, perm, phase, iop, axis, kaxis=0, conjugate=False):
    # Apply one selected-grid index transformation. This is separate from
    # PBCSymmetry because perm/phase depend on the particular ISDF point set.
    axis = axis % T.ndim
    kaxis = kaxis % T.ndim
    out = torch.empty_like(T)
    for k in range(T.shape[kaxis]):
        phase_k = phase[iop, k]
        if conjugate:
            phase_k = phase_k.conj()
        idx = [slice(None)] * T.ndim
        idx[kaxis] = k
        axis0 = axis if axis < kaxis else axis - 1
        T0 = torch.moveaxis(T[tuple(idx)], axis0, 0)
        Y = torch.empty_like(T0)
        shape = (len(phase_k),) + (1,) * (T0.ndim - 1)
        Y[perm[iop]] = phase_k.reshape(shape) * T0
        out[tuple(idx)] = torch.moveaxis(Y, 0, axis0)
    return out


def transform_isdf_X(symm, X, perm, phase, iop):
    # X[k, I, a] has one k index, one selected-grid index, and one AO index.
    # Transform each index explicitly: AO, grid, then k label.
    X = symm.transform_ao(X, iop, axis=2, kaxis=0, dagger=True)
    X = transform_grid(X, perm, phase, iop, axis=1, kaxis=0, conjugate=True)
    X = symm.transform_k(X, iop, axis=0)
    return X


def transform_isdf_W(symm, W, perm, phase, iop):
    # W[q, I, J] has two selected-grid indices. The first carries the conjugate
    # grid phase and the second carries the direct grid phase.
    W = transform_grid(W, perm, phase, iop, axis=1, kaxis=0, conjugate=True)
    W = transform_grid(W, perm, phase, iop, axis=2, kaxis=0)
    W = symm.transform_k(W, iop, axis=0)
    return W


def symmetrize_isdf_X(symm, X, perm, phase):
    X_avg = torch.zeros_like(X)
    for iop in range(symm.nops):
        X_avg = X_avg + transform_isdf_X(symm, X, perm, phase, iop)
    return X_avg / symm.nops


def symmetrize_isdf_W(symm, W, perm, phase):
    W_avg = torch.zeros_like(W)
    for iop in range(symm.nops):
        W_avg = W_avg + transform_isdf_W(symm, W, perm, phase, iop)
    return W_avg / symm.nops


def symmetrize_ao_operator(symm, A):
    # Average an AO-index operator over all symmetry operations in the full BZ.
    A_avg = torch.zeros_like(A)
    for iop in range(symm.nops):
        A_g = symm.transform_ao(A, iop, axis=-2, kaxis=0)
        A_g = symm.transform_ao(A_g, iop, axis=-1, kaxis=0, dagger=True)
        A_g = symm.transform_k(A_g, iop, axis=0)
        A_avg += A_g
    return A_avg / symm.nops


def classify_grid_by_symmetry(cell, mesh, coords=None, tol=1e-6):
    # Find all space-group operations that preserve the chosen mesh, then
    # classify grid points into orbits under those operations.
    #
    # Not every crystal symmetry preserves every finite mesh. For example, a
    # fractional translation or rotation may land between grid points. Such
    # operations are skipped here because they cannot be represented as a direct
    # permutation of the finite ISDF grid.
    mesh = np.asarray(mesh, dtype=np.int64)
    if coords is None:
        from pyscf.pbc.dft.gen_grid import get_uniform_grids
        coords = get_uniform_grids(cell, mesh)
    coords = np.asarray(coords)
    frac0 = np.mod(coords @ np.linalg.inv(cell.lattice_vectors()), 1.0)

    symm = Symmetry(cell).build(
        space_group_symmetry=True,
        symmorphic=False,
        check_mesh_symmetry=False,
    )

    op_ids = []
    perms = []
    for iop, op in enumerate(symm.ops):
        # Symmetry operation in fractional coordinates:
        # r -> R r + t. If all transformed points land on mesh nodes, store the
        # induced grid permutation.
        perm, _L_frac, resid = finite_grid_transform(cell, coords, op, mesh)
        if resid >= tol:
            continue
        op_ids.append(iop)
        perms.append(perm)

    ngrid = len(coords)
    orbit_id = -np.ones(ngrid, dtype=np.int64)
    orbits = []
    for i0 in range(ngrid):
        if orbit_id[i0] >= 0:
            continue
        # Flood-fill the orbit generated by all grid-preserving operations.
        orbit = {i0}
        frontier = [i0]
        while frontier:
            i = frontier.pop()
            for perm in perms:
                j = int(perm[i])
                if j not in orbit:
                    orbit.add(j)
                    frontier.append(j)
        orbit = np.array(sorted(orbit), dtype=np.int64)
        iorb = len(orbits)
        orbit_id[orbit] = iorb
        orbits.append(orbit)

    return {
        "op_ids": np.asarray(op_ids, dtype=np.int64),
        "orbit_id": orbit_id,
        "orbits": orbits,
    }
