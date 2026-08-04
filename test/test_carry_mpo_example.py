import numpy as np

import lib_qtt


rng = np.random.default_rng(781)

nsite = 4
dim = 2
N = dim**nsite
rank = 3

A = []
A.append(rng.normal(size=(dim, dim, rank)))
for isite in range(1, nsite - 1):
    A.append(rng.normal(size=(rank, dim, dim, rank)))
A.append(rng.normal(size=(rank, dim, dim)))

# i digits: a b c d
# j digits: i j k l
# q digits: p q r s
# alpha bonds: A B C
# borrow bonds: X Y Z
L_ij = np.einsum("aiA,AbjB,BckC,Cdl->abcdijkl", A[0], A[1], A[2], A[3], optimize=True)
L_ij_digits = L_ij

L_ij = np.empty((N, N))
for i in range(N):
    for j in range(N):
        idig = tuple((i >> isite) & 1 for isite in range(nsite))
        jdig = tuple((j >> isite) & 1 for isite in range(nsite))
        L_ij[i, j] = L_ij_digits[idig + jdig]

B = []
B.append(np.zeros((dim, dim, dim, 2, rank)))
for isite in range(1, nsite - 1):
    B.append(np.zeros((2, rank, dim, dim, dim, 2, rank)))
B.append(np.zeros((2, rank, dim, dim, dim)))

for i in range(dim):
    for j in range(dim):
        b0 = 0
        t = j - i - b0
        q = t % dim
        b1 = 1 if t < 0 else 0
        B[0][i, j, q, b1, :] = A[0][i, j, :]

for isite in range(1, nsite - 1):
    for b0 in range(2):
        for i in range(dim):
            for j in range(dim):
                t = j - i - b0
                q = t % dim
                b1 = 1 if t < 0 else 0
                B[isite][b0, :, i, j, q, b1, :] = A[isite][:, i, j, :]

for b0 in range(2):
    for i in range(dim):
        for j in range(dim):
            t = j - i - b0
            q = t % dim
            B[-1][b0, :, i, j, q] = A[-1][:, i, j]


def true_norm_from_B(T, isite):
    if isite == 0:
        norms = []
        for q in range(2):
            for bout in range(2):
                M = T[:, :, q, bout, :].reshape(dim * dim, rank)
                norms.append(np.linalg.svd(M, compute_uv=False)[0])
        return max(norms), norms
    if isite == nsite - 1:
        norms = []
        for q in range(2):
            M = T[:, :, :, :, q].reshape(2 * rank * dim * dim, 1)
            norms.append(np.linalg.svd(M, compute_uv=False)[0])
        return max(norms), norms

    norms = []
    for q in range(2):
        for bout in range(2):
            M = T[:, :, :, :, q, bout, :].reshape(2 * T.shape[1] * dim * dim, T.shape[-1])
            norms.append(np.linalg.svd(M, compute_uv=False)[0])
    return max(norms), norms

L_ijq_B = np.einsum("aipXA,XAbjqYB->abijpqYB", B[0], B[1], optimize=True)
L_ijq_B = np.einsum("abijpqYB,YBckrZC->abcijkpqrZC", L_ijq_B, B[2], optimize=True)
L_ijq_B = np.einsum("abcijkpqrZC,ZCdls->abcdijklpqrs", L_ijq_B, B[3], optimize=True)
L_ijq_B_digits = np.einsum("abcdijklpqrs->abcdijklpqrs", L_ijq_B, optimize=True)

L_ijq_B = np.empty((N, N, N))
L_ijq_direct = np.zeros((N, N, N))
for i in range(N):
    for j in range(N):
        q_good = (j - i) % N
        L_ijq_direct[i, j, q_good] = L_ij[i, j]
        idig = tuple((i >> isite) & 1 for isite in range(nsite))
        jdig = tuple((j >> isite) & 1 for isite in range(nsite))
        for q in range(N):
            qdig = tuple((q >> isite) & 1 for isite in range(nsite))
            L_ijq_B[i, j, q] = L_ijq_B_digits[idig + jdig + qdig]

print("nsite =", nsite)
print("dim =", dim)
print("N =", N)
print("rank =", rank)
print("A shapes =", [x.shape for x in A])
print("B shapes =", [x.shape for x in B])
print("L_ijq shape =", L_ijq_B.shape)

print("")
print("Block norm check")
norms_A, block_norms_A = lib_qtt.mpo_borrow_block_norm(A, return_blocks=True)
norm_errors = []
for isite in range(nsite):
    nA = norms_A[isite]
    blocks_A = block_norms_A[isite]
    nB, blocks_B = true_norm_from_B(B[isite], isite)
    print("site", isite)
    print("  blocks from A =", blocks_A)
    print("  blocks from B =", blocks_B)
    print("  max from A    =", nA)
    print("  max from B    =", nB)
    print("  diff          =", abs(nA - nB))
    norm_errors.append(abs(nA - nB))

diff = L_ijq_B - L_ijq_direct
print("max |B - direct| =", np.abs(diff).max())
print("||B - direct|| / ||direct|| =", np.linalg.norm(diff) / np.linalg.norm(L_ijq_direct))

for i in range(N):
    for j in range(N):
        q_good = (j - i) % N
        for q in range(N):
            n = abs(L_ijq_B[i, j, q])
            if q == q_good:
                assert abs(n - abs(L_ij[i, j])) < 1e-10
            else:
                assert n < 1e-12

print("All wrong-q blocks are zero.")


def test_carry_mpo_block_norms():
    assert max(norm_errors) < 1e-12


def test_carry_mpo_matches_dense_tensor():
    assert np.abs(diff).max() < 1e-12
