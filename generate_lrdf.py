import argparse
import os
import pickle
import numpy as np
import torch

import optimize_X_common
import system_common
import utils


def get_pair_matrix(Xo, Xv, q, kmesh):
    nkpts, nI, nocc = Xo.shape
    nvir = Xv.shape[-1]
    k = torch.arange(nkpts, device=Xo.device)
    kq = utils.add_k(k, q, kmesh).to(device=Xo.device)
    P = torch.einsum("kIi,kIa->kiaI", Xo, Xv[kq].conj()).reshape(nkpts * nocc * nvir, nI)
    return P.conj()


def apply_reference(P, W, Z):
    t = torch.einsum("pI,pr->Ir", P.conj(), Z)
    t = torch.einsum("IJ,Jr->Ir", W, t)
    return torch.einsum("pI,Ir->pr", P, t)


def get_svd_initial(Xo, Xv, W, kmesh, rank):
    nkpts, _, nocc = Xo.shape
    nvir = Xv.shape[-1]
    npair = nkpts * nocc * nvir
    H = torch.zeros((npair, npair), dtype=Xo.dtype, device=Xo.device)
    for q in range(nkpts):
        P = get_pair_matrix(Xo, Xv, q, kmesh)
        Wq = (W[q] + W[q].conj().transpose(-1, -2)) / 2.0
        V = torch.einsum("pI,IJ,qJ->pq", P, Wq, P.conj())
        H += torch.einsum("pr,qr->pq", V, V.conj())
    eigvals, U = torch.linalg.eigh(H)
    idx = torch.arange(npair - 1, npair - rank - 1, -1, device=Xo.device)
    singular_values = torch.sqrt(torch.clamp(eigvals[idx], min=0.0))
    U = U[:, idx]

    D = U.conj().reshape(nkpts, nocc, nvir, rank).permute(0, 3, 1, 2)
    G = torch.empty((nkpts, rank, rank), dtype=Xo.dtype, device=Xo.device)
    for q in range(nkpts):
        P = get_pair_matrix(Xo, Xv, q, kmesh)
        Wq = (W[q] + W[q].conj().transpose(-1, -2)) / 2.0
        VU = apply_reference(P, Wq, U)
        G[q] = torch.einsum("pr,ps->rs", U.conj(), VU)
    return D, G, singular_values


complex_dtype = torch.complex128
parser = argparse.ArgumentParser()
parser.add_argument("system")
parser.add_argument("kx", type=int)
parser.add_argument("ky", type=int)
parser.add_argument("kz", type=int)
parser.add_argument("basis")
parser.add_argument("c_ref", type=int)
parser.add_argument("-suffix", default=None)
parser.add_argument("-cuda", type=int, default=None)
args = parser.parse_args()

if args.cuda is None:
    device, gpu_reserve_tensor = optimize_X_common.get_device()
else:
    device = torch.device(f"cuda:{args.cuda}")

system = args.system
kmesh = (args.kx, args.ky, args.kz)
basis = args.basis
suffix = args.suffix
c_ref = args.c_ref
klabel = system_common.get_klabel(kmesh)
data_dir = system_common.get_data_dir(system, basis, suffix=suffix)
wannier_path = os.path.join(data_dir, f"Clocal_pywannier_{klabel}.npy")
ref_chk = os.path.join(data_dir, f"ISDFov_bareGDF_{klabel}_c{c_ref}.chk")
save_path = os.path.join(data_dir, f"LRDF_init_bareGDF_{klabel}_cref{c_ref}.pt")
dft_pkl = os.path.join(data_dir, f"DFT_{klabel}.pkl")
gdf_chk = os.path.join(data_dir, f"GDF_{klabel}.chk")

with open(dft_pkl, "rb") as f:
    mf = pickle.load(f)
mf.with_df._cderi = gdf_chk
naux = mf.with_df.get_naoaux()

wannier = np.load(wannier_path, allow_pickle=True).item()
X_ref_np, W_ref_np = optimize_X_common.load_isdf(ref_chk)

Cocc = torch.from_numpy(wannier["occ"]).to(device=device, dtype=complex_dtype)
Cvir = torch.from_numpy(wannier["vir"]).to(device=device, dtype=complex_dtype)
X_ref = torch.from_numpy(X_ref_np).to(device=device, dtype=complex_dtype)
W_ref = torch.from_numpy(W_ref_np).to(device=device, dtype=complex_dtype)
Xo_ref = torch.einsum("kIu,kui->kIi", X_ref, Cocc)
Xv_ref = torch.einsum("kIu,kua->kIa", X_ref, Cvir)

nkpts = int(np.prod(kmesh))
nocc = Cocc.shape[-1]
nvir = Cvir.shape[-1]
npair = nkpts * nocc * nvir
rank = min(naux, npair)
ref_norm2_q = utils.thc_ovvo_inner_from_mo(
    Xo_ref, Xv_ref, W_ref,
    Xo_ref, Xv_ref, W_ref,
    kmesh,
    by_q=True,
).real

print("device       =", device)
print("system       =", system)
print("kmesh        =", kmesh)
print("basis        =", basis)
print("suffix       =", suffix)
print("c_ref        =", c_ref)
print("naux         =", naux)
print("rank         =", rank)
print("nocc         =", nocc)
print("nvir         =", nvir)
print("reference    =", ref_chk)
print("pywannier    =", wannier_path)
print("")

D, G, singular_values = get_svd_initial(Xo_ref, Xv_ref, W_ref, kmesh, rank)

ref_norm2 = ref_norm2_q.sum()
lrdf_norm2 = utils.lrdf_ov_inner_from_mo(D, G, D, G, kmesh).real
cross = utils.thc_lrdf_ov_inner_from_mo(Xo_ref, Xv_ref, W_ref, D, G, kmesh).real
error2 = ref_norm2 + lrdf_norm2 - 2.0 * cross
rel_error = torch.sqrt(error2 / ref_norm2)

print("")
print("singular values =", singular_values.detach().cpu().numpy())
print("reference norm  = %.16e" % ref_norm2.item())
print("LRDF norm       = %.16e" % lrdf_norm2.item())
print("THC-LRDF inner  = %.16e" % cross.item())
print("LRDF rel_error  = %.16e" % rel_error.item())

data = {
    "D": D.detach().cpu(),
    "G": G.detach().cpu(),
    "kmesh": kmesh,
    "c_ref": c_ref,
    "naux": naux,
    "rank": rank,
    "singular_values": singular_values.detach().cpu(),
}
torch.save(data, save_path)
print("saved to", save_path)
