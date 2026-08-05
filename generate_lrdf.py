import argparse
import os
import pickle

import numpy as np
import torch

import optimize_X_common
import system_common
import utils


def get_svd_initial(R, rank):
    nkpts, _, naux, nocc, nvir = R.shape
    npair = nkpts * nocc * nvir

    Rq = torch.einsum("kqxia->qkiax", R).reshape(nkpts, npair, naux)
    M = torch.einsum("qpx,qpy->qxy", Rq, Rq.conj())
    eigvals, eigvecs = torch.linalg.eigh(M)
    M_sqrt = torch.einsum(
        "qxi,qi,qyi->qxy",
        eigvecs,
        torch.sqrt(torch.clamp(eigvals, min=0.0)),
        eigvecs.conj(),
    )
    B = torch.einsum("qpx,qxy->qpy", Rq.conj(), M_sqrt)
    B = torch.einsum("qpy->pqy", B).reshape(npair, nkpts * naux)

    U, singular_values, _ = torch.linalg.svd(B, full_matrices=False)
    U = U[:, :rank]
    singular_values = singular_values[:rank]

    D = U.conj().reshape(nkpts, nocc, nvir, rank).permute(0, 3, 1, 2)
    F = torch.einsum("qpx,pr->qxr", Rq, U)
    G = torch.einsum("qxr,qxs->qrs", F.conj(), F)
    return D, G, singular_values


complex_dtype = torch.complex128
parser = argparse.ArgumentParser()
parser.add_argument("system")
parser.add_argument("kx", type=int)
parser.add_argument("ky", type=int)
parser.add_argument("kz", type=int)
parser.add_argument("basis")
parser.add_argument("-suffix", default=None)
parser.add_argument("-cuda", type=int, default=None)
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()

if args.cpu:
    device = torch.device("cpu")
elif args.cuda is None:
    device, gpu_reserve_tensor = optimize_X_common.get_device()
else:
    device = torch.device(f"cuda:{args.cuda}")

system = args.system
kmesh = (args.kx, args.ky, args.kz)
basis = args.basis
suffix = args.suffix
klabel = system_common.get_klabel(kmesh)
data_dir = system_common.get_data_dir(system, basis, suffix=suffix)
wannier_path = os.path.join(data_dir, f"Clocal_pywannier_{klabel}.npy")
save_path = os.path.join(data_dir, f"LRDF_init_bareGDF_{klabel}.pt")
dft_pkl = os.path.join(data_dir, f"DFT_{klabel}.pkl")
gdf_chk = os.path.join(data_dir, f"GDF_{klabel}.chk")

with open(dft_pkl, "rb") as f:
    mf = pickle.load(f)
mf.with_df._cderi = gdf_chk
naux = mf.with_df.get_naoaux()
kpts = mf.cell.make_kpts(kmesh)
kpts_int = np.round(mf.cell.get_scaled_kpts(kpts) * kmesh).astype(int) % kmesh
assert utils.is_k_ordered(kpts_int, kmesh)

wannier = np.load(wannier_path, allow_pickle=True).item()
assert wannier["layout"] == "k2q"
Cocc = np.asarray(wannier["occ"])
Cvir = np.asarray(mf.mo_coeff)[:, :, Cocc.shape[-1]:]
R = utils.get_gdf_tensor_compact(
    mf.with_df, kpts_int, kmesh, Cocc, Cvir, layout="k2q",
)
R = torch.from_numpy(R).to(device=device, dtype=complex_dtype)

nkpts = int(np.prod(kmesh))
nocc = Cocc.shape[-1]
nvir = Cvir.shape[-1]
npair = nkpts * nocc * nvir
rank = min(naux, npair)

print("device       =", device)
print("system       =", system)
print("kmesh        =", kmesh)
print("basis        =", basis)
print("suffix       =", suffix)
print("naux         =", naux)
print("rank         =", rank)
print("nocc         =", nocc)
print("nvir         =", nvir)
print("reference    =", gdf_chk)
print("pywannier    =", wannier_path)
print("")

D, G, singular_values = get_svd_initial(R, rank)

ref_norm2 = utils.df_inner_from_mo(R, kmesh).real
lrdf_norm2 = utils.lrdf_ov_inner_from_mo(D, G, D, G, kmesh).real
cross = utils.lrdf_df_ov_inner_from_mo(D, G, R, kmesh).real
error2 = torch.clamp(ref_norm2 + lrdf_norm2 - 2.0 * cross, min=0.0)
rel_error = torch.sqrt(error2 / ref_norm2)

print("")
print("singular values =", singular_values.detach().cpu().numpy())
print("GDF norm        = %.16e" % ref_norm2.item())
print("LRDF norm       = %.16e" % lrdf_norm2.item())
print("LRDF-GDF inner  = %.16e" % cross.item())
print("LRDF rel_error  = %.16e" % rel_error.item())

data = {
    "D": D.detach().cpu(),
    "G": G.detach().cpu(),
    "kmesh": kmesh,
    "naux": naux,
    "rank": rank,
    "reference": "GDF",
    "orbital_basis": "pywannier_occ_canonical_vir",
    "layout": "k2q",
    "singular_values": singular_values.detach().cpu(),
}
torch.save(data, save_path)
print("saved to", save_path)
