import argparse
import os
import sys
print(*sys.argv)
import pickle

import numpy as np
import torch

import libsymm
import optimize_X_common
import system_common
import utils


def fmt_label(x):
    return "%g" % x


def get_W_error2_from_X(X, reg, ref_norm2_use, force_complex128=False):
    X_ao = optimize_X_common.ao_from_state(X, C, C128, state_AO, force_complex128=force_complex128)
    if use_symm:
        symm_use = symm128 if force_complex128 else symm
        perm_use = perm128 if force_complex128 else perm
        phase_use = phase128 if force_complex128 else phase
        X_ao = libsymm.symmetrize_isdf_X(symm_use, X_ao, perm_use, phase_use)
    X_mo = optimize_X_common.X_mo_from_ao(X_ao, C, C128, force_complex128=force_complex128)
    Xo = X_mo[:, :, :nocc]
    Xv = X_mo[:, :, nocc:]

    Xo_ref_use = Xo_ref128 if force_complex128 else Xo_ref
    Xv_ref_use = Xv_ref128 if force_complex128 else Xv_ref
    W_ref_use = W_ref128 if force_complex128 else W_ref
    W, L, rhs = utils.thc_ovvo_solve_w_intermediate_from_mo(
        Xo_ref_use, Xv_ref_use, W_ref_use,
        Xo, Xv,
        kmesh,
        reg=reg,
    )
    if use_symm:
        W = libsymm.symmetrize_isdf_W(symm_use, W, perm_use, phase_use)
    error2 = utils.thc_solve_w_error2_from_intermediate(W, L, rhs, ref_norm2_use)
    return W, error2


@utils.maybe_profile
def get_abs_norm_loss(X, W, force_complex128=False):
    if (not use_symm) and (not state_AO):
        X_mo = X
    else:
        X_ao = optimize_X_common.ao_from_state(X, C, C128, state_AO, force_complex128=force_complex128)
        if use_symm:
            symm_use = symm128 if force_complex128 else symm
            perm_use = perm128 if force_complex128 else perm
            phase_use = phase128 if force_complex128 else phase
            X_ao = libsymm.symmetrize_isdf_X(symm_use, X_ao, perm_use, phase_use)
        X_mo = optimize_X_common.X_mo_from_ao(X_ao, C, C128, force_complex128=force_complex128)
    Xo = X_mo[:, :, :nocc]
    Xv = X_mo[:, :, nocc:]
    norm_Xo = torch.linalg.svdvals(Xo).max()
    norm_Xv = torch.linalg.svdvals(Xv).max()
    if use_Fnorm:
        norm_W = torch.linalg.norm(W, dim=(1, 2)).max() / nkpts
    else:
        norm_W = torch.linalg.eigvalsh(W).abs().max() / nkpts
    return norm_Xo**2 * norm_Xv**2 * norm_W


def print_header():
    print("cell.ke_cutoff =", cell.ke_cutoff)
    print("cell.mesh      =", cell.mesh)
    print("system         =", system)
    print("kmesh          =", kmesh)
    print("basis          =", basis)
    print("nao            =", nao)
    print("nocc           =", nocc)
    print("nvir           =", nvir)
    print("screen         =", screen)
    print("c_isdf         =", c_isdf)
    print("c_ref          =", c_ref)
    print("norm_const     =", norm_const)
    print("norm_power     =", norm_power)
    print("norm_ratio     =", norm_ratio)
    print("norm_label     =", norm_label)
    print("base_lr        =", base_lr)
    print("nsteps_factor  =", nsteps_factor)
    print("state_AO       =", state_AO)
    print("use_symm       =", use_symm)
    print("ISDF init cache =", init_chk)
    print("ISDF ref cache  =", ref_chk)
    if use_symm:
        print("symm grid mesh  =", cell_isdf.mesh)
        print("symm ix shape   =", ix_sel.shape)
        print("symm group shape =", None if group_sel is None else group_sel.shape)
    print("project-unproject relerr = %.16e" % project_unproject_error.item())
    print("torch device   =", device)
    print("torch dtype    =", complex_dtype)
    print("X_init max     =", torch.abs(X_init).max().item())
    print("Adam schedule:")
    for nstep, lr_begin, lr_end in adam_schedule:
        print("  nstep %6d  lr %.4e -> %.4e" % (nstep, lr_begin, lr_end))
    print("")


complex_dtype = torch.complex64
#complex_dtype = torch.complex128
real_dtype = torch.float32 if complex_dtype is torch.complex64 else torch.float64

parser = argparse.ArgumentParser()
parser.add_argument("system")
parser.add_argument("nk", type=int)
parser.add_argument("basis")
parser.add_argument("c_isdf", type=int)
parser.add_argument("c_ref", type=int)
parser.add_argument("norm_const", type=float)
parser.add_argument("base_lr", type=float)
parser.add_argument("nsteps_factor", type=int)
parser.add_argument("-power", type=int, default=0)
parser.add_argument("-suffix", default=None)
parser.add_argument("-device", type=int)
parser.add_argument("-init_reg", type=float, default=1e-2)
parser.add_argument("--use_Fnorm", action="store_true")
parser.add_argument("--screen", action="store_true")
parser.add_argument("--save", action="store_true")
parser.add_argument("--symm", action="store_true")
args = parser.parse_args()

if args.device is None:
    device, gpu_reserve_tensor = optimize_X_common.get_device()
else:
    device = torch.device(f"cuda:{args.device}")

system = args.system
nk = args.nk
kmesh = (nk, nk, nk)
klabel = system_common.get_klabel(kmesh)
basis = args.basis
suffix = args.suffix
c_isdf = args.c_isdf
c_ref = args.c_ref
norm_const = args.norm_const
norm_power = args.power
norm_ratio = norm_const * nk**norm_power
norm_label = "norm%sp%s" % (fmt_label(norm_const), fmt_label(norm_power))
base_lr = args.base_lr
nsteps_factor = args.nsteps_factor
screen = args.screen
screen_tag = "screen" if screen else "bare"
state_AO = False
use_symm = args.symm
symm_tag = "_symm" if use_symm else ""
use_Fnorm = args.use_Fnorm
init_reg = args.init_reg

data_dir = system_common.get_data_dir(system, basis, suffix=suffix)
dft_pkl = os.path.join(data_dir, f"DFT_{klabel}{symm_tag}.pkl")
init_chk = os.path.join(data_dir, f"ISDFov_{screen_tag}GDF{symm_tag}_{klabel}_c{c_isdf}.chk")
ref_chk = os.path.join(data_dir, f"ISDFov_{screen_tag}GDF{symm_tag}_{klabel}_c{c_ref}.chk")
opt_save_path = os.path.join(data_dir, f"opt_X_ov_{screen_tag}GDF{symm_tag}_{klabel}_c{c_isdf}_cref{c_ref}_{norm_label}.pt")
chk_save_path = os.path.join(data_dir, f"ISDFov_opt_{screen_tag}GDF{symm_tag}_{klabel}_c{c_isdf}_cref{c_ref}_{norm_label}.chk")

if args.save:
    print("Optimization result will be saved to", opt_save_path)
    print("ISDF chk            will be saved to", chk_save_path)

with open(dft_pkl, "rb") as f:
    mf = pickle.load(f)

cell = mf.cell
kpts = cell.make_kpts(kmesh)
kpts_int = np.round(cell.get_scaled_kpts(kpts) * kmesh).astype(int) % kmesh
assert utils.is_k_ordered(kpts_int, kmesh)

C_np = np.asarray(mf.mo_coeff)
C = optimize_X_common.to_tensor(C_np, device, complex_dtype)
C128 = optimize_X_common.to_tensor128(C_np, device)
nkpts, nao, nmo = C.shape
nocc = cell.nelectron // 2
Cocc = C[:, :, :nocc]
Cvir = C[:, :, nocc:]
Cocc128 = C128[:, :, :nocc]
Cvir128 = C128[:, :, nocc:]
nvir = Cvir.shape[2]

X_ref_np, W_ref_np = optimize_X_common.load_isdf(ref_chk)
X_init_np, _ = optimize_X_common.load_isdf(init_chk)
X_ref_ao = optimize_X_common.to_tensor(X_ref_np, device, complex_dtype)
X_ref_ao128 = optimize_X_common.to_tensor128(X_ref_np, device)
W_ref = optimize_X_common.to_tensor(W_ref_np, device, complex_dtype)
W_ref128 = optimize_X_common.to_tensor128(W_ref_np, device)
X_init_ao = optimize_X_common.to_tensor(X_init_np, device, complex_dtype)

if use_symm:
    cell_isdf = cell.copy()
    mesh, ix_sel, group_sel = optimize_X_common.load_isdf_grid(init_chk)
    cell_isdf.mesh = mesh
    coords = cell_isdf.gen_uniform_grids(cell_isdf.mesh)
    symm = libsymm.PBCSymmetry(cell_isdf, kmesh, kpts, dtype=complex_dtype, device=device)
    perm, phase = libsymm.build_isdf_grid_transform(symm, coords, ix_sel, mesh=cell_isdf.mesh)
    symm128 = libsymm.PBCSymmetry(cell_isdf, kmesh, kpts, dtype=torch.complex128, device=device)
    perm128, phase128 = libsymm.build_isdf_grid_transform(symm128, coords, ix_sel, mesh=cell_isdf.mesh)
else:
    cell_isdf = None
    mesh = ix_sel = group_sel = None
    symm = perm = phase = None
    symm128 = perm128 = phase128 = None

Xo_ref = X_ref_ao @ Cocc
Xv_ref = X_ref_ao @ Cvir
Xo_ref128 = X_ref_ao128 @ Cocc128
Xv_ref128 = X_ref_ao128 @ Cvir128
ref_norm2 = utils.thc_ovvo_inner_from_mo(Xo_ref, Xv_ref, W_ref, Xo_ref, Xv_ref, W_ref, kmesh).real
ref_norm2_128 = utils.thc_ovvo_inner_from_mo(Xo_ref128, Xv_ref128, W_ref128, Xo_ref128, Xv_ref128, W_ref128, kmesh).real
print('Xo_ref norm', torch.linalg.norm(Xo_ref).item())
print('Xv_ref norm', torch.linalg.norm(Xv_ref).item())
print(' W_ref norm', torch.linalg.norm(W_ref).item())

X_init = optimize_X_common.state_from_ao(X_init_ao, C, state_AO)
project_unproject_error = optimize_X_common.state_unproject_error(X_init, X_init_ao, C, C128, state_AO)
X_opt = X_init.detach().clone().requires_grad_(True)
log_reg = torch.tensor(np.log(init_reg), dtype=real_dtype, device=device, requires_grad=True)
adam_schedule = optimize_X_common.make_schedule(base_lr, nsteps_factor)

print_header()
optimization_loss_args = (
    get_W_error2_from_X,
    get_abs_norm_loss,
    ref_norm2,
    ref_norm2_128,
    norm_ratio,
    1e-2,
    1e-1,
    False
)
(loss_opt, error2_opt, rel_error_opt, norm_loss_opt, W_opt, reg_opt), opt = optimize_X_common.run_adam(
    X_opt,
    log_reg,
    optimize_X_common.optimization_loss_from_X_log_reg,
    optimization_loss_args,
    adam_schedule,
    base_lr,
    device,
    complex_dtype,
    "Running Adam optimization over X and log_reg ...",
    "ov",
)

X_ao_opt = optimize_X_common.ao_from_state(X_opt.detach(), C, C128, state_AO)
W_save = W_opt.detach()
if use_symm:
    X_symm = libsymm.symmetrize_isdf_X(symm, X_ao_opt, perm, phase)
    W_symm = libsymm.symmetrize_isdf_W(symm, W_save, perm, phase)
    X_symm_err = torch.linalg.norm(X_symm - X_ao_opt) / torch.linalg.norm(X_ao_opt)
    W_symm_err = torch.linalg.norm(W_symm - W_save) / torch.linalg.norm(W_save)

print("final loss      = %.16e" % loss_opt.item())
print("final error2    = %.16e" % error2_opt.item())
print("final rel_error = %.16e" % rel_error_opt.item())
print("final norm_loss = %.16e" % norm_loss_opt.item())
print("final reg       = %.16e" % reg_opt.item())
if use_symm:
    print("final X symm err = %.16e" % X_symm_err.item())
    print("final W symm err = %.16e" % W_symm_err.item())

if args.save:
    X_ao = X_ao_opt.cpu().numpy()
    data = {
        "X_opt": X_opt.detach().cpu(),
        "log_reg": log_reg.detach().cpu(),
        "opt_state": opt.state_dict(),
    }
    torch.save(data, opt_save_path)
    optimize_X_common.save_isdf(chk_save_path, X_ao, W_save.cpu().numpy(), mesh=mesh, ix_sel=ix_sel, group_sel=group_sel)
    print("Saved optimization result to", opt_save_path)
    print("Saved ISDF chk to", chk_save_path)
