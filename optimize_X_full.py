import argparse
import os
import sys
print(*sys.argv)
import pickle

import numpy as np
import torch

import optimize_X_common
import system_common
import utils


def fmt_label(x):
    return "%g" % x


def get_W_error2_from_X(X, reg, ref_norm2_use, force_complex128=False):
    X_ref_use = X_ref128 if force_complex128 else X_ref
    W_ref_use = W_ref128 if force_complex128 else W_ref

    X_ao = optimize_X_common.ao_from_state(X, C, C128, state_AO, force_complex128=force_complex128)
    if fit_AO:
        X_loss = X_ao
    else:
        X_loss = optimize_X_common.X_mo_from_ao(X_ao, C, C128, force_complex128=force_complex128)

    W, error2 = utils.thc_solve_w_error2_from_mo(
        X_ref_use, W_ref_use,
        X_loss,
        kmesh,
        ref_norm2_use,
        reg=reg,
    )
    return W, error2


@utils.maybe_profile
def get_abs_norm_loss(X, W, force_complex128=False):
    X_mo = X if not state_AO else optimize_X_common.X_mo_from_ao(X, C, C128, force_complex128=force_complex128)
    norm_X = torch.linalg.svdvals(X_mo).max()
    W_R = utils.fourier_transform_3d(W, axis=0, kmesh=kmesh, inverse=False) / np.sqrt(nkpts).item()
    norm_W = torch.abs(W_R).max()
    return norm_X**4 * norm_W


def print_header():
    print("cell.ke_cutoff =", cell.ke_cutoff)
    print("cell.mesh      =", cell.mesh)
    print("system         =", system)
    print("kmesh          =", kmesh)
    print("basis          =", basis)
    print("nao            =", nao)
    print("nmo            =", nmo)
    print("fit_AO         =", fit_AO)
    print("state_AO       =", state_AO)
    print("screen         =", screen)
    print("c_isdf         =", c_isdf)
    print("c_ref          =", c_ref)
    print("norm_const     =", norm_const)
    print("norm_ratio     =", norm_ratio)
    print("norm_label     =", norm_label)
    print("base_lr        =", base_lr)
    print("nsteps_factor  =", nsteps_factor)
    print("ISDF init cache =", init_chk)
    print("ISDF ref cache  =", ref_chk)
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
fit_AO = True
state_AO = False
init_reg = 1e-2

parser = argparse.ArgumentParser()
parser.add_argument("system")
parser.add_argument("nk", type=int)
parser.add_argument("basis")
parser.add_argument("c_isdf", type=int)
parser.add_argument("c_ref", type=int)
parser.add_argument("norm_const", type=float)
parser.add_argument("base_lr", type=float)
parser.add_argument("nsteps_factor", type=int)
parser.add_argument("-suffix", default=None)
parser.add_argument("-device", type=int)
parser.add_argument("--screen", action="store_true")
parser.add_argument("--save", action="store_true")
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
norm_ratio = norm_const
norm_label = "norm%s" % fmt_label(norm_const)
base_lr = args.base_lr
nsteps_factor = args.nsteps_factor
screen = args.screen
screen_tag = "screen" if screen else "bare"

data_dir = system_common.get_data_dir(system, basis, suffix=suffix)
dft_pkl = os.path.join(data_dir, f"DFT_{klabel}.pkl")
init_chk = os.path.join(data_dir, f"ISDFfull_{screen_tag}GDF_{klabel}_c{c_isdf}.chk")
ref_chk = os.path.join(data_dir, f"ISDFfull_{screen_tag}GDF_{klabel}_c{c_ref}.chk")
opt_save_path = os.path.join(data_dir, f"opt_X_full_{screen_tag}GDF_{klabel}_c{c_isdf}_cref{c_ref}_{norm_label}.pt")
chk_save_path = os.path.join(data_dir, f"ISDFfull_opt_{screen_tag}GDF_{klabel}_c{c_isdf}_cref{c_ref}_{norm_label}.chk")

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

X_ref_np, W_ref_np = optimize_X_common.load_isdf(ref_chk)
X_init_np, _ = optimize_X_common.load_isdf(init_chk)
X_ref_ao = optimize_X_common.to_tensor(X_ref_np, device, complex_dtype)
X_ref_ao128 = optimize_X_common.to_tensor128(X_ref_np, device)
W_ref = optimize_X_common.to_tensor(W_ref_np, device, complex_dtype)
W_ref128 = optimize_X_common.to_tensor128(W_ref_np, device)
X_init_ao = optimize_X_common.to_tensor(X_init_np, device, complex_dtype)

if fit_AO:
    X_ref = X_ref_ao
    X_ref128 = X_ref_ao128
else:
    X_ref = optimize_X_common.X_mo_from_ao(X_ref_ao, C)
    X_ref128 = optimize_X_common.X_mo_from_ao(X_ref_ao128, C, C128, force_complex128=True)
ref_norm2 = utils.thc_inner_from_mo(X_ref, W_ref, X_ref, W_ref, kmesh).real
ref_norm2_128 = utils.thc_inner_from_mo(X_ref128, W_ref128, X_ref128, W_ref128, kmesh).real

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
    1e-3,
    1e-2,
    True,
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
    "full",
)

print("final loss      = %.16e" % loss_opt.item())
print("final error2    = %.16e" % error2_opt.item())
print("final rel_error = %.16e" % rel_error_opt.item())
print("final norm_loss = %.16e" % norm_loss_opt.item())
print("final reg       = %.16e" % reg_opt.item())

if args.save:
    X_ao = optimize_X_common.ao_from_state(X_opt.detach(), C, C128, state_AO).cpu().numpy()
    data = {
        "X_opt": X_opt.detach().cpu(),
        "log_reg": log_reg.detach().cpu(),
        "opt_state": opt.state_dict(),
    }
    torch.save(data, opt_save_path)
    optimize_X_common.save_isdf(chk_save_path, X_ao, W_opt.detach().cpu().numpy())
    print("Saved optimization result to", opt_save_path)
    print("Saved ISDF chk to", chk_save_path)
