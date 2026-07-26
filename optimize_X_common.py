import time

import h5py
import numpy as np
import torch


def get_device():
    import gpu_register
    return gpu_register.acquire_gpu()


def to_tensor(A, device, complex_dtype):
    return torch.from_numpy(A).to(device=device, dtype=complex_dtype)


def to_tensor128(A, device):
    return torch.from_numpy(A).to(device=device, dtype=torch.complex128)


def load_isdf(chkfile):
    with h5py.File(chkfile, "r") as f:
        X = np.asarray(f["inpv_kpt"])
        W = np.asarray(f["coul_kpt"])
    return X, W


def save_isdf(chkfile, X, W):
    with h5py.File(chkfile, "w") as f:
        f["inpv_kpt"] = X
        f["coul_kpt"] = W


def X_mo_from_ao(X_ao, C, C128=None, force_complex128=False):
    C_use = C128 if force_complex128 else C
    return X_ao @ C_use


def X_ao_from_mo(X_mo, C, C128=None, force_complex128=False):
    C_use = C128 if force_complex128 else C
    return torch.linalg.solve(
        C_use.transpose(-1, -2),
        X_mo.transpose(-1, -2),
    ).transpose(-1, -2)


def state_from_ao(X_ao, C, state_AO):
    if state_AO:
        return X_ao
    else:
        return X_mo_from_ao(X_ao, C)


def ao_from_state(X, C, C128, state_AO, force_complex128=False):
    if state_AO:
        return X
    else:
        return X_ao_from_mo(X, C, C128, force_complex128=force_complex128)


def state_unproject_error(X, X_ao, C, C128, state_AO):
    X_ao_check = ao_from_state(X, C, C128, state_AO)
    return torch.linalg.norm(X_ao_check - X_ao) / torch.linalg.norm(X_ao)


def ref_norm2_value(ref_norm2, ref_norm2_128, force_complex128=False):
    if force_complex128:
        return ref_norm2_128
    else:
        return ref_norm2


def objective_from_X_log_reg(
    X,
    log_reg,
    get_W_error2_from_X,
    get_abs_norm_loss,
    ref_norm2,
    ref_norm2_128,
    force_complex128=False,
):
    if force_complex128:
        with torch.no_grad():
            X = X.detach().to(dtype=torch.complex128)
            log_reg = log_reg.detach().to(dtype=torch.float64)

    reg = torch.exp(log_reg)
    ref_norm2_use = ref_norm2_value(ref_norm2, ref_norm2_128, force_complex128=force_complex128)
    W, error2 = get_W_error2_from_X(X, reg, ref_norm2_use, force_complex128=force_complex128)
    rel_error = torch.sqrt(error2 / ref_norm2_use + 1e-10)
    norm_loss = get_abs_norm_loss(X, W, force_complex128=force_complex128)
    return error2, rel_error, norm_loss, W, reg


def optimization_loss_from_X_log_reg(
    X,
    log_reg,
    get_W_error2_from_X,
    get_abs_norm_loss,
    ref_norm2,
    ref_norm2_128,
    norm_ratio,
    rel_error_scale,
    cosh_scale,
    cosh_sqrt=False,
    force_complex128=False,
):
    error2, rel_error, norm_loss, W, reg = objective_from_X_log_reg(
        X,
        log_reg,
        get_W_error2_from_X,
        get_abs_norm_loss,
        ref_norm2,
        ref_norm2_128,
        force_complex128=force_complex128,
    )
    punishment = torch.cosh(torch.sqrt(rel_error / cosh_scale)) if cosh_sqrt else torch.cosh(rel_error / cosh_scale)
    loss = torch.log10(rel_error / rel_error_scale + 1) + punishment - 1 + norm_loss * norm_ratio
    return loss, error2, rel_error, norm_loss, W, reg


def make_schedule(base_lr, nsteps_factor):
    nsteps_warmup = 100
    adam_schedule = [
        (nsteps_warmup, base_lr * 1e-4, base_lr),
    ]
    schedule = [
            (2500 * nsteps_factor, 1.0),
            (1500 * nsteps_factor, 0.5),
            (500 * nsteps_factor, 0.3),
            (400 * nsteps_factor, 0.2),
            (300 * nsteps_factor, 0.1),
            (200 * nsteps_factor, 0.05),
            (100 * nsteps_factor, 0.02),
            (50 * nsteps_factor, 0.01),
            (50 * nsteps_factor, 0.005),
            (50 * nsteps_factor, 0.002),
            (50 * nsteps_factor, 0.001),
            (50 * nsteps_factor, 0.0005),
            (50 * nsteps_factor, 0.0002),
            (50 * nsteps_factor, 0.0001),
            (50 * nsteps_factor, 0.00005),
            (50 * nsteps_factor, 0.00002),
            (50 * nsteps_factor, 0.00001),
    ]
    for nsteps, ratio in schedule:
        adam_schedule.append((nsteps, base_lr * ratio, base_lr * ratio))
    return adam_schedule


def timer_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def print_step(label, adam_lr, loss_value, best_loss, rel_error, norm_loss, reg, grad_x, grad_logreg, t_tot):
    print(
        "%-10s  X_lr %+.4e  loss %+11.5f  best_loss %+11.5f  rel_err %+.5e  norm %+.5e  reg %+.5e  |gX| %+.5e  |glreg| %+.5e  t_tot %.4f"
        % (
            label,
            adam_lr,
            loss_value,
            best_loss,
            rel_error.item(),
            norm_loss.item(),
            reg.item(),
            grad_x,
            grad_logreg,
            t_tot,
        ),
        flush=True,
    )


def lr_from_step(step, adam_schedule, lr_steps):
    idx = np.searchsorted(lr_steps, step, side="right")
    if idx >= len(adam_schedule):
        return adam_schedule[-1][2]

    nstep, lr_begin, lr_end = adam_schedule[idx]
    step0 = 0 if idx == 0 else lr_steps[idx - 1]
    x = (step - step0) / nstep
    return lr_begin * (lr_end / lr_begin) ** x


def run_adam(
    X_opt,
    log_reg,
    optimization_loss,
    optimization_loss_args,
    adam_schedule,
    base_lr,
    device,
    complex_dtype,
    run_message,
    chk_tag,
):
    adam_steps = sum(nstep for nstep, _, _ in adam_schedule)
    lr_steps = np.cumsum([nstep for nstep, _, _ in adam_schedule])
    lr_ratio = 2e-2 / base_lr
    nsteps_warmup = adam_schedule[0][0]
    print_every = 1
    complex128_every = 100
    chk_every = 0

    opt = torch.optim.Adam(
        [
            {"params": [X_opt], "lr": 1.0},
            {"params": [log_reg], "lr": 1.0},
        ],
        betas=(0.8, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=[
            lambda step: lr_from_step(step, adam_schedule, lr_steps),
            lambda step: lr_from_step(step, adam_schedule, lr_steps) * lr_ratio,
        ],
    )

    print(run_message)
    best_loss = np.inf
    best_loss_complex128 = np.inf
    for istep in range(adam_steps + 1):
        istep_main = istep - nsteps_warmup
        adam_lr = opt.param_groups[0]["lr"]
        timer_sync(device)
        time0 = time.perf_counter()

        opt.zero_grad()
        loss, error2, rel_error, norm_loss, W_opt, reg = optimization_loss(
            X_opt, log_reg, *optimization_loss_args,
        )
        log_reg.retain_grad()

        if istep == adam_steps:
            break

        loss.backward()
        loss_value = loss.item()
        best_loss = min(best_loss, loss_value)

        opt.step()
        scheduler.step()
        timer_sync(device)
        time1 = time.perf_counter()

        if istep % print_every == 0:
            print_step(
                "step %5d" % istep_main,
                adam_lr,
                loss_value,
                best_loss,
                rel_error,
                norm_loss,
                reg,
                torch.linalg.norm(X_opt.grad).item(),
                torch.linalg.norm(log_reg.grad).item(),
                time1 - time0,
            )

        if (complex_dtype is torch.complex64) and (istep % complex128_every == 0):
            timer_sync(device)
            time0 = time.perf_counter()
            loss128, error2_128, rel_error128, norm_loss128, W_128, reg128 = optimization_loss(
                X_opt, log_reg, *optimization_loss_args, force_complex128=True,
            )
            timer_sync(device)
            time1 = time.perf_counter()
            loss128_value = loss128.item()
            best_loss_complex128 = min(best_loss_complex128, loss128_value)
            print_step(
                "complex128",
                adam_lr,
                loss128_value,
                best_loss_complex128,
                rel_error128,
                norm_loss128,
                reg128,
                np.nan,
                np.nan,
                time1 - time0,
            )

        if (chk_every > 0) and (istep_main > 0) and (istep_main % chk_every == 0):
            data = {
                "X_opt": X_opt.detach().cpu(),
                "log_reg": log_reg.detach().cpu(),
                "opt_state": opt.state_dict(),
            }
            torch.save(data, f"data_GDF/chk/opt_{chk_tag}_it{istep_main}.pt")

    return optimization_loss(X_opt, log_reg, *optimization_loss_args, force_complex128=True), opt
