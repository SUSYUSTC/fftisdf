import argparse
import os
import subprocess
import sys


def read_cref(path, system):
    with open(path) as f:
        for line in f:
            cols = line.split()
            if len(cols) >= 2 and cols[0] == system:
                return cols[1]
    raise RuntimeError(f"cannot find {system} in {path}")


parser = argparse.ArgumentParser()
parser.add_argument("device")
parser.add_argument("system")
parser.add_argument("basis")
parser.add_argument("c", type=int)
parser.add_argument("norm", type=float)
parser.add_argument("nk", nargs="+", type=int)
parser.add_argument("-base_lr", default="1e-2")
parser.add_argument("-nsteps", type=int, default=4)
parser.add_argument("--screen", action="store_true")
parser.add_argument("-suffix", default="final")
parser.add_argument("--symm", action="store_true")
parser.add_argument("--real", action="store_true")
parser.add_argument("--save", action="store_true", default=True)
parser.add_argument("--run", action="store_true")
args = parser.parse_args()

os.makedirs("opt_full", exist_ok=True)
cref_path = "cref_system_full_symm" if args.symm else "cref_system_full"
cref = read_cref(cref_path, args.system)

prefix = "opt"
if args.symm:
    prefix += "symm"
if args.real:
    prefix += "real"
prefix += "_fullscreen" if args.screen else "_fullbare"

for nk in args.nk:
    cmd = [
        sys.executable,
        "optimize_X_full.py",
        args.system,
        str(nk),
        args.basis,
        str(args.c),
        cref,
        str(args.norm),
        args.base_lr,
        str(args.nsteps),
        "-suffix",
        args.suffix,
        "-device",
        args.device,
    ]
    if args.screen:
        cmd.append("--screen")
    if args.symm:
        cmd.append("--symm")
    if args.real:
        cmd.append("--real")
    if args.save:
        cmd.append("--save")

    log = "opt_full/%s_%s_nk%d_c%d_norm%s_%d" % (
        prefix,
        args.system,
        nk,
        args.c,
        args.norm,
        args.nsteps,
    )
    print(" ".join(cmd), flush=True)
    print("log", log, flush=True)
    if args.run:
        with open(log, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
