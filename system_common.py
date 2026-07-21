import os
import re
import ast
import numpy as np
from pyscf.pbc import gto
import pyscf.gto.basis as basis
import pyscf.gto.basis.parse_nwchem as parse_nwchem

ccgto_basis_map = {
    "gth-cc-pvdz": {
        "lc": os.path.expanduser("~/software/ccgto/basis/gth-hf-rev/cc-pvdz-lc.dat"),
        "sc": os.path.expanduser("~/software/ccgto/basis/gth-hf-rev/cc-pvdz-sc.dat"),
    },
    "gth-cc-pvtz": {
        "lc": os.path.expanduser("~/software/ccgto/basis/gth-hf-rev/cc-pvtz-lc.dat"),
        "sc": os.path.expanduser("~/software/ccgto/basis/gth-hf-rev/cc-pvtz-sc.dat"),
    },
    "gth-cc-pvqz": {
        "lc": os.path.expanduser("~/software/ccgto/basis/gth-hf-rev/cc-pvqz-lc.dat"),
        "sc": os.path.expanduser("~/software/ccgto/basis/gth-hf-rev/cc-pvqz-sc.dat"),
    },
}

_basis_variant_cache = {}


def get_data_dir(system, basis, suffix=None):
    if suffix is None:
        return os.path.abspath(f"data_{system}_{basis}")
    return os.path.abspath(f"data_{system}_{basis}_{suffix}")


def ensure_data_dir(system, basis, suffix=None):
    data_dir = get_data_dir(system, basis, suffix=suffix)
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def read_text(path):
    with open(path, "r") as f:
        return f.read().strip()


def read_text_if_exists(path):
    if os.path.exists(path):
        return read_text(path)
    return None


def load_settings(system, basis, suffix=None):
    path = os.path.join(get_data_dir(system, basis, suffix=suffix), "settings.pydata")
    with open(path, "r") as f:
        settings = ast.literal_eval(f.read())
    return settings


def load_xyz(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    natom = int(lines[0])
    atom = []
    for line in lines[2:2+natom]:
        fields = line.split()
        symbol = fields[0]
        coord = [float(x) for x in fields[1:4]]
        atom.append((symbol, coord))
    return atom


def load_lattice(path):
    return np.loadtxt(path)


def load_structure(system, basis, suffix=None):
    data_dir = get_data_dir(system, basis, suffix=suffix)
    xyz_path = os.path.join(data_dir, f"{system}_prim.xyz")
    lattice_path = os.path.join(data_dir, f"{system}_prim.lattice")
    atom = load_xyz(xyz_path)
    lattice = load_lattice(lattice_path)
    return atom, lattice


def load_pseudo(system, basis, suffix=None):
    settings = load_settings(system, basis, suffix=suffix)
    return settings["pseudo_potential"]


def load_xc(system, basis, suffix=None):
    settings = load_settings(system, basis, suffix=suffix)
    return settings["xc"]


def load_setting(system, basis, name, suffix=None):
    settings = load_settings(system, basis, suffix=suffix)
    return settings.get("scf", {}).get(name)


def load_cell_setting(system, basis, name, suffix=None):
    settings = load_settings(system, basis, suffix=suffix)
    return settings.get("cell", {}).get(name)


def load_section_setting(system, basis, section, name, suffix=None, default=None):
    settings = load_settings(system, basis, suffix=suffix)
    return settings.get(section, {}).get(name, default)


def _get_basis_variants(path):
    if path in _basis_variant_cache:
        return _basis_variant_cache[path]

    variants = {}
    with open(path, "r") as f:
        for line in f:
            if not line.startswith("#BASIS SET:"):
                continue
            label = line.split()[-1]
            match = re.fullmatch(r"([A-Za-z]+)(?:q(\d+))?", label)
            if match is None:
                continue
            symbol = match.group(1)
            q = None if match.group(2) is None else int(match.group(2))
            variants[symbol] = q
    _basis_variant_cache[path] = variants
    return variants


def _get_pseudo_q(pseudo, symbol):
    if isinstance(pseudo, str):
        pp = basis.load_pseudo(pseudo, symbol)
        return sum(pp[0])

    if isinstance(pseudo, dict):
        pp = pseudo[symbol]
        if isinstance(pp, str):
            pp = basis.load_pseudo(pp, symbol)
        return sum(pp[0])

    pp = pseudo
    if isinstance(pp, str):
        pp = basis.load_pseudo(pp, symbol)
    return sum(pp[0])


def load_basis(atom, basis_name, pseudo):
    if basis_name not in ccgto_basis_map:
        return basis_name

    basis_files = ccgto_basis_map[basis_name]
    basis_variants = {tag: _get_basis_variants(path) for tag, path in basis_files.items()}

    basis_dict = {}
    for symbol, _ in atom:
        if symbol in basis_dict:
            continue

        q = _get_pseudo_q(pseudo, symbol)
        candidates = []
        for tag in ["sc", "lc"]:
            path = basis_files[tag]
            variants = basis_variants[tag]
            if symbol in variants:
                candidates.append((path, variants[symbol]))

        if len(candidates) == 0:
            raise RuntimeError(f"No {basis_name} basis found for {symbol}")

        if len(candidates) == 1:
            basis_dict[symbol] = parse_nwchem.load(candidates[0][0], symbol)
            continue

        matched = [path for path, q_basis in candidates if q_basis == q]
        if len(matched) == 1:
            basis_dict[symbol] = parse_nwchem.load(matched[0], symbol)
            continue

        raise RuntimeError(f"No unique {basis_name} basis matches pseudo q={q} for {symbol}")
    return basis_dict


def make_cell(system, basis, verbose=0, suffix=None):
    atom, lattice = load_structure(system, basis, suffix=suffix)
    settings = load_settings(system, basis, suffix=suffix)
    pseudo = settings["pseudo_potential"]
    basis_input = load_basis(atom, basis, pseudo)
    cell_settings = dict(settings.get("cell", {}))
    is_2d = cell_settings.pop("is_2d", False)

    cell = gto.Cell()
    cell.unit = "A"
    cell.atom = atom
    cell.a = lattice
    cell.basis = basis_input
    cell.pseudo = pseudo
    if is_2d:
        cell.dimension = 2
    for key, value in cell_settings.items():
        if key == "ke_cutoff":
            continue
        setattr(cell, key, value)
    cell.verbose = verbose
    cell.build()
    return cell


def get_klabel(kmesh):
    return f"{kmesh[0]}x{kmesh[1]}x{kmesh[2]}"
