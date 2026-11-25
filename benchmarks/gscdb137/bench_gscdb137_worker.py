#!/usr/bin/env python3
"""
Sharded Multi-Process GSCDB137 Accuracy & Wall-Clock Timing Worker
==================================================================
Evaluates PGFT v2 (C++20), PBE (GGA), and CF22D (Hybrid meta-GGA + D3(0))
on identical geometries and basis sets from GSCDB137 (`Allmols_info.json` /
`DatasetEval.csv`), using ProcessPoolExecutor with 1 thread/worker for 100%
core utilization and exact single-core per-molecule wall-clock timings.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Enforce single-threaded BLAS/OpenMP per worker BEFORE importing numpy/pyscf
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

import numpy as np
from pyscf import dft, gto, scf
import dftd3.interface as d3i

PGFT_SRC = Path("/content/PGFT/src")
if PGFT_SRC.exists():
    if str(PGFT_SRC) not in sys.path:
        sys.path.insert(0, str(PGFT_SRC))
else:
    p = Path(__file__).resolve()
    if len(p.parents) >= 3:
        local_src = p.parents[2] / "src"
        if str(local_src) not in sys.path:
            sys.path.insert(0, str(local_src))

from pgft import run_pgft  # noqa: E402


def calc_d3_cf22d(mol: gto.Mole) -> Tuple[float, float]:
    """Exact CF22D D3(0) 2-body dispersion (s6=1.0, s8=0.0, rs6=1.53, rs8=1.0, alp=14.0, s9=0.0)."""
    t0 = time.perf_counter()
    if mol.natm <= 1:
        return 0.0, time.perf_counter() - t0
    numbers = np.array([mol.atom_charge(i) for i in range(mol.natm)], dtype=np.int32)
    positions = np.array(mol.atom_coords(unit="Bohr"), dtype=np.float64)
    disp = d3i.DispersionModel(numbers, positions)
    param = d3i.ZeroDampingParam(s6=1.0, s8=0.0, rs6=1.53, rs8=1.0, alp=14.0, s9=0.0)
    e_d3 = float(disp.get_dispersion(param, grad=False)["energy"])
    return e_d3, time.perf_counter() - t0


def build_mol_from_gscdb(info: Dict[str, Any], basis: str) -> gto.Mole:
    lines = [ln.strip() for ln in info["molecule"].strip().splitlines() if ln.strip()]
    atom_str = "; ".join(lines)
    chg = int(info.get("charge", 0))
    mult = int(info.get("multiplicity", 1))
    spin = mult - 1
    mol = gto.M(
        atom=atom_str,
        charge=chg,
        spin=spin,
        basis=basis,
        unit="Angstrom",
        symmetry=False,
        verbose=0,
    )
    return mol


def robust_scf(mf: scf.hf.SCF) -> scf.hf.SCF:
    """Converge HF or KS-DFT reliably with SOSCF and level-shift fallbacks."""
    mf.conv_tol = 1e-10
    mf.max_cycle = 150
    mf.kernel()
    if mf.converged:
        return mf
    dm = mf.make_rdm1()
    mf2 = mf.newton()
    mf2.conv_tol = 1e-10
    mf2.max_cycle = 80
    mf2.kernel(dm)
    if mf2.converged:
        return mf2
    # Level-shifted DIIS followed by Newton
    mf.level_shift = 0.35
    mf.damp = 0.5
    mf.max_cycle = 200
    mf.kernel(dm)
    if mf.converged:
        mf.level_shift = 0.0
        mf.damp = 0.0
        mf.kernel(mf.make_rdm1())
        if mf.converged:
            return mf
    dm2 = mf.make_rdm1()
    mf3 = mf.newton()
    mf3.conv_tol = 1e-9
    mf3.max_cycle = 100
    mf3.kernel(dm2)
    return mf3


def eval_one_molecule(task: Tuple[str, Dict[str, Any], str, List[str], float, int, str, bool]) -> Dict[str, Any]:
    mkey, info, basis, methods, c_thc, n_rad, grid_spec, use_df = task
    out: Dict[str, Any] = {
        "mol_key": mkey,
        "basis": basis,
        "charge": int(info.get("charge", 0)),
        "multiplicity": int(info.get("multiplicity", 1)),
        "status": "ok",
    }
    try:
        mol = build_mol_from_gscdb(info, basis)
        out["natm"] = int(mol.natm)
        out["nelec"] = int(mol.nelectron)
        out["nao"] = int(mol.nao_nr())

        if "PBE" in methods:
            t0 = time.perf_counter()
            mf_p = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
            if use_df:
                mf_p = mf_p.density_fit()
            mf_p.xc = "PBE"
            if grid_spec == "99_590":
                mf_p.grids.atom_grid = (99, 590)
            else:
                mf_p.grids.level = 3
            mf_p = robust_scf(mf_p)
            t_pbe = time.perf_counter() - t0
            out["PBE"] = {
                "e_tot": float(mf_p.e_tot),
                "converged": bool(mf_p.converged),
                "t_tot_s": float(t_pbe),
            }

        if "CF22D" in methods:
            t0 = time.perf_counter()
            mf_c = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
            if use_df:
                mf_c = mf_c.density_fit()
            mf_c.xc = "HYB_MGGA_X_CF22D, MGGA_C_CF22D"
            if grid_spec == "99_590":
                mf_c.grids.atom_grid = (99, 590)
            else:
                mf_c.grids.level = 3
            mf_c = robust_scf(mf_c)
            t_scf = time.perf_counter() - t0
            e_d3, t_d3 = calc_d3_cf22d(mol)
            out["CF22D"] = {
                "e_tot": float(mf_c.e_tot) + e_d3,
                "e_scf": float(mf_c.e_tot),
                "e_d3": e_d3,
                "converged": bool(mf_c.converged),
                "t_scf_s": float(t_scf),
                "t_d3_s": float(t_d3),
                "t_tot_s": float(t_scf + t_d3),
            }

        if "PGFT" in methods:
            t0 = time.perf_counter()
            mf_hf = scf.RHF(mol) if mol.spin == 0 else scf.ROHF(mol)
            mf_hf = robust_scf(mf_hf)
            t_hf = time.perf_counter() - t0

            res = run_pgft(mol, mf_init=mf_hf, backend="cpp", c_thc=c_thc, n_rad=n_rad)
            t_tot = time.perf_counter() - t0
            out["PGFT"] = {
                "e_tot": float(res.e_tot),
                "e_stat": float(res.e_stat),
                "e_dyn": float(res.e_dyn),
                "e_hf": float(res.e_hf),
                "e_singlet_pair": float(res.e_singlet_pair),
                "n_thc": int(res.n_thc),
                "n_aux": int(res.n_aux),
                "hf_converged": bool(mf_hf.converged),
                "backend": str(res.backend_used),
                "t_hf_s": float(t_hf),
                "t_thc_s": float(res.time_thc_s),
                "t_stat_s": float(res.time_stat_s),
                "t_dyn_s": float(res.time_dyn_s),
                "t_pgft_only_s": float(res.time_tot_s),
                "t_tot_s": float(t_tot),
            }
    except Exception as exc:
        out["status"] = "error"
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()[-800:]
    return out


def collect_molecules_for_datasets(gscdb_root: Path, datasets: List[str]) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    ds_set = set(datasets)
    mol_keys: Set[str] = set()
    with open(gscdb_root / "Info" / "DatasetEval.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Dataset"] in ds_set:
                parts = [p.strip() for p in r["Stoichiometry"].split(",")]
                for i in range(1, len(parts), 2):
                    mol_keys.add(parts[i])
    with open(gscdb_root / "Allmols_info.json", encoding="utf-8") as f:
        all_mols = json.load(f)
    selected = {mk: all_mols[mk] for mk in sorted(mol_keys) if mk in all_mols}
    # Sort descending by estimated size (num_basis in QZVPPD is a great proxy for LPT scheduling!)
    ordered_keys = sorted(selected.keys(), key=lambda k: -(int(selected[k].get("num_basis", 0) or 0)))
    return ordered_keys, selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gscdb", default="/content/GSCDB")
    ap.add_argument("--datasets", required=True, help="Comma-separated dataset names")
    ap.add_argument("--basis", default="def2-SVP")
    ap.add_argument("--methods", default="PGFT,PBE,CF22D")
    ap.add_argument("--c-thc", type=float, default=18.0)
    ap.add_argument("--n-rad", type=int, default=28)
    ap.add_argument("--grid", default="99_590", choices=["default", "99_590"])
    ap.add_argument("--density-fit", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gscdb_root = Path(args.gscdb)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys: Set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    rec = json.loads(ln)
                    if rec.get("status") == "ok":
                        done_keys.add(rec["mol_key"])

    ordered_keys, mol_map = collect_molecules_for_datasets(gscdb_root, datasets)
    todo_keys = [k for k in ordered_keys if k not in done_keys]

    print(
        f"[Worker] Datasets={datasets} | Basis={args.basis} | Methods={methods} | "
        f"c_thc={args.c_thc} n_rad={args.n_rad} grid={args.grid} | "
        f"Total={len(ordered_keys)} Cached={len(done_keys)} Todo={len(todo_keys)} | Workers={args.workers}",
        flush=True,
    )
    if not todo_keys:
        print("[Worker] All molecules already cached. Done!", flush=True)
        return

    tasks = [(mk, mol_map[mk], args.basis, methods, args.c_thc, args.n_rad, args.grid, args.density_fit) for mk in todo_keys]
    t_start = time.perf_counter()
    completed = 0
    errors = 0

    with open(out_path, "a", encoding="utf-8") as fout:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(eval_one_molecule, t) for t in tasks]
            for fut in concurrent.futures.as_completed(futs):
                rec = fut.result()
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                completed += 1
                if rec["status"] != "ok":
                    errors += 1
                    print(f"  [ERR {completed}/{len(todo_keys)}] {rec['mol_key']}: {rec.get('error')}", flush=True)
                elif completed % 10 == 0 or completed == len(todo_keys) or rec.get("PGFT", {}).get("t_tot_s", 0) > 15.0:
                    elapsed = time.perf_counter() - t_start
                    pg_t = rec.get("PGFT", {}).get("t_tot_s", 0.0)
                    pbe_t = rec.get("PBE", {}).get("t_tot_s", 0.0)
                    cf_t = rec.get("CF22D", {}).get("t_tot_s", 0.0)
                    print(
                        f"  [{completed:4d}/{len(todo_keys):4d} | {elapsed:6.1f}s] {rec['mol_key']:32s} "
                        f"NAO={rec.get('nao',0):3d} | PBE={pbe_t:5.2f}s CF22D={cf_t:5.2f}s PGFT={pg_t:6.2f}s",
                        flush=True,
                    )

    print(
        f"[Worker] FINISHED {completed} molecules ({errors} errors) in {time.perf_counter() - t_start:.1f}s -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
