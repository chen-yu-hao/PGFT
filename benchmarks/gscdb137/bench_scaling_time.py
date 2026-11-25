#!/usr/bin/env python3
"""
Apples-to-Apples Scaling & Wall-Clock Time Benchmark (Resumable):
PGFT v2 (C++20) vs CF22D (Hybrid meta-GGA + D3(0)) vs PBE (GGA)
===============================================================
Measures empirical O(N^alpha) scaling, component breakdowns, exact-ERI vs DF-RI,
basis-set scaling, thermodynamic size-extensivity, and OpenMP thread scaling
on identical hardware and thread counts.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
    t0 = time.perf_counter()
    if mol.natm <= 1:
        return 0.0, time.perf_counter() - t0
    numbers = np.array([mol.atom_charge(i) for i in range(mol.natm)], dtype=np.int32)
    positions = np.array(mol.atom_coords(unit="Bohr"), dtype=np.float64)
    disp = d3i.DispersionModel(numbers, positions)
    param = d3i.ZeroDampingParam(s6=1.0, s8=0.0, rs6=1.53, rs8=1.0, alp=14.0, s9=0.0)
    e_d3 = float(disp.get_dispersion(param, grad=False)["energy"])
    return e_d3, time.perf_counter() - t0


def run_dft_timed(
    mol: gto.Mole,
    xc_name: str,
    density_fit: bool = False,
    grid_spec: str = "default",
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
    if density_fit:
        mf = mf.density_fit()
    mf.xc = "HYB_MGGA_X_CF22D, MGGA_C_CF22D" if xc_name == "CF22D" else "PBE"
    if grid_spec == "99_590":
        mf.grids.atom_grid = (99, 590)
    else:
        mf.grids.level = 3
    mf.conv_tol = 1e-10
    mf.max_cycle = 150
    e_scf = float(mf.kernel())
    if not mf.converged:
        mf = mf.newton()
        e_scf = float(mf.kernel())
    t_scf = time.perf_counter() - t0

    e_d3, t_d3 = calc_d3_cf22d(mol) if xc_name == "CF22D" else (0.0, 0.0)
    return {
        "e_tot": e_scf + e_d3,
        "e_scf": e_scf,
        "e_d3": e_d3,
        "converged": bool(mf.converged),
        "t_scf_s": t_scf,
        "t_d3_s": t_d3,
        "t_tot_s": t_scf + t_d3,
    }


def run_pgft_timed(mol: gto.Mole, c_thc: float = 18.0, n_rad: int = 28) -> Dict[str, Any]:
    t0 = time.perf_counter()
    mf = scf.RHF(mol) if mol.spin == 0 else scf.ROHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 150
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel()
    t_hf = time.perf_counter() - t0

    res = run_pgft(mol, mf_init=mf, backend="cpp", c_thc=c_thc, n_rad=n_rad)
    t_tot = time.perf_counter() - t0
    return {
        "e_tot": float(res.e_tot),
        "e_stat": float(res.e_stat),
        "e_dyn": float(res.e_dyn),
        "e_hf": float(res.e_hf),
        "e_corr": float(res.e_tot - res.e_hf),
        "n_thc": int(res.n_thc),
        "n_aux": int(res.n_aux),
        "backend": str(res.backend_used),
        "t_hf_s": t_hf,
        "t_thc_s": float(res.time_thc_s),
        "t_stat_s": float(res.time_stat_s),
        "t_dyn_s": float(res.time_dyn_s),
        "t_pgft_only_s": float(res.time_tot_s),
        "t_tot_s": t_tot,
    }


def make_h2_chain(k: int, basis: str = "6-31g", r_intra: float = 0.74, spacing: float = 3.0) -> gto.Mole:
    atoms = []
    for j in range(k):
        z0 = j * spacing
        atoms.append(("H", (0.0, 0.0, z0)))
        atoms.append(("H", (0.0, 0.0, z0 + r_intra)))
    return gto.M(atom=atoms, basis=basis, unit="Angstrom", symmetry=False, verbose=0)


def make_alkane(n_c: int, basis: str = "6-31g") -> gto.Mole:
    r_cc, r_ch = 1.53, 1.09
    half_ang = math.radians(109.47122 / 2.0)
    dx = r_cc * math.sin(half_ang)
    dy = r_cc * math.cos(half_ang)
    atoms = []
    for i in range(n_c):
        x = i * dx
        y = 0.0 if (i % 2 == 0) else dy
        atoms.append(("C", (x, y, 0.0)))
        sgn = -1.0 if (i % 2 == 0) else 1.0
        atoms.append(("H", (x, y + sgn * r_ch * math.cos(half_ang), r_ch * math.sin(half_ang))))
        atoms.append(("H", (x, y + sgn * r_ch * math.cos(half_ang), -r_ch * math.sin(half_ang))))
    atoms.append(("H", (-r_ch * math.sin(half_ang), r_ch * math.cos(half_ang), 0.0)))
    x_last = (n_c - 1) * dx
    y_last = 0.0 if ((n_c - 1) % 2 == 0) else dy
    sgn_last = 1.0 if ((n_c - 1) % 2 == 0) else -1.0
    atoms.append(("H", (x_last + r_ch * math.sin(half_ang), y_last + sgn_last * r_ch * math.cos(half_ang), 0.0)))
    return gto.M(atom=atoms, basis=basis, unit="Angstrom", symmetry=False, verbose=0)


def make_water_cluster(n_w: int, basis: str = "6-31g") -> gto.Mole:
    r_oh = 0.96
    ang = math.radians(104.5 / 2.0)
    centers = [
        (0.00, 0.00, 0.00),
        (2.85, 0.00, 0.00),
        (1.42, 2.47, 0.00),
        (1.42, 0.82, 2.33),
        (-1.42, 2.47, 1.40),
        (4.27, 2.47, 1.40),
        (1.42, -2.20, 1.75),
        (1.42, 3.30, 2.33),
    ][:n_w]
    atoms = []
    for idx, (ox, oy, oz) in enumerate(centers):
        phi = 0.7 * idx
        c, s = math.cos(phi), math.sin(phi)
        h1 = (r_oh * math.sin(ang), r_oh * math.cos(ang), 0.0)
        h2 = (-r_oh * math.sin(ang), r_oh * math.cos(ang), 0.0)
        atoms.append(("O", (ox, oy, oz)))
        atoms.append(("H", (ox + c * h1[0] - s * h1[2], oy + h1[1], oz + s * h1[0] + c * h1[2])))
        atoms.append(("H", (ox + c * h2[0] - s * h2[2], oy + h2[1], oz + s * h2[0] + c * h2[2])))
    return gto.M(atom=atoms, basis=basis, unit="Angstrom", symmetry=False, verbose=0)


def fit_power_law(x: List[float], y: List[float]) -> Dict[str, float]:
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2)) + 1e-30
    return {
        "alpha": float(slope),
        "prefactor": float(np.exp(intercept)),
        "r2": float(1.0 - ss_res / ss_tot),
    }


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def eval_point(
    label: str,
    mol: gto.Mole,
    run_pgft_default: bool = True,
    run_pgft_fast: bool = False,
    run_df_dft: bool = True,
    run_fine_cf22d: bool = False,
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "label": label,
        "natm": int(mol.natm),
        "nelec": int(mol.nelectron),
        "nao": int(mol.nao_nr()),
    }
    rec["PBE_exact"] = run_dft_timed(mol, "PBE", density_fit=False, grid_spec="default")
    rec["CF22D_exact"] = run_dft_timed(mol, "CF22D", density_fit=False, grid_spec="default")
    if run_df_dft:
        rec["PBE_DF"] = run_dft_timed(mol, "PBE", density_fit=True, grid_spec="default")
        rec["CF22D_DF"] = run_dft_timed(mol, "CF22D", density_fit=True, grid_spec="default")
    if run_fine_cf22d:
        rec["CF22D_exact_99_590"] = run_dft_timed(mol, "CF22D", density_fit=False, grid_spec="99_590")
    if run_pgft_default:
        rec["PGFT_v2_default"] = run_pgft_timed(mol, c_thc=18.0, n_rad=28)
    if run_pgft_fast:
        rec["PGFT_v2_c10"] = run_pgft_timed(mol, c_thc=10.0, n_rad=20)
    return rec


def run_thread_scaling() -> Dict[str, Any]:
    """Measure wall-time vs thread count (1, 2, 4, 8) via clean subprocesses."""
    print("\n[Family 5] OpenMP / BLAS Thread Scaling (1, 2, 4, 8 threads)...", flush=True)
    worker_code = r"""
import json, os, sys, time
sys.path.insert(0, "/content/PGFT/src")
import numpy as np
from pyscf import dft, gto, scf
import dftd3.interface as d3i
from pgft import run_pgft

sys_name = sys.argv[1]
if sys_name == "H2_12_631g":
    atoms = []
    for j in range(12):
        atoms.append(("H", (0.0, 0.0, j*3.0)))
        atoms.append(("H", (0.0, 0.0, j*3.0 + 0.74)))
    mol = gto.M(atom=atoms, basis="6-31g", symmetry=False, verbose=0)
else:
    mol = gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", basis="def2-SVP", symmetry=False, verbose=0)

# Warmup
mw = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
run_pgft(mw, backend="cpp", c_thc=10.0, n_rad=20)

t0 = time.perf_counter()
m1 = dft.RKS(mol); m1.xc="PBE"; m1.conv_tol=1e-10; m1.kernel()
t_pbe = time.perf_counter() - t0

t0 = time.perf_counter()
m2 = dft.RKS(mol); m2.xc="HYB_MGGA_X_CF22D, MGGA_C_CF22D"; m2.conv_tol=1e-10; m2.kernel()
nums = np.array([mol.atom_charge(i) for i in range(mol.natm)], dtype=np.int32)
pos = np.array(mol.atom_coords(unit="Bohr"), dtype=np.float64)
d3i.DispersionModel(nums, pos).get_dispersion(d3i.ZeroDampingParam(s6=1.0, s8=0.0, rs6=1.53, rs8=1.0, alp=14.0, s9=0.0), grad=False)
t_cf = time.perf_counter() - t0

t0 = time.perf_counter()
mh = scf.RHF(mol); mh.conv_tol=1e-10; mh.kernel()
rp = run_pgft(mol, mf_init=mh, backend="cpp")
t_pg = time.perf_counter() - t0

print(json.dumps({
    "nao": int(mol.nao_nr()),
    "t_pbe_s": t_pbe,
    "t_cf22d_s": t_cf,
    "t_pgft_s": t_pg,
    "t_pgft_thc_s": float(rp.time_thc_s),
    "t_pgft_stat_s": float(rp.time_stat_s),
    "t_pgft_dyn_s": float(rp.time_dyn_s),
}))
"""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for sname in ["H2O_def2SVP", "H2_12_631g"]:
        rows = []
        for nth in [1, 2, 4, 8]:
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(nth)
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            p = subprocess.run(["python3", "-c", worker_code, sname], env=env, capture_output=True, text=True, check=True)
            rec = json.loads(p.stdout.strip().splitlines()[-1])
            rec["omp_threads"] = nth
            rec["openblas_threads"] = 1
            rows.append(rec)
            print(
                f"  {sname:<12s} OMP={nth} (OBLAS=1) | PBE={rec['t_pbe_s']:.3f}s CF22D={rec['t_cf22d_s']:.3f}s "
                f"PGFT={rec['t_pgft_s']:.3f}s (thc={rec['t_pgft_thc_s']:.2f}, stat={rec['t_pgft_stat_s']:.2f}, dyn={rec['t_pgft_dyn_s']:.2f})",
                flush=True,
            )
        out[sname] = rows
    return out


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/content/scaling_time_results.json")
    if out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"Resuming from existing {out_path} (families: {list(results.get('families', {}).keys())})", flush=True)
    else:
        results = {
            "metadata": {
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "families": {},
        }

    # Warmup
    _w = make_h2_chain(2, "sto-3g")
    _ = run_dft_timed(_w, "PBE")
    _ = run_pgft_timed(_w, c_thc=10.0, n_rad=20)

    # Family 1: keep cached (H2)_k if complete
    fam1 = results["families"].get("h2_chain_631g", [])
    done_k = {r["k"] for r in fam1}
    print(f"\n[Family 1] 1D (H2)_k Chain (6-31G, cached={sorted(done_k)})...", flush=True)
    for k in [2, 4, 6, 8, 12, 16, 20, 24]:
        if k in done_k:
            continue
        mol = make_h2_chain(k, basis="6-31g")
        rec = eval_point(f"(H2)_{k} [6-31G]", mol, run_pgft_default=(k <= 16), run_pgft_fast=True, run_df_dft=True, run_fine_cf22d=(k <= 16))
        rec["k"] = k
        fam1.append(rec)
        results["families"]["h2_chain_631g"] = fam1
        save_json(results, out_path)

    # Family 2: 1D Covalent n-Alkanes C_n H_{2n+2} (6-31G): PGFT default up to n=4 (NAO=56), PBE/CF22D up to n=10 (NAO=134)
    fam2 = results["families"].get("alkanes_631g", [])
    done_nc = {r["n_c"] for r in fam2}
    print(f"\n[Family 2] 1D Covalent n-Alkanes C_n H_{{2n+2}} (6-31G, cached={sorted(done_nc)})...", flush=True)
    for n_c in [1, 2, 3, 4, 5, 6, 8, 10]:
        if n_c in done_nc:
            continue
        mol = make_alkane(n_c, basis="6-31g")
        run_pg = (n_c <= 4)
        rec = eval_point(f"C{n_c}H{2*n_c+2} [6-31G]", mol, run_pgft_default=run_pg, run_pgft_fast=False, run_df_dft=True)
        rec["n_c"] = n_c
        fam2.append(rec)
        pg_s = f"{rec['PGFT_v2_default']['t_tot_s']:7.3f}s" if run_pg else "    --- "
        print(
            f"  C{n_c}H{2*n_c+2:<2d} NAO={rec['nao']:3d} | PBE={rec['PBE_exact']['t_tot_s']:6.3f}s (DF={rec['PBE_DF']['t_tot_s']:6.3f}s) | "
            f"CF22D={rec['CF22D_exact']['t_tot_s']:6.3f}s (DF={rec['CF22D_DF']['t_tot_s']:6.3f}s) | PGFT_def={pg_s}",
            flush=True,
        )
        results["families"]["alkanes_631g"] = fam2
        save_json(results, out_path)

    # Family 3: 3D Water Clusters (H2O)_n (6-31G): PGFT default up to n=4 (NAO=52), PBE/CF22D up to n=8 (NAO=104)
    fam3 = results["families"].get("water_clusters_631g", [])
    done_nw = {r["n_w"] for r in fam3}
    print(f"\n[Family 3] 3D Water Clusters (H2O)_n (6-31G, cached={sorted(done_nw)})...", flush=True)
    for n_w in [1, 2, 3, 4, 5, 6, 8]:
        if n_w in done_nw:
            continue
        mol = make_water_cluster(n_w, basis="6-31g")
        run_pg = (n_w <= 4)
        rec = eval_point(f"(H2O)_{n_w} [6-31G]", mol, run_pgft_default=run_pg, run_pgft_fast=False, run_df_dft=True)
        rec["n_w"] = n_w
        fam3.append(rec)
        pg_s = f"{rec['PGFT_v2_default']['t_tot_s']:7.3f}s" if run_pg else "    --- "
        print(
            f"  (H2O)_{n_w} NAO={rec['nao']:3d} | PBE={rec['PBE_exact']['t_tot_s']:6.3f}s (DF={rec['PBE_DF']['t_tot_s']:6.3f}s) | "
            f"CF22D={rec['CF22D_exact']['t_tot_s']:6.3f}s (DF={rec['CF22D_DF']['t_tot_s']:6.3f}s) | PGFT_def={pg_s}",
            flush=True,
        )
        results["families"]["water_clusters_631g"] = fam3
        save_json(results, out_path)

    # Family 4: Basis-Set Scaling on H2O (PGFT default up to cc-pVTZ NAO=58, DFT up to def2-QZVPP NAO=117)
    fam4 = results["families"].get("h2o_basis_scaling", [])
    done_b = {r["basis"] for r in fam4}
    print(f"\n[Family 4] Basis-Set Scaling on H2O (cached={sorted(done_b)})...", flush=True)
    h2o_geom = "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692"
    for bas in ["sto-3g", "6-31g", "def2-SVP", "cc-pVDZ", "aug-cc-pVDZ", "def2-TZVP", "cc-pVTZ", "def2-TZVPPD", "def2-QZVPP"]:
        if bas in done_b:
            continue
        mol = gto.M(atom=h2o_geom, basis=bas, unit="Angstrom", symmetry=False, verbose=0)
        run_pg = (mol.nao_nr() <= 58)
        rec = eval_point(f"H2O [{bas}]", mol, run_pgft_default=run_pg, run_pgft_fast=False, run_df_dft=True, run_fine_cf22d=True)
        rec["basis"] = bas
        fam4.append(rec)
        pg_s = f"{rec['PGFT_v2_default']['t_tot_s']:7.3f}s" if run_pg else "    --- "
        print(
            f"  H2O {bas:<12s} NAO={rec['nao']:3d} | PBE={rec['PBE_exact']['t_tot_s']:6.3f}s | CF22D={rec['CF22D_exact']['t_tot_s']:6.3f}s (99,590={rec['CF22D_exact_99_590']['t_tot_s']:6.3f}s) | PGFT_def={pg_s}",
            flush=True,
        )
        results["families"]["h2o_basis_scaling"] = fam4
        save_json(results, out_path)

    # Family 5: Thread Scaling
    results["thread_scaling"] = run_thread_scaling()
    save_json(results, out_path)

    # Power-law fits
    fits: Dict[str, Any] = {}
    for fname, pts, min_nao in [
        ("h2_chain_631g", fam1, 24),
        ("alkanes_631g", fam2, 17),
        ("water_clusters_631g", fam3, 13),
        ("h2o_basis_scaling", fam4, 13),
    ]:
        sub = [p for p in pts if p["nao"] >= min_nao]
        fits[fname] = {}
        for mkey in ["PBE_exact", "PBE_DF", "CF22D_exact", "CF22D_DF", "PGFT_v2_default", "PGFT_v2_c10"]:
            valid = [p for p in sub if mkey in p]
            if len(valid) >= 3:
                xs = [p["nao"] for p in valid]
                ys = [p[mkey]["t_tot_s"] for p in valid]
                fits[fname][mkey] = fit_power_law(xs, ys)
                if mkey.startswith("PGFT"):
                    fits[fname][f"{mkey}_dyn"] = fit_power_law(xs, [p[mkey]["t_dyn_s"] for p in valid])
                    fits[fname][f"{mkey}_thc"] = fit_power_law(xs, [p[mkey]["t_thc_s"] for p in valid])
                    fits[fname][f"{mkey}_stat"] = fit_power_law(xs, [p[mkey]["t_stat_s"] for p in valid])
    results["scaling_fits"] = fits
    save_json(results, out_path)

    print("\n=== Power-Law Scaling Exponents (t ~ N_AO^alpha) ===", flush=True)
    for fname, fdict in fits.items():
        print(f"\n  [{fname}]")
        for mkey, fd in fdict.items():
            print(f"    {mkey:<22s}: alpha = {fd['alpha']:.2f}  (R^2 = {fd['r2']:.4f})")
    print(f"\nALL SCALING DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
