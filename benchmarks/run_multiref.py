#!/usr/bin/env python3
"""
PGFT Multireference Dissociation, Diradical & 1-Electron SIE Benchmark Suite
============================================================================
Evaluates Pauli-Purified Gor'kov Fluctuation Theory (`run_pgft(mol)`) across all
multireference, diradical, two-electron, and one-electron self-interaction curves in
`/home/antigravity/project/tests/reference_data.json`, checkpoints after every curve to
`/home/antigravity/project/benchmarks/multiref_results.json`, and generates Nature/PRX-style
300 DPI Figures 1 and 2 in `/home/antigravity/project/benchmarks/figures/`.
"""

from __future__ import annotations

import os
import sys

# Enforce single-threaded BLAS/OpenMP per worker process to avoid thread contention
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import concurrent.futures
import datetime
import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from pyscf import dft, gto, scf

PROJECT_ROOT = "/home/antigravity/project"
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
TESTS_DIR = os.path.join(PROJECT_ROOT, "tests")
BENCH_DIR = os.path.join(PROJECT_ROOT, "benchmarks")
FIG_DIR = os.path.join(BENCH_DIR, "figures")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from pgft import run_pgft
import gauntlet_reference as gr

HARTREE_TO_KCAL = gr.HARTREE_TO_KCAL
REF_JSON_PATH = os.path.join(TESTS_DIR, "reference_data.json")
OUT_JSON_PATH = os.path.join(BENCH_DIR, "multiref_results.json")


# =============================================================================
# Worker & Checkpointing Utilities
# =============================================================================

def eval_pgft_from_spec(
    spec: Dict[str, Any],
    check_rhf_stability: bool = False,
) -> Dict[str, Any]:
    """Reconstruct PySCF Mole from spec, run safe RHF/ROHF + O(N^3) PGFT, return dict."""
    mol = gr.reconstruct_mol_from_spec(spec)
    mf = scf.RHF(mol) if mol.spin == 0 else scf.ROHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 150
    mf = gr.safe_scf_kernel(mf)

    if check_rhf_stability and mol.spin == 0:
        mo_i, _, stable_i, _ = mf.stability(return_status=True, verbose=0)
        if not stable_i:
            dm_new = mf.make_rdm1(mo_i, mf.mo_occ)
            mf_retry = scf.RHF(mol)
            mf_retry.conv_tol = 1e-10
            mf_retry.max_cycle = 150
            mf = gr.safe_scf_kernel(mf_retry, dm0=dm_new)

    res = run_pgft(mol, mf_init=mf)
    return {
        "e_tot": float(res.e_tot),
        "e_stat": float(res.e_stat),
        "e_dyn": float(res.e_dyn),
        "e_hf": float(res.e_hf),
        "hf_converged": bool(mf.converged),
        "n_alpha_top6": [round(float(x), 6) for x in res.n_alpha[:6]],
        "n_beta_top6": [round(float(x), 6) for x in res.n_beta[:6]],
        "n_thc": int(res.n_thc),
        "n_aux": int(res.n_aux),
        "time_thc_s": round(float(res.time_thc_s), 4),
        "time_stat_s": round(float(res.time_stat_s), 4),
        "time_dyn_s": round(float(res.time_dyn_s), 4),
        "time_tot_s": round(float(res.time_tot_s), 4),
    }


def _worker_task(args: Tuple[int, Dict[str, Any], bool]) -> Tuple[int, Dict[str, Any]]:
    idx, spec, check_stab = args
    return idx, eval_pgft_from_spec(spec, check_rhf_stability=check_stab)


def eval_specs_batch(
    specs: List[Dict[str, Any]],
    check_rhf_stability: bool = False,
    max_workers: int = 2,
) -> List[Dict[str, Any]]:
    """Evaluate a list of molecular specs deterministically using up to `max_workers` processes."""
    if len(specs) == 0:
        return []
    if len(specs) == 1 or max_workers <= 1:
        return [eval_pgft_from_spec(sp, check_rhf_stability=check_rhf_stability) for sp in specs]

    tasks = [(i, sp, check_rhf_stability) for i, sp in enumerate(specs)]
    results: List[Optional[Dict[str, Any]]] = [None] * len(specs)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        for idx, out in pool.map(_worker_task, tasks):
            results[idx] = out
    return [r for r in results if r is not None]


def save_checkpoint(data: Dict[str, Any], path: str = OUT_JSON_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def curve_mae_and_npe_kcal(
    pts: List[Dict[str, Any]],
    ref_key: str = "Exact_FCI",
) -> Dict[str, Dict[str, Optional[float]]]:
    """Compute MAE and NPE (max(E - E_ref) - min(E - E_ref)) in kcal/mol across a PEC."""
    methods = set()
    for p in pts:
        methods.update(p["energies"].keys())
    methods.discard(ref_key)

    stats: Dict[str, Dict[str, Optional[float]]] = {}
    for m in sorted(methods):
        errs = []
        all_present = True
        for p in pts:
            v = p["energies"].get(m)
            v_ref = p["energies"].get(ref_key)
            if v is None or v_ref is None:
                all_present = False
            elif v is not None and v_ref is not None:
                errs.append((v - v_ref) * HARTREE_TO_KCAL)
        if len(errs) == 0:
            stats[m] = {"mae_kcal_mol": None, "npe_kcal_mol": None, "complete_curve": False}
        else:
            stats[m] = {
                "mae_kcal_mol": float(np.mean(np.abs(errs))),
                "max_abs_err_kcal_mol": float(np.max(np.abs(errs))),
                "npe_kcal_mol": float(np.max(errs) - np.min(errs)) if len(errs) >= 2 else 0.0,
                "n_converged_points": len(errs),
                "complete_curve": bool(all_present),
            }
    return stats


# =============================================================================
# Section Runners (Checkpoint after EVERY curve)
# =============================================================================

def run_all_benchmarks(rebuild: bool = False, max_workers: int = 2) -> Dict[str, Any]:
    t_global = time.perf_counter()
    with open(REF_JSON_PATH, "r", encoding="utf-8") as f:
        ref = json.load(f)

    if (not rebuild) and os.path.exists(OUT_JSON_PATH):
        with open(OUT_JSON_PATH, "r", encoding="utf-8") as f:
            results: Dict[str, Any] = json.load(f)
    else:
        results = {}

    results["metadata"] = {
        "solver": "Pauli-Purified Gor'kov Fluctuation Theory (PGFT, O(N^3))",
        "reference_json": REF_JSON_PATH,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hartree_to_kcal_mol": HARTREE_TO_KCAL,
    }

    # -------------------------------------------------------------------------
    # SECTION 1: 1-Electron Self-Interaction Freedom (cc-pVDZ)
    # -------------------------------------------------------------------------
    s1_ref = ref["section_1_one_electron_sie"]
    s1_out = results.setdefault("section_1_one_electron_sie", {"basis": s1_ref["basis"]})

    if "H_atom_and_H2_plus_curve" not in s1_out:
        print("[Sec 1] Evaluating H atom and H2+ dissociation curve (cc-pVDZ)...", flush=True)
        t0 = time.perf_counter()
        specs_s1 = [s1_ref["H_atom"]["spec"]] + [pt["spec"] for pt in s1_ref["H2_plus_curve"]]
        pgft_s1 = eval_specs_batch(specs_s1, max_workers=max_workers)

        h_pgft = pgft_s1[0]
        h_energies = dict(s1_ref["H_atom"]["energies"])
        h_energies["PGFT"] = h_pgft["e_tot"]
        e_h_exact = h_energies["Exact_FCI"]
        e_h_pgft = h_pgft["e_tot"]

        h2p_pts = []
        for pt_ref, pg in zip(s1_ref["H2_plus_curve"], pgft_s1[1:]):
            ens = dict(pt_ref["energies"])
            ens["PGFT"] = pg["e_tot"]
            bind = dict(pt_ref["binding_kcal_mol"])
            bind["PGFT"] = float((pg["e_tot"] - e_h_exact) * HARTREE_TO_KCAL)
            bind["PGFT_self_ref"] = float((pg["e_tot"] - e_h_pgft) * HARTREE_TO_KCAL)
            sie = dict(pt_ref["sie_error_kcal_mol"])
            sie["PGFT"] = float((pg["e_tot"] - ens["Exact_FCI"]) * HARTREE_TO_KCAL)
            h2p_pts.append({
                "R_angstrom": pt_ref["R_angstrom"],
                "spec": pt_ref["spec"],
                "energies": ens,
                "pgft_detail": pg,
                "binding_kcal_mol": bind,
                "sie_error_kcal_mol": sie,
            })

        s1_out["H_atom_and_H2_plus_curve"] = {
            "H_atom": {
                "spec": s1_ref["H_atom"]["spec"],
                "energies": h_energies,
                "pgft_detail": h_pgft,
                "sie_error_ha": float(h_pgft["e_tot"] - e_h_exact),
                "sie_error_kcal_mol": float((h_pgft["e_tot"] - e_h_exact) * HARTREE_TO_KCAL),
            },
            "H2_plus_curve": h2p_pts,
            "max_abs_pgft_sie_kcal_mol": float(
                max(abs(p["sie_error_kcal_mol"]["PGFT"]) for p in h2p_pts)
            ),
            "R6p0_sie_kcal_mol": h2p_pts[-1]["sie_error_kcal_mol"],
        }
        save_checkpoint(results)
        print(f"  -> Done in {time.perf_counter() - t0:.2f}s", flush=True)

    # -------------------------------------------------------------------------
    # SECTION 2 (Part A) + SECTION 1 Anti-Cheat: 2e Ions & H2+(4A)...He(100A)
    # -------------------------------------------------------------------------
    s2_ref = ref["section_2_two_electron_correlation"]
    s2_out = results.setdefault("section_2_two_electron_correlation", {"basis": s2_ref["basis"]})

    if "two_electron_ions" not in s2_out or "anti_cheat_H2plus_4A_plus_He_100A" not in s1_out:
        print("[Sec 1 & 2a] Evaluating 2e ions (H-, He, Li+) & H2+(4A)...He(100A) anti-cheat...", flush=True)
        t0 = time.perf_counter()
        ion_keys = ["H_minus", "He", "Li_plus"]
        ac_ref = s1_ref["anti_cheat_H2plus_4A_plus_He_100A"]
        specs_ions_ac = [s2_ref["two_electron_ions"][k]["spec"] for k in ion_keys] + [
            ac_ref["supersystem"]["spec"]
        ]
        pgft_ions_ac = eval_specs_batch(specs_ions_ac, max_workers=max_workers)

        ions_out = {}
        for k, pg in zip(ion_keys, pgft_ions_ac[:3]):
            iref = s2_ref["two_electron_ions"][k]
            ens = dict(iref["energies"])
            ens["PGFT"] = pg["e_tot"]
            errs_kcal = {
                m: float((v - ens["Exact_FCI"]) * HARTREE_TO_KCAL)
                for m, v in ens.items()
                if v is not None
            }
            errs_mha = {
                m: float((v - ens["Exact_FCI"]) * 1000.0)
                for m, v in ens.items()
                if v is not None
            }
            ions_out[k] = {
                "spec": iref["spec"],
                "energies": ens,
                "pgft_detail": pg,
                "exact_correlation_energy_ha": iref["exact_correlation_energy_ha"],
                "pgft_correlation_energy_ha": float(pg["e_tot"] - pg["e_hf"]),
                "errors_vs_fci_kcal_mol": errs_kcal,
                "errors_vs_fci_mHa": errs_mha,
            }
        s2_out["two_electron_ions"] = ions_out

        # Anti-cheat H2+(4A)...He(100A)
        pg_super = pgft_ions_ac[3]
        pg_he = ions_out["He"]["pgft_detail"]
        h2p_4_pt = [
            p for p in s1_out["H_atom_and_H2_plus_curve"]["H2_plus_curve"]
            if abs(p["R_angstrom"] - 4.0) < 1e-6
        ][0]
        e_h2p_4_pgft = h2p_4_pt["pgft_detail"]["e_tot"]
        e_super_fci = ac_ref["supersystem"]["energies"]["Exact_FCI"]
        add_err_ha = float(pg_super["e_tot"] - (e_h2p_4_pgft + pg_he["e_tot"]))
        extracted_h2p_4_pgft = float(pg_super["e_tot"] - pg_he["e_tot"])

        super_ens = dict(ac_ref["supersystem"]["energies"])
        super_ens["PGFT"] = pg_super["e_tot"]
        s1_out["anti_cheat_H2plus_4A_plus_He_100A"] = {
            "spec": ac_ref["supersystem"]["spec"],
            "energies": super_ens,
            "pgft_detail": pg_super,
            "E_H2plus_4A_pgft": e_h2p_4_pgft,
            "E_He_pgft": pg_he["e_tot"],
            "E_H2plus_4A_exact": ac_ref["E_H2plus_4A_exact"],
            "E_He_exact": ac_ref["E_He_exact"],
            "pgft_additivity_error_ha": add_err_ha,
            "pgft_additivity_error_kcal_mol": float(add_err_ha * HARTREE_TO_KCAL),
            "pgft_extracted_H2plus_4A_sie_kcal_mol": float(
                (extracted_h2p_4_pgft - ac_ref["E_H2plus_4A_exact"]) * HARTREE_TO_KCAL
            ),
            "pgft_total_error_vs_fci_kcal_mol": float(
                (pg_super["e_tot"] - e_super_fci) * HARTREE_TO_KCAL
            ),
        }
        save_checkpoint(results)
        print(f"  -> Done in {time.perf_counter() - t0:.2f}s", flush=True)

    # -------------------------------------------------------------------------
    # SECTION 2 (Part B): H2 Full Dissociation Curve (cc-pVDZ) + 2 E(H)
    # -------------------------------------------------------------------------
    if "H2_dissociation_curve" not in s2_out:
        print("[Sec 2b] Evaluating H2 full PEC (R = 0.5 .. 4.5 A, cc-pVDZ)...", flush=True)
        t0 = time.perf_counter()
        specs_h2 = [pt["spec"] for pt in s2_ref["H2_dissociation_curve"]]
        pgft_h2 = eval_specs_batch(specs_h2, max_workers=max_workers)

        e_h_pgft = s1_out["H_atom_and_H2_plus_curve"]["H_atom"]["pgft_detail"]["e_tot"]
        e_2h_pgft = 2.0 * e_h_pgft
        e_2h_exact = s2_ref["H2_exact_2H_asymptote_ha"]

        h2_pts = []
        for pt_ref, pg in zip(s2_ref["H2_dissociation_curve"], pgft_h2):
            ens = dict(pt_ref["energies"])
            ens["PGFT"] = pg["e_tot"]
            errs = dict(pt_ref["errors_vs_fci_kcal_mol"])
            errs["PGFT"] = float((pg["e_tot"] - ens["Exact_FCI"]) * HARTREE_TO_KCAL)
            h2_pts.append({
                "R_angstrom": pt_ref["R_angstrom"],
                "spec": pt_ref["spec"],
                "energies": ens,
                "pgft_detail": pg,
                "errors_vs_fci_kcal_mol": errs,
            })

        eq_pt = [p for p in h2_pts if abs(p["R_angstrom"] - 0.74) < 1e-6][0]
        s2_out["H2_dissociation_curve"] = {
            "points": h2_pts,
            "exact_2H_asymptote_ha": e_2h_exact,
            "pgft_2H_asymptote_ha": e_2h_pgft,
            "De_exact_kcal_mol": float((e_2h_exact - eq_pt["energies"]["Exact_FCI"]) * HARTREE_TO_KCAL),
            "De_pgft_kcal_mol": float((e_2h_pgft - eq_pt["energies"]["PGFT"]) * HARTREE_TO_KCAL),
            "R4p5_minus_2H_pgft_kcal_mol": float((h2_pts[-1]["energies"]["PGFT"] - e_2h_pgft) * HARTREE_TO_KCAL),
            "curve_metrics": curve_mae_and_npe_kcal(h2_pts, ref_key="Exact_FCI"),
        }
        save_checkpoint(results)
        print(f"  -> Done in {time.perf_counter() - t0:.2f}s", flush=True)

    # -------------------------------------------------------------------------
    # SECTION 3: H4 Diradical Circle (6-31G AND cc-pVDZ, theta = 70 .. 110 deg)
    # -------------------------------------------------------------------------
    s3_ref = ref["section_3_h4_diradical_circle"]
    s3_out = results.setdefault(
        "section_3_h4_diradical_circle",
        {
            "radius_angstrom": s3_ref["radius_angstrom"],
            "angles_deg": s3_ref["angles_deg"],
            "curves": {},
        },
    )

    for basis in ["6-31g", "cc-pvdz"]:
        if basis not in s3_out["curves"]:
            print(f"[Sec 3] Evaluating H4 diradical circle ({basis}, 11 angles)...", flush=True)
            t0 = time.perf_counter()
            b_ref = s3_ref["curves"][basis]
            specs_h4 = [pt["spec"] for pt in b_ref["points"]]
            pgft_h4 = eval_specs_batch(specs_h4, check_rhf_stability=True, max_workers=max_workers)

            pts_h4 = []
            by_theta: Dict[float, Dict[str, Any]] = {}
            for pt_ref, pg in zip(b_ref["points"], pgft_h4):
                ens = dict(pt_ref["energies"])
                ens["PGFT"] = pg["e_tot"]
                errs = {
                    m: float((v - ens["Exact_FCI"]) * HARTREE_TO_KCAL)
                    for m, v in ens.items()
                    if v is not None
                }
                item = {
                    "theta_deg": pt_ref["theta_deg"],
                    "spec": pt_ref["spec"],
                    "energies": ens,
                    "pgft_detail": pg,
                    "errors_vs_fci_kcal_mol": errs,
                }
                pts_h4.append(item)
                by_theta[pt_ref["theta_deg"]] = item

            cusp_mha = dict(b_ref["cusp_metric_mHa_at_90deg"])
            e90 = by_theta[90.0]["energies"]["PGFT"]
            e895 = by_theta[89.5]["energies"]["PGFT"]
            e905 = by_theta[90.5]["energies"]["PGFT"]
            e70 = by_theta[70.0]["energies"]["PGFT"]
            cusp_mha["PGFT"] = float((e90 - 0.5 * (e895 + e905)) * 1000.0)

            barrier_kcal = dict(b_ref["barrier_90_minus_70_kcal_mol"])
            barrier_kcal["PGFT"] = float((e90 - e70) * HARTREE_TO_KCAL)

            s3_out["curves"][basis] = {
                "points": pts_h4,
                "cusp_metric_mHa_at_90deg": cusp_mha,
                "barrier_90_minus_70_kcal_mol": barrier_kcal,
                "curve_metrics": curve_mae_and_npe_kcal(pts_h4, ref_key="Exact_FCI"),
            }
            save_checkpoint(results)
            print(
                f"  -> Done ({basis}) in {time.perf_counter() - t0:.2f}s | "
                f"PGFT cusp = {cusp_mha['PGFT']:+.4f} mHa, barrier = {barrier_kcal['PGFT']:.2f} kcal/mol",
                flush=True,
            )

    # -------------------------------------------------------------------------
    # SECTION 4: Multi-Bond & Heteronuclear Dissociation Curves
    # -------------------------------------------------------------------------
    s4_ref = ref["section_4_multibond_breaking"]
    s5_atoms_631g = ref["section_5_equilibrium_chemistry"]["thermochemistry"]["6-31g"]["atoms"]
    s4_out = results.setdefault("section_4_multibond_breaking", {})

    # 4.0 Isolated Open-Shell Atomic Asymptotes (6-31G: H, Li, N, O, F; STO-3G: N)
    if "isolated_atoms" not in s4_out:
        print("[Sec 4.0] Evaluating isolated open-shell atoms (H, Li, N, O, F in 6-31G; N in STO-3G)...", flush=True)
        t0 = time.perf_counter()
        atom_tasks = [
            ("H_631g", s5_atoms_631g["H"]["spec"], s5_atoms_631g["H"]["energies"]),
            ("Li_631g", s5_atoms_631g["Li"]["spec"], s5_atoms_631g["Li"]["energies"]),
            ("N_631g", s5_atoms_631g["N"]["spec"], s5_atoms_631g["N"]["energies"]),
            ("O_631g", s5_atoms_631g["O"]["spec"], s5_atoms_631g["O"]["energies"]),
            ("F_631g", s5_atoms_631g["F"]["spec"], s5_atoms_631g["F"]["energies"]),
            (
                "N_sto3g",
                gr.mol_spec(gr.build_mol("N 0 0 0", basis="sto-3g", spin=3)),
                {"Exact_FCI": 0.5 * s4_ref["N2_triple_bond_sto3g"]["exact_2N4S_asymptote_ha"]},
            ),
        ]
        pgft_atoms = eval_specs_batch([x[1] for x in atom_tasks], max_workers=max_workers)
        atoms_dict = {}
        for (label, sp, ref_ens), pg in zip(atom_tasks, pgft_atoms):
            ens = dict(ref_ens)
            ens["PGFT"] = pg["e_tot"]
            atoms_dict[label] = {
                "spec": sp,
                "energies": ens,
                "pgft_detail": pg,
                "error_vs_fci_kcal_mol": float((pg["e_tot"] - ens["Exact_FCI"]) * HARTREE_TO_KCAL),
            }
        s4_out["isolated_atoms"] = atoms_dict
        save_checkpoint(results)
        print(f"  -> Done in {time.perf_counter() - t0:.2f}s", flush=True)

    atoms_d = s4_out["isolated_atoms"]

    # 4.1 HF Stretch (6-31G, R = 0.75 .. 3.6 A) + F(^2P) + H
    if "HF_stretch_631g" not in s4_out:
        print("[Sec 4.1] Evaluating HF stretch (6-31G, R = 0.75 .. 3.6 A)...", flush=True)
        t0 = time.perf_counter()
        c_ref = s4_ref["HF_stretch_631g"]
        pg_list = eval_specs_batch([p["spec"] for p in c_ref["points"]], max_workers=max_workers)
        pts = []
        for pt_ref, pg in zip(c_ref["points"], pg_list):
            ens = dict(pt_ref["energies"])
            ens["PGFT"] = pg["e_tot"]
            pts.append({
                "R_angstrom": pt_ref["R_angstrom"],
                "spec": pt_ref["spec"],
                "energies": ens,
                "pgft_detail": pg,
                "errors_vs_fci_kcal_mol": {
                    m: float((v - ens["Exact_FCI"]) * HARTREE_TO_KCAL)
                    for m, v in ens.items() if v is not None
                },
            })
        e_asym_fci = c_ref["exact_F2P_plus_H2S_asymptote_ha"]
        e_asym_pgft = atoms_d["F_631g"]["energies"]["PGFT"] + atoms_d["H_631g"]["energies"]["PGFT"]
        eq_pt = [p for p in pts if abs(p["R_angstrom"] - 0.917) < 1e-6][0]
        s4_out["HF_stretch_631g"] = {
            "exact_F2P_plus_H2S_asymptote_ha": e_asym_fci,
            "pgft_F2P_plus_H2S_asymptote_ha": float(e_asym_pgft),
            "De_exact_kcal_mol": float((e_asym_fci - eq_pt["energies"]["Exact_FCI"]) * HARTREE_TO_KCAL),
            "De_pgft_kcal_mol": float((e_asym_pgft - eq_pt["energies"]["PGFT"]) * HARTREE_TO_KCAL),
            "R3p6_minus_atoms_pgft_kcal_mol": float((pts[-1]["energies"]["PGFT"] - e_asym_pgft) * HARTREE_TO_KCAL),
            "points": pts,
            "curve_metrics": curve_mae_and_npe_kcal(pts, ref_key="Exact_FCI"),
        }
        save_checkpoint(results)
        print(f"  -> Done in {time.perf_counter() - t0:.2f}s", flush=True)

    # 4.2 LiH Stretch (6-31G, R = 0.8 .. 3.6 A) + Li(^2S) + H
    if "LiH_stretch_631g" not in s4_out:
        print("[Sec 4.2] Evaluating LiH stretch (6-31G, R = 0.8 .. 3.6 A)...", flush=True)
        t0 = time.perf_counter()
        c_ref = s4_ref["LiH_stretch_631g"]
        pg_list = eval_specs_batch([p["spec"] for p in c_ref["points"]], max_workers=max_workers)
        pts = []
        for pt_ref, pg in zip(c_ref["points"], pg_list):
            ens = dict(pt_ref["energies"])
            ens["PGFT"] = pg["e_tot"]
            pts.append({
                "R_angstrom": pt_ref["R_angstrom"],
                "spec": pt_ref["spec"],
                "energies": ens,
                "pgft_detail": pg,
                "errors_vs_fci_kcal_mol": {
                    m: float((v - ens["Exact_FCI"]) * HARTREE_TO_KCAL)
                    for m, v in ens.items() if v is not None
                },
            })
        e_asym_fci = c_ref["exact_Li2S_plus_H2S_asymptote_ha"]
        e_asym_pgft = atoms_d["Li_631g"]["energies"]["PGFT"] + atoms_d["H_631g"]["energies"]["PGFT"]
        eq_pt = [p for p in pts if abs(p["R_angstrom"] - 1.595) < 1e-6][0]
        s4_out["LiH_stretch_631g"] = {
            "exact_Li2S_plus_H2S_asymptote_ha": e_asym_fci,
            "pgft_Li2S_plus_H2S_asymptote_ha": float(e_asym_pgft),
            "De_exact_kcal_mol": float((e_asym_fci - eq_pt["energies"]["Exact_FCI"]) * HARTREE_TO_KCAL),
            "De_pgft_kcal_mol": float((e_asym_pgft - eq_pt["energies"]["PGFT"]) * HARTREE_TO_KCAL),
            "R3p6_minus_atoms_pgft_kcal_mol": float((pts[-1]["energies"]["PGFT"] - e_asym_pgft) * HARTREE_TO_KCAL),
            "points": pts,
            "curve_metrics": curve_mae_and_npe_kcal(pts, ref_key="Exact_FCI"),
        }
        save_checkpoint(results)
        print(f"  -> Done in {time.perf_counter() - t0:.2f}s", flush=True)

    # 4.3 H2O Symmetric Double-Bond Stretch (6-31G, R_OH = 0.8 .. 3.0 A) + O(^3P) + 2H
    if "H2O_symmetric_stretch_631g" not in s4_out:
        print("[Sec 4.3] Evaluating H2O symmetric stretch (6-31G, R_OH = 0.8 .. 3.0 A)...", flush=True)
        t0 = time.perf_counter()
        c_ref = s4_ref["H2O_symmetric_stretch_631g"]
        pg_list = eval_specs_batch([p["spec"] for p in c_ref["points"]], max_workers=max_workers)
        pts = []
        for pt_ref, pg in zip(c_ref["points"], pg_list):
            ens = dict(pt_ref["energies"])
            ens["PGFT"] = pg["e_tot"]
            pts.append({
                "R_OH_angstrom": pt_ref["R_OH_angstrom"],
                "spec": pt_ref["spec"],
                "energies": ens,
                "pgft_detail": pg,
                "unconverged_R_CCSD_last_e": pt_ref.get("unconverged_R_CCSD_last_e"),
                "errors_vs_fci_kcal_mol": {
                    m: float((v - ens["Exact_FCI"]) * HARTREE_TO_KCAL)
                    for m, v in ens.items() if v is not None
                },
            })
        e_asym_fci = c_ref["exact_O3P_plus_2H_asymptote_ha"]
        e_asym_pgft = atoms_d["O_631g"]["energies"]["PGFT"] + 2.0 * atoms_d["H_631g"]["energies"]["PGFT"]
        eq_pt = [p for p in pts if abs(p["R_OH_angstrom"] - 0.96) < 1e-6][0]
        s4_out["H2O_symmetric_stretch_631g"] = {
            "angle_HOH_deg": c_ref["angle_HOH_deg"],
            "exact_O3P_plus_2H_asymptote_ha": e_asym_fci,
            "pgft_O3P_plus_2H_asymptote_ha": float(e_asym_pgft),
            "De_exact_kcal_mol": float((e_asym_fci - eq_pt["energies"]["Exact_FCI"]) * HARTREE_TO_KCAL),
            "De_pgft_kcal_mol": float((e_asym_pgft - eq_pt["energies"]["PGFT"]) * HARTREE_TO_KCAL),
            "R3p0_minus_atoms_pgft_kcal_mol": float((pts[-1]["energies"]["PGFT"] - e_asym_pgft) * HARTREE_TO_KCAL),
            "points": pts,
            "curve_metrics": curve_mae_and_npe_kcal(pts, ref_key="Exact_FCI"),
        }
        save_checkpoint(results)
        print(f"  -> Done in {time.perf_counter() - t0:.2f}s", flush=True)

    # 4.4 N2 Triple-Bond Stretch in STO-3G (R = 0.9 .. 3.2 A + R = 6.0 A vs Exact FCI & 2N(^4S))
    if "N2_triple_bond_sto3g" not in s4_out:
        print("[Sec 4.4] Evaluating N2 triple-bond stretch (STO-3G, R = 0.9 .. 3.2 A + 6.0 A)...", flush=True)
        t0 = time.perf_counter()
        c_ref = s4_ref["N2_triple_bond_sto3g"]
        mol_6a_sto = gr.build_mol("N 0 0 0; N 0 0 6.0", basis="sto-3g", symmetry=False)
        mol_6a_sto_sym = gr.build_mol("N 0 0 0; N 0 0 6.0", basis="sto-3g", symmetry="D2h")
        spec_6a_sto = gr.mol_spec(mol_6a_sto)

        specs_sto = [p["spec"] for p in c_ref["points"]] + [spec_6a_sto]
        pg_list = eval_specs_batch(specs_sto, max_workers=max_workers)

        pts = []
        for pt_ref, pg in zip(c_ref["points"], pg_list[:-1]):
            ens = dict(pt_ref["energies"])
            ens["PGFT"] = pg["e_tot"]
            pts.append({
                "R_angstrom": pt_ref["R_angstrom"],
                "spec": pt_ref["spec"],
                "energies": ens,
                "pgft_detail": pg,
                "unconverged_R_CCSD_last_e": pt_ref.get("unconverged_R_CCSD_last_e"),
                "errors_vs_fci_kcal_mol": {
                    m: float((v - ens["Exact_FCI"]) * HARTREE_TO_KCAL)
                    for m, v in ens.items() if v is not None
                },
            })

        # Exact FCI at R = 6.0 A in STO-3G (takes < 0.05s)
        mf_6a_sym = gr.safe_scf_kernel(scf.RHF(mol_6a_sto_sym))
        e_fci_6a_sto, _ = gr.solve_singlet_fci(mol_6a_sto_sym, mf_6a_sym, wfnsym="Ag", nroots=2)
        pg_6a_sto = pg_list[-1]
        e_2n_fci_sto = c_ref["exact_2N4S_asymptote_ha"]
        e_2n_pgft_sto = 2.0 * atoms_d["N_sto3g"]["energies"]["PGFT"]
        eq_pt = [p for p in pts if abs(p["R_angstrom"] - 1.098) < 1e-6][0]

        s4_out["N2_triple_bond_sto3g"] = {
            "exact_2N4S_asymptote_ha": e_2n_fci_sto,
            "pgft_2N4S_asymptote_ha": float(e_2n_pgft_sto),
            "De_exact_kcal_mol": float((e_2n_fci_sto - eq_pt["energies"]["Exact_FCI"]) * HARTREE_TO_KCAL),
            "De_pgft_kcal_mol": float((e_2n_pgft_sto - eq_pt["energies"]["PGFT"]) * HARTREE_TO_KCAL),
            "points": pts,
            "R_6p0_asymptotic_check": {
                "R_angstrom": 6.0,
                "spec": spec_6a_sto,
                "pgft_detail": pg_6a_sto,
                "Exact_FCI_6A": float(e_fci_6a_sto),
                "pgft_minus_2N4S_pgft_kcal_mol": float((pg_6a_sto["e_tot"] - e_2n_pgft_sto) * HARTREE_TO_KCAL),
                "pgft_stat_minus_2N4S_stat_mHa": float(
                    (pg_6a_sto["e_stat"] - 2.0 * atoms_d["N_sto3g"]["pgft_detail"]["e_stat"]) * 1000.0
                ),
                "pgft_minus_exact_2N4S_fci_kcal_mol": float((pg_6a_sto["e_tot"] - e_2n_fci_sto) * HARTREE_TO_KCAL),
            },
            "curve_metrics": curve_mae_and_npe_kcal(pts, ref_key="Exact_FCI"),
        }
        save_checkpoint(results)
        print(f"  -> Done in {time.perf_counter() - t0:.2f}s", flush=True)

    # 4.5 N2 Triple-Bond Stretch in 6-31G (R = 0.9 .. 3.2 A + R = 6.0 A vs NEVPT2 & Exact 2N(^4S))
    if "N2_triple_bond_631g" not in s4_out:
        print("[Sec 4.5] Evaluating N2 triple-bond stretch (6-31G, R = 0.9 .. 3.2 A + 6.0 A)...", flush=True)
        t0 = time.perf_counter()
        c_ref = s4_ref["N2_triple_bond_631g"]
        mol_6a_631g = gr.build_mol("N 0 0 0; N 0 0 6.0", basis="6-31g", symmetry=False)
        spec_6a_631g = gr.mol_spec(mol_6a_631g)

        specs_631g = [p["spec"] for p in c_ref["points"]] + [spec_6a_631g]
        pg_list = eval_specs_batch(specs_631g, max_workers=max_workers)

        pts = []
        for pt_ref, pg in zip(c_ref["points"], pg_list[:-1]):
            ens = dict(pt_ref["energies"])
            ens["PGFT"] = pg["e_tot"]
            pts.append({
                "R_angstrom": pt_ref["R_angstrom"],
                "spec": pt_ref["spec"],
                "energies": ens,
                "pgft_detail": pg,
                "unconverged_R_CCSD_last_e": pt_ref.get("unconverged_R_CCSD_last_e"),
                "errors_vs_nevpt2_kcal_mol": {
                    m: float((v - ens["CASSCF_10e8o_NEVPT2"]) * HARTREE_TO_KCAL)
                    for m, v in ens.items() if v is not None
                },
            })

        # Also evaluate R-B3LYP at R = 6.0 A for clean reference comparison
        mf_6a_rhf = scf.RHF(mol_6a_631g)
        mf_6a_rhf.conv_tol = 1e-10
        mf_6a_rhf = gr.safe_scf_kernel(mf_6a_rhf)
        rks_6a = dft.RKS(mol_6a_631g, xc="b3lyp")
        rks_6a.grids.level = 3
        rks_6a.conv_tol = 1e-9
        rks_6a = gr.safe_scf_kernel(rks_6a, dm0=mf_6a_rhf.make_rdm1())

        pg_6a_631g = pg_list[-1]
        e_2n_fci_631g = c_ref["exact_all_electron_2N4S_fci_asymptote_ha"]
        e_2n_pgft_631g = 2.0 * atoms_d["N_631g"]["energies"]["PGFT"]
        eq_pt = [p for p in pts if abs(p["R_angstrom"] - 1.098) < 1e-6][0]

        s4_out["N2_triple_bond_631g"] = {
            "exact_all_electron_2N4S_fci_asymptote_ha": e_2n_fci_631g,
            "pgft_2N4S_asymptote_ha": float(e_2n_pgft_631g),
            "De_ccsd_t_to_2N_fci_kcal_mol": float(
                (e_2n_fci_631g - eq_pt["energies"]["R_CCSD_T"]) * HARTREE_TO_KCAL
            ),
            "De_nevpt2_3p2_minus_eq_kcal_mol": float(
                (pts[-1]["energies"]["CASSCF_10e8o_NEVPT2"] - eq_pt["energies"]["CASSCF_10e8o_NEVPT2"]) * HARTREE_TO_KCAL
            ),
            "De_pgft_kcal_mol": float((e_2n_pgft_631g - eq_pt["energies"]["PGFT"]) * HARTREE_TO_KCAL),
            "points": pts,
            "R_6p0_asymptotic_check": {
                "R_angstrom": 6.0,
                "spec": spec_6a_631g,
                "pgft_detail": pg_6a_631g,
                "RHF_6A": float(mf_6a_rhf.e_tot),
                "R_B3LYP_6A": float(rks_6a.e_tot),
                "pgft_minus_2N4S_pgft_kcal_mol": float((pg_6a_631g["e_tot"] - e_2n_pgft_631g) * HARTREE_TO_KCAL),
                "pgft_stat_minus_2N4S_stat_mHa": float(
                    (pg_6a_631g["e_stat"] - 2.0 * atoms_d["N_631g"]["pgft_detail"]["e_stat"]) * 1000.0
                ),
                "pgft_minus_exact_2N4S_fci_kcal_mol": float((pg_6a_631g["e_tot"] - e_2n_fci_631g) * HARTREE_TO_KCAL),
            },
            "curve_metrics_vs_nevpt2": curve_mae_and_npe_kcal(pts, ref_key="CASSCF_10e8o_NEVPT2"),
        }
        save_checkpoint(results)
        print(f"  -> Done in {time.perf_counter() - t0:.2f}s", flush=True)

    results["metadata"]["total_elapsed_wall_s"] = round(time.perf_counter() - t_global, 3)
    save_checkpoint(results)
    return results


# =============================================================================
# Publication Figure Generation (Nature / PRX Style, 300 DPI)
# =============================================================================

def configure_prx_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "legend.fontsize": 8.2,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.size": 2.2,
        "ytick.minor.size": 2.2,
        "savefig.dpi": 300,
        "figure.dpi": 300,
    })


PALETTE = {
    "Exact": "#111827",     # Deep slate-black
    "PGFT": "#0072B2",      # Okabe-Ito rich sapphire blue (Ours)
    "CCSD_T": "#D55E00",    # Okabe-Ito vermilion / burnt orange
    "B3LYP": "#009E73",     # Okabe-Ito bluish green
    "PBE0": "#CC79A7",      # Okabe-Ito reddish purple
    "RHF": "#7F8C8D",       # Neutral cool gray
}


def generate_fig1_multireference_dissociation(results: Dict[str, Any]) -> str:
    """
    Generate Figure 1 (`fig1_multireference_dissociation.png`):
    2x2 panel comparing Exact FCI (or CASSCF(10,8)+NEVPT2 + exact 2N(^4S) limit on N2 6-31G),
    PGFT (Ours, O(N^3)), RHF, R-B3LYP, and R-CCSD(T) on:
      (a) H2 (cc-pVDZ)
      (b) HF (6-31G)
      (c) H2O symmetric double-bond stretch (6-31G)
      (d) N2 triple-bond dissociation (6-31G)
    Plotted as Relative Potential Energy E(R) - E(R_eq) [kcal/mol] so well depths,
    dissociation asymptotes, RHF/R-B3LYP static-correlation over-dissociation, and
    R-CCSD(T) volcano turnovers are simultaneously visible on an apples-to-apples scale.
    """
    configure_prx_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "fig1_multireference_dissociation.png")

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.2), constrained_layout=True)

    def _extract_rel_kcal(pts: List[Dict[str, Any]], r_key: str, m_key: str, r_eq: float):
        eq_pt = [p for p in pts if abs(p[r_key] - r_eq) < 1e-6][0]
        e_eq = eq_pt["energies"][m_key]
        xs, ys = [], []
        for p in pts:
            val = p["energies"].get(m_key)
            if val is not None:
                xs.append(p[r_key])
                ys.append((val - e_eq) * HARTREE_TO_KCAL)
        return np.array(xs), np.array(ys)

    panels = [
        {
            "ax": axes[0, 0],
            "pts": results["section_2_two_electron_correlation"]["H2_dissociation_curve"]["points"],
            "r_key": "R_angstrom",
            "r_eq": 0.74,
            "ref_key": "Exact_FCI",
            "ref_label": "Exact FCI",
            "title": r"$\mathbf{(a)}$ $\mathrm{H_2}$ Single-Bond Stretch ($\mathrm{cc\text{-}pVDZ}$)",
            "xlabel": "Bond length $R(\\mathrm{H}\\text{\u2013}\\mathrm{H})$ ($\\mathrm{\\AA}$)",
            "de_exact": results["section_2_two_electron_correlation"]["H2_dissociation_curve"]["De_exact_kcal_mol"],
            "asym_label": r"Exact $2\mathrm{H}(^2S)$ limit",
            "asym_xy": (4.56, 110.5, "right"),
            "ylim": (-18, 248),
            "xlim": (0.4, 4.65),
        },
        {
            "ax": axes[0, 1],
            "pts": results["section_4_multibond_breaking"]["HF_stretch_631g"]["points"],
            "r_key": "R_angstrom",
            "r_eq": 0.917,
            "ref_key": "Exact_FCI",
            "ref_label": "Exact FCI",
            "title": r"$\mathbf{(b)}$ $\mathrm{HF}$ Polar Heteronuclear Stretch ($\mathrm{6\text{-}31G}$)",
            "xlabel": "Bond length $R(\\mathrm{F}\\text{\u2013}\\mathrm{H})$ ($\\mathrm{\\AA}$)",
            "de_exact": results["section_4_multibond_breaking"]["HF_stretch_631g"]["De_exact_kcal_mol"],
            "asym_label": r"Exact $\mathrm{F}(^2P)+\mathrm{H}(^2S)$ limit",
            "asym_xy": (3.66, 113.5, "right"),
            "ylim": (-18, 265),
            "xlim": (0.65, 3.75),
        },
        {
            "ax": axes[1, 0],
            "pts": results["section_4_multibond_breaking"]["H2O_symmetric_stretch_631g"]["points"],
            "r_key": "R_OH_angstrom",
            "r_eq": 0.96,
            "ref_key": "Exact_FCI",
            "ref_label": "Exact FCI",
            "title": r"$\mathbf{(c)}$ $\mathrm{H_2O}$ Symmetric Double-Bond Stretch ($\mathrm{6\text{-}31G}$)",
            "xlabel": r"Bond length $R_{\mathrm{OH}}$ ($\mathrm{\AA}$, $\theta_{\mathrm{HOH}}=104.5^\circ$)",
            "de_exact": results["section_4_multibond_breaking"]["H2O_symmetric_stretch_631g"]["De_exact_kcal_mol"],
            "asym_label": r"Exact $\mathrm{O}(^3P)+2\mathrm{H}(^2S)$ limit",
            "asym_xy": (0.76, 187.0, "left"),
            "ylim": (-28, 410),
            "xlim": (0.7, 3.12),
        },
        {
            "ax": axes[1, 1],
            "pts": results["section_4_multibond_breaking"]["N2_triple_bond_631g"]["points"],
            "r_key": "R_angstrom",
            "r_eq": 1.098,
            "ref_key": "CASSCF_10e8o_NEVPT2",
            "ref_label": "CASSCF(10,8)+NEVPT2",
            "title": r"$\mathbf{(d)}$ $\mathrm{N_2}$ Triple-Bond Dissociation ($\mathrm{6\text{-}31G}$)",
            "xlabel": "Bond length $R(\\mathrm{N}\\text{\u2013}\\mathrm{N})$ ($\\mathrm{\\AA}$)",
            "de_exact": results["section_4_multibond_breaking"]["N2_triple_bond_631g"]["De_ccsd_t_to_2N_fci_kcal_mol"],
            "asym_label": r"Exact $2\mathrm{N}(^4S)$ FCI $D_e$",
            "asym_xy": (3.38, 179.5, "right"),
            "ylim": (-45, 615),
            "xlim": (0.8, 3.48),
        },
    ]

    for idx, cfg in enumerate(panels):
        ax = cfg["ax"]
        pts = cfg["pts"]
        rk = cfg["r_key"]
        req = cfg["r_eq"]

        ax.grid(True, which="major", linestyle=":", linewidth=0.55, color="#D1D5DB", alpha=0.8)

        # Exact dissociation limit horizontal line
        ax.axhline(
            cfg["de_exact"],
            color="#374151",
            linestyle=(0, (3, 2)),
            linewidth=1.1,
            alpha=0.85,
            zorder=1,
        )
        ax_x, ax_y, ax_ha = cfg["asym_xy"]
        ax.text(
            ax_x, ax_y,
            cfg["asym_label"] + f" ({cfg['de_exact']:.1f} kcal/mol)",
            fontsize=7.4,
            color="#374151",
            ha=ax_ha,
            va="bottom",
        )

        # 1. RHF
        x_rhf, y_rhf = _extract_rel_kcal(pts, rk, "RHF", req)
        ax.plot(
            x_rhf, y_rhf,
            linestyle="--", linewidth=1.4, color=PALETTE["RHF"],
            marker="^", markersize=4.2, alpha=0.9,
            label="RHF" if idx == 0 else None, zorder=2,
        )

        # 2. R-B3LYP
        x_b3, y_b3 = _extract_rel_kcal(pts, rk, "R_B3LYP", req)
        ax.plot(
            x_b3, y_b3,
            linestyle="-.", linewidth=1.55, color=PALETTE["B3LYP"],
            marker="s", markersize=4.2, alpha=0.92,
            label="R-B3LYP" if idx == 0 else None, zorder=3,
        )

        # 3. R-CCSD(T)
        x_cc, y_cc = _extract_rel_kcal(pts, rk, "R_CCSD_T", req)
        ax.plot(
            x_cc, y_cc,
            linestyle="-", linewidth=1.7, color=PALETTE["CCSD_T"],
            marker="D", markersize=4.4, alpha=0.95,
            label="R-CCSD(T)" if idx == 0 else None, zorder=4,
        )
        # Mark where R-CCSD(T) fails to converge (H2O at 3.0 A, N2 at >= 2.6 A)
        if len(x_cc) < len(pts):
            ax.scatter(
                [x_cc[-1]], [y_cc[-1]],
                marker="X", s=64, color="#B91C1C", edgecolor="white", linewidth=0.7, zorder=7,
            )

        # 4. Exact FCI (or CASSCF+NEVPT2 on N2 6-31G)
        x_ref, y_ref = _extract_rel_kcal(pts, rk, cfg["ref_key"], req)
        ax.plot(
            x_ref, y_ref,
            linestyle="-", linewidth=2.5, color=PALETTE["Exact"],
            marker="o", markersize=5.2, markerfacecolor="white", markeredgewidth=1.5,
            label=("Exact FCI / NEVPT2" if idx == 0 else None), zorder=5,
        )

        # 5. PGFT (Ours, O(N^3))
        x_pg, y_pg = _extract_rel_kcal(pts, rk, "PGFT", req)
        ax.plot(
            x_pg, y_pg,
            linestyle="-", linewidth=2.2, color=PALETTE["PGFT"],
            marker="o", markersize=4.6,
            label=r"PGFT (Ours, $\mathcal{O}(N^3)$)" if idx == 0 else None, zorder=6,
        )

        # Panel-specific annotations
        if idx == 1:  # HF
            ax.annotate(
                "R-CCSD(T) turnover\n(\u221220.6 kcal/mol)",
                xy=(x_cc[-1], y_cc[-1]),
                xytext=(2.35, 30.0),
                fontsize=7.6,
                color=PALETTE["CCSD_T"],
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=PALETTE["CCSD_T"], lw=0.9),
            )
        elif idx == 2:  # H2O
            ax.annotate(
                "R-CCSD(T) turnover &\nnon-convergence (3.0 \u00c5)",
                xy=(x_cc[-1], y_cc[-1]),
                xytext=(1.70, 44.0),
                fontsize=7.6,
                color="#B91C1C",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#B91C1C", lw=0.9),
            )
        elif idx == 3:  # N2 6-31G
            # Indicate PGFT 6.0 A asymptotic value cleanly below star in open white space
            n2_sec = results["section_4_multibond_breaking"]["N2_triple_bond_631g"]
            eq_pgft = [p["energies"]["PGFT"] for p in pts if abs(p["R_angstrom"] - 1.098) < 1e-6][0]
            e6_rel = (n2_sec["R_6p0_asymptotic_check"]["pgft_detail"]["e_tot"] - eq_pgft) * HARTREE_TO_KCAL
            ax.scatter([3.38], [e6_rel], marker="*", s=88, color=PALETTE["PGFT"], zorder=7)
            ax.text(
                3.41, e6_rel - 22.0,
                f"PGFT 6.0 \u00c5\n({e6_rel:.1f} kcal/mol)",
                fontsize=7.2, color=PALETTE["PGFT"], ha="right", va="top", fontweight="bold",
            )
            ax.annotate(
                "Catastrophic R-CCSD(T)\nvolcano turnover (\u22122.7 kcal/mol)\n& divergence (R \u2265 2.6 \u00c5)",
                xy=(x_cc[-1], y_cc[-1]),
                xytext=(1.86, 265.0),
                fontsize=7.4,
                color="#B91C1C",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="#FEF2F2", ec="#FCA5A5", lw=0.7),
                arrowprops=dict(arrowstyle="->", color="#B91C1C", lw=1.0),
            )

        ax.set_title(cfg["title"], loc="left", pad=6)
        ax.set_xlabel(cfg["xlabel"])
        ax.set_ylabel(r"Relative Energy $E(R) - E(R_{\mathrm{eq}})$ (kcal/mol)")
        ax.set_xlim(*cfg["xlim"])
        ax.set_ylim(*cfg["ylim"])

        # Add clean MAE/NPE badge in corner
        if idx < 3:
            cm = results["section_2_two_electron_correlation"]["H2_dissociation_curve"]["curve_metrics"] if idx == 0 else (
                results["section_4_multibond_breaking"]["HF_stretch_631g"]["curve_metrics"] if idx == 1 else
                results["section_4_multibond_breaking"]["H2O_symmetric_stretch_631g"]["curve_metrics"]
            )
            mae_pg = cm["PGFT"]["mae_kcal_mol"]
            npe_pg = cm["PGFT"]["npe_kcal_mol"]
            badge = f"PGFT vs FCI:\nMAE = {mae_pg:.2f} kcal/mol\nNPE = {npe_pg:.2f} kcal/mol"
        else:
            cm = results["section_4_multibond_breaking"]["N2_triple_bond_631g"]["curve_metrics_vs_nevpt2"]
            mae_pg = cm["PGFT"]["mae_kcal_mol"]
            npe_pg = cm["PGFT"]["npe_kcal_mol"]
            badge = f"PGFT vs NEVPT2:\nMAE = {mae_pg:.2f} kcal/mol\nNPE = {npe_pg:.2f} kcal/mol"

        ax.text(
            0.035, 0.955, badge,
            transform=ax.transAxes, va="top", ha="left", fontsize=7.6,
            bbox=dict(boxstyle="round,pad=0.3", fc="#EFF6FF", ec="#93C5FD", lw=0.8, alpha=0.94),
        )

    # Shared figure-top legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=5, frameon=True,
        facecolor="white", edgecolor="#D1D5DB",
        bbox_to_anchor=(0.5, 1.045), fontsize=8.8,
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_fig2_diradical_and_sie(results: Dict[str, Any]) -> str:
    """
    Generate Figure 2 (`fig2_diradical_and_sie.png`):
    1x2 panel showing:
      (a) H4 circle diradical transition (cc-pVDZ, relative energy E(theta) - E(70 deg) vs theta,
          with zoomed inset at 88..92 deg showing RHF/B3LYP upward V-cusp and R-CCSD(T) downward
          cusp vs smooth PGFT & Exact FCI).
      (b) H2+ 1-electron dissociation (cc-pVDZ, showing -44.5 kcal/mol B3LYP & -39.6 kcal/mol PBE0
          delocalization catastrophe vs machine-exact PGFT & Exact FCI).
    """
    configure_prx_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "fig2_diradical_and_sie.png")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)

    # -------------------------------------------------------------------------
    # Panel (a): H4 Circle Diradical Transition (cc-pVDZ)
    # -------------------------------------------------------------------------
    h4_sec = results["section_3_h4_diradical_circle"]["curves"]["cc-pvdz"]
    pts_h4 = h4_sec["points"]
    cusps = h4_sec["cusp_metric_mHa_at_90deg"]
    thetas = np.array([p["theta_deg"] for p in pts_h4])

    def _h4_rel(m_key: str) -> np.ndarray:
        e70 = pts_h4[0]["energies"][m_key]
        return np.array([(p["energies"][m_key] - e70) * HARTREE_TO_KCAL for p in pts_h4])

    ax_a.grid(True, which="major", linestyle=":", linewidth=0.55, color="#D1D5DB", alpha=0.8)
    ax_a.axvline(90.0, color="#9CA3AF", linestyle=":", linewidth=1.0, zorder=1)
    ax_a.axhline(0.0, color="#9CA3AF", linestyle=":", linewidth=0.8, zorder=1)

    y_rhf = _h4_rel("RHF")
    y_b3 = _h4_rel("R_B3LYP")
    y_cc = _h4_rel("R_CCSD_T")
    y_fci = _h4_rel("Exact_FCI")
    y_pg = _h4_rel("PGFT")

    ax_a.plot(thetas, y_rhf, "--^", lw=1.4, ms=4.2, color=PALETTE["RHF"],
              label=f"RHF ($\\Delta_{{\\mathrm{{cusp}}}}={cusps['RHF']:+.2f}$ mHa)")
    ax_a.plot(thetas, y_b3, "-.s", lw=1.55, ms=4.2, color=PALETTE["B3LYP"],
              label=f"R-B3LYP ($\\Delta_{{\\mathrm{{cusp}}}}={cusps['R_B3LYP']:+.2f}$ mHa)")
    ax_a.plot(thetas, y_cc, "-D", lw=1.65, ms=4.3, color=PALETTE["CCSD_T"],
              label=f"R-CCSD(T) ($\\Delta_{{\\mathrm{{cusp}}}}={cusps['R_CCSD_T']:+.2f}$ mHa)")
    ax_a.plot(thetas, y_fci, "-o", lw=2.5, ms=5.2, mfc="white", mew=1.5, color=PALETTE["Exact"],
              label=f"Exact FCI ($\\Delta_{{\\mathrm{{cusp}}}}={cusps['Exact_FCI']:+.2f}$ mHa)")
    ax_a.plot(thetas, y_pg, "-o", lw=2.2, ms=4.5, color=PALETTE["PGFT"],
              label=f"PGFT (Ours, $\\Delta_{{\\mathrm{{cusp}}}}={cusps['PGFT']:+.3f}$ mHa)")

    ax_a.set_title(r"$\mathbf{(a)}$ $\mathrm{H_4}$ Circle Diradical Transition ($\mathrm{cc\text{-}pVDZ}$)", loc="left", pad=6)
    ax_a.set_xlabel(r"Angle $\theta$ (deg, $R = 1.738\ \mathrm{\AA}$)")
    ax_a.set_ylabel(r"Relative Energy $E(\theta) - E(70^\circ)$ (kcal/mol)")
    ax_a.set_xlim(68.5, 111.5)
    ax_a.set_ylim(-10.5, 89.0)
    ax_a.legend(loc="upper left", fontsize=7.3, framealpha=0.94, edgecolor="#D1D5DB")

    # Zoomed cusp inset around theta in [88, 92] deg showing E(theta) - E(90 deg) in mHa
    ax_ins = inset_axes(
        ax_a, width="36%", height="35%",
        loc="upper right",
        bbox_to_anchor=(0.0, -0.055, 0.985, 0.985),
        bbox_transform=ax_a.transAxes,
    )
    mask_z = (thetas >= 88.0) & (thetas <= 92.0)
    th_z = thetas[mask_z]
    idx90 = int(np.where(np.isclose(thetas, 90.0))[0][0])

    for m_key, col, ls, mk in [
        ("RHF", PALETTE["RHF"], "--", "^"),
        ("R_B3LYP", PALETTE["B3LYP"], "-.", "s"),
        ("R_CCSD_T", PALETTE["CCSD_T"], "-", "D"),
        ("Exact_FCI", PALETTE["Exact"], "-", "o"),
        ("PGFT", PALETTE["PGFT"], "-", "o"),
    ]:
        e90 = pts_h4[idx90]["energies"][m_key]
        yz_mha = np.array([(pts_h4[i]["energies"][m_key] - e90) * 1000.0 for i in np.where(mask_z)[0]])
        mfc = "white" if m_key == "Exact_FCI" else col
        ax_ins.plot(th_z, yz_mha, linestyle=ls, marker=mk, markersize=3.4, mfc=mfc, lw=1.3, color=col)

    ax_ins.axhline(0.0, color="#9CA3AF", lw=0.6, ls=":")
    ax_ins.axvline(90.0, color="#9CA3AF", lw=0.6, ls=":")
    ax_ins.set_title(r"$D_{4h}$ Cusp ($\Delta E$ vs $90^\circ$, mHa)", fontsize=7.0, pad=3)
    ax_ins.tick_params(labelsize=6.6)
    ax_ins.set_xlim(87.8, 92.2)

    # -------------------------------------------------------------------------
    # Panel (b): H2+ 1-Electron Self-Interaction Curve (cc-pVDZ)
    # -------------------------------------------------------------------------
    h2p_pts = results["section_1_one_electron_sie"]["H_atom_and_H2_plus_curve"]["H2_plus_curve"]
    rs_h2p = np.array([p["R_angstrom"] for p in h2p_pts])

    b_fci = np.array([p["binding_kcal_mol"]["Exact_FCI"] for p in h2p_pts])
    b_rohf = np.array([p["binding_kcal_mol"]["ROHF"] for p in h2p_pts])
    b_pbe0 = np.array([p["binding_kcal_mol"]["UKS_PBE0"] for p in h2p_pts])
    b_b3 = np.array([p["binding_kcal_mol"]["UKS_B3LYP"] for p in h2p_pts])
    b_pg = np.array([p["binding_kcal_mol"]["PGFT"] for p in h2p_pts])

    sie_b3_6 = h2p_pts[-1]["sie_error_kcal_mol"]["UKS_B3LYP"]
    sie_pbe_6 = h2p_pts[-1]["sie_error_kcal_mol"]["UKS_PBE0"]
    sie_pg_max = results["section_1_one_electron_sie"]["H_atom_and_H2_plus_curve"]["max_abs_pgft_sie_kcal_mol"]
    ac_sie = results["section_1_one_electron_sie"]["anti_cheat_H2plus_4A_plus_He_100A"]["pgft_additivity_error_kcal_mol"]

    ax_b.grid(True, which="major", linestyle=":", linewidth=0.55, color="#D1D5DB", alpha=0.8)
    ax_b.axhline(0.0, color="#374151", linestyle=(0, (3, 2)), linewidth=1.1, zorder=1)
    ax_b.text(6.15, 2.0, r"Exact $\mathrm{H}(^2S) + \mathrm{H^+}$ limit ($0.0$ kcal/mol)",
              fontsize=7.4, color="#374151", ha="right", va="bottom")

    ax_b.plot(rs_h2p, b_b3, "-.s", lw=1.65, ms=4.5, color=PALETTE["B3LYP"],
              label=f"UKS-B3LYP ($6.0\\ \\mathrm{{\\AA}}$ SIE: {sie_b3_6:.1f} kcal/mol)")
    ax_b.plot(rs_h2p, b_pbe0, "--^", lw=1.65, ms=4.5, color=PALETTE["PBE0"],
              label=f"UKS-PBE0 ($6.0\\ \\mathrm{{\\AA}}$ SIE: {sie_pbe_6:.1f} kcal/mol)")
    ax_b.plot(rs_h2p, b_rohf, ":d", lw=1.4, ms=4.2, color=PALETTE["RHF"],
              label="ROHF / UHF (1e exact)")
    ax_b.plot(rs_h2p, b_fci, "-o", lw=2.6, ms=5.5, mfc="white", mew=1.5, color=PALETTE["Exact"],
              label="Exact FCI")
    ax_b.plot(rs_h2p, b_pg, "-o", lw=2.2, ms=4.5, color=PALETTE["PGFT"],
              label=r"PGFT (Ours, $|\mathrm{SIE}| \leq 6.3\times 10^{-10}$ kcal/mol)")

    # Place legend cleanly in wide-open gap between Exact/PGFT (~0) and DFT (~-36) for R in [3.5, 6.2]
    ax_b.legend(
        loc="upper right",
        bbox_to_anchor=(0.985, 0.655),
        fontsize=7.05,
        framealpha=0.95,
        edgecolor="#D1D5DB",
    )

    # Place 1e & 3e Anti-Cheat badge cleanly in open upper-left white space above y = 0
    ax_b.text(
        0.03, 0.965,
        "1e & 3e Anti-Cheat Audit:\n"
        f"\u2022 Max $|E_{{\\mathrm{{PGFT}}}} - E_{{\\mathrm{{FCI}}}}|$: {sie_pg_max:.2e} kcal/mol\n"
        f"\u2022 $\\mathrm{{H_2^+(4\\AA)\\cdots He(100\\AA)}}$ additivity: {ac_sie:+.2e} kcal/mol",
        transform=ax_b.transAxes, ha="left", va="top", fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.28", fc="#EFF6FF", ec="#93C5FD", lw=0.8, alpha=0.95),
    )

    # Annotate DFT delocalization catastrophe in open bottom-right space pointing to (6.0, -44.5)
    ax_b.annotate(
        f"DFT Delocalization Catastrophe:\nB3LYP: {sie_b3_6:.1f} kcal/mol\nPBE0:  {sie_pbe_6:.1f} kcal/mol",
        xy=(5.96, b_b3[-1] - 0.6),
        xytext=(2.65, -67.5),
        fontsize=7.4,
        color="#065F46",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", fc="#ECFDF5", ec="#6EE7B7", lw=0.8),
        arrowprops=dict(arrowstyle="->", color=PALETTE["B3LYP"], lw=1.0),
    )

    ax_b.set_title(r"$\mathbf{(b)}$ $\mathrm{H_2^+}$ One-Electron Dissociation & SIE ($\mathrm{cc\text{-}pVDZ}$)", loc="left", pad=6)
    ax_b.set_xlabel("Bond length $R(\\mathrm{H}\\text{\u2013}\\mathrm{H^+})$ ($\\mathrm{\\AA}$)")
    ax_b.set_ylabel(r"Interaction Energy $E(\mathrm{H_2^+}) - E_{\mathrm{exact}}(\mathrm{H})$ (kcal/mol)")
    ax_b.set_xlim(0.5, 6.3)
    ax_b.set_ylim(-73.5, 25.5)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    plot_only = "--plot-only" in sys.argv
    rebuild = "--rebuild" in sys.argv
    if plot_only:
        with open(OUT_JSON_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = run_all_benchmarks(rebuild=rebuild, max_workers=2)

    f1 = generate_fig1_multireference_dissociation(results)
    f2 = generate_fig2_diradical_and_sie(results)
    print(f"[Figures] Saved:\n  - {f1}\n  - {f2}", flush=True)


if __name__ == "__main__":
    main()
