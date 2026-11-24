#!/usr/bin/env python3
"""
Single-Reference Thermochemistry, Barrier Heights, Dispersion, Size-Consistency & O(N^3) Scaling
================================================================================================
Evaluates Pauli-Purified Gor'kov Fluctuation Theory (`pgft`) across:
  1. Equilibrium Reactions (`6-31g` & `cc-pvdz`, 5 reactions vs CCSD(T) & Exact FCI)
  2. Barrier Heights (`H + H2 -> [H3]^‡`, `NH3` inversion, `H2O` linear inversion, `C2H6` rotation)
  3. `He2` Counterpoise Van der Waals Dispersion Curve (`aug-cc-pvdz`, shared dimer THC pivots)
  4. Size-Consistency, Open-Shell & Heterogeneous Additivity, and 3D `SO(3)` Rotational Invariance
  5. Empirical `O(N^3)` Wall-Time Scaling & Thermodynamic Size-Extensivity on `(H2)_k` (`k = 2..32`)

Checkpoints `/home/antigravity/project/benchmarks/singleref_scaling_results.json` incrementally
and generates `/home/antigravity/project/benchmarks/figures/fig3_thermochem_and_scaling.png` (300 DPI).
"""

from __future__ import annotations

import os
import sys

# Enforce single-threaded BLAS/OpenMP per worker process to avoid futex contention on 2-vCPU VM
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import concurrent.futures
import json
import math
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyscf import cc, df, gto, mp, scf

PROJECT_ROOT = "/home/antigravity/project"
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
TESTS_DIR = os.path.join(PROJECT_ROOT, "tests")
BENCH_DIR = os.path.join(PROJECT_ROOT, "benchmarks")
FIG_DIR = os.path.join(BENCH_DIR, "figures")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import gauntlet_reference as gr
from pgft import PGFTResult, PGFTSolver, THCData, build_isdf_thc, run_pgft

warnings.filterwarnings("ignore", message=".*basis-set-exchange.*")

HARTREE_TO_KCAL = gr.HARTREE_TO_KCAL
REF_JSON_PATH = os.path.join(TESTS_DIR, "reference_data.json")
OUT_JSON_PATH = os.path.join(BENCH_DIR, "singleref_scaling_results.json")
FIG3_PATH = os.path.join(FIG_DIR, "fig3_thermochem_and_scaling.png")


# =============================================================================
# Core Evaluation, Parallel Pool & Incremental Checkpointing Helpers
# =============================================================================

def save_checkpoint(data: Dict[str, Any], path: str = OUT_JSON_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def eval_pgft_mol(mol: gto.Mole, **pgft_kwargs: Any) -> Dict[str, Any]:
    """Run converged RHF/ROHF + PGFT on `mol` and return structured JSON-serializable dict."""
    mf = scf.RHF(mol) if mol.spin == 0 else scf.ROHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 150
    mf = gr.safe_scf_kernel(mf)

    res: PGFTResult = run_pgft(mol, mf_init=mf, **pgft_kwargs)
    return {
        "e_tot": float(res.e_tot),
        "e_stat": float(res.e_stat),
        "e_dyn": float(res.e_dyn),
        "e_hf": float(res.e_hf),
        "e_corr": float(res.e_tot - res.e_hf),
        "nao": int(mol.nao_nr()),
        "n_aux": int(res.n_aux),
        "n_thc": int(res.n_thc),
        "occ_alpha_top6": [round(float(x), 6) for x in res.n_alpha[:6]],
        "occ_beta_top6": [round(float(x), 6) for x in res.n_beta[:6]],
        "time_thc_s": float(res.time_thc_s),
        "time_stat_s": float(res.time_stat_s),
        "time_dyn_s": float(res.time_dyn_s),
        "time_tot_s": float(res.time_tot_s),
    }


def _eval_spec_worker(args: Tuple[str, Dict[str, Any], Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    key, spec, kwargs = args
    mol = gr.reconstruct_mol_from_spec(spec)
    return key, eval_pgft_mol(mol, **kwargs)


def eval_named_specs_cached(
    named_specs: List[Tuple[str, Dict[str, Any], Dict[str, Any]]],
    cache_dict: Dict[str, Any],
    full_results: Dict[str, Any],
    max_workers: int = 2,
    log_prefix: str = "    ",
) -> Dict[str, Any]:
    """Evaluate `(name, spec, kwargs)` items not yet in `cache_dict` using ProcessPoolExecutor & checkpoint each."""
    todo = [(k, sp, kw) for (k, sp, kw) in named_specs if k not in cache_dict]
    if not todo:
        return cache_dict

    if len(todo) == 1 or max_workers <= 1:
        for k, sp, kw in todo:
            _, ev = _eval_spec_worker((k, sp, kw))
            cache_dict[k] = ev
            save_checkpoint(full_results)
            print(f"{log_prefix}{k}: E_PGFT={ev['e_tot']:.6f} Ha [{ev['time_tot_s']:.2f}s]", flush=True)
        return cache_dict

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(_eval_spec_worker, item): item[0] for item in todo}
        for fut in concurrent.futures.as_completed(future_map):
            k, ev = fut.result()
            cache_dict[k] = ev
            save_checkpoint(full_results)
            print(f"{log_prefix}{k}: E_PGFT={ev['e_tot']:.6f} Ha [{ev['time_tot_s']:.2f}s]", flush=True)

    return cache_dict


def build_thc_on_shared_pivots(
    mol: gto.Mole,
    pts: np.ndarray,
    auxbasis: str | dict,
) -> THCData:
    """
    Evaluate ISDF-THC tensors `(X_ao, Z, M_thc)` for `mol` on a fixed collocation pivot set `pts`
    (in Bohr) so counterpoise dimer/ghost-monomer evaluations share identical real-space pivots
    and cancel THC quadrature/pivot differences to sub-0.01 uHa.
    """
    nao = mol.nao_nr()
    with_df = df.DF(mol, auxbasis=auxbasis)
    with_df.build()

    X_ao = np.ascontiguousarray(mol.eval_gto("GTOval_sph", pts).T)  # (nao, nk)
    nk = X_ao.shape[1]

    XtX = X_ao.T @ X_ao
    AtA = XtX * XtX

    iu, iv = np.tril_indices(nao)
    mult_tril = np.where(iu == iv, 1.0, 2.0)
    B_tril = np.vstack([blk for blk in with_df.loop()])
    pair_norms = np.sum(B_tril * B_tril, axis=0)
    sig = pair_norms > 1e-18

    A_sig = X_ao[iu[sig], :] * X_ao[iv[sig], :]
    B_sig = B_tril[:, sig] * mult_tril[sig][None, :]
    AtB = A_sig.T @ B_sig.T

    d_scale = np.sqrt(np.maximum(np.diag(AtA), 1e-30))
    AtA_norm = AtA / np.outer(d_scale, d_scale) + 1e-13 * np.eye(nk)
    Z_norm = np.linalg.solve(AtA_norm, AtB / d_scale[:, None])
    Z = np.ascontiguousarray((Z_norm / d_scale[:, None]).T)

    M_thc = Z.T @ Z
    M_thc = 0.5 * (M_thc + M_thc.T)
    return THCData(
        X_ao=X_ao,
        Z=Z,
        M_thc=M_thc,
        pts=pts.copy(),
        auxbasis=auxbasis,
    )


# =============================================================================
# Section 1: Equilibrium Reactions (`6-31g` & `cc-pvdz`)
# =============================================================================

REACTIONS_DEF: List[Tuple[str, Dict[str, float]]] = [
    ("H2 + F2 -> 2 HF", {"HF": 2.0, "H2": -1.0, "F2": -1.0}),
    ("N2 + 3 H2 -> 2 NH3", {"NH3": 2.0, "N2": -1.0, "H2": -3.0}),
    ("CO + H2O -> CO2 + H2", {"CO2": 1.0, "H2": 1.0, "CO": -1.0, "H2O": -1.0}),
    ("CH4 + 2 H2O -> CO2 + 4 H2", {"CO2": 1.0, "H2": 4.0, "CH4": -1.0, "H2O": -2.0}),
    ("Li2 + H2 -> 2 LiH", {"LiH": 2.0, "Li2": -1.0, "H2": -1.0}),
]

RXN_SPECIES = ["H2", "F2", "HF", "N2", "NH3", "CO", "H2O", "CO2", "CH4", "Li2", "LiH"]
REF_METHODS_CLOSED = ["RHF", "R_B3LYP", "R_PBE0", "R_MP2", "R_CCSD", "R_CCSD_T"]


def run_section_1_equilibrium_reactions(ref: Dict[str, Any], full_results: Dict[str, Any]) -> Dict[str, Any]:
    print("[Section 1] Evaluating Equilibrium Reactions in 6-31g and cc-pvdz...", flush=True)
    t_sec = time.perf_counter()
    s5_thermo = ref["section_5_equilibrium_chemistry"]["thermochemistry"]

    out = full_results.setdefault("section_1_equilibrium_reactions", {})
    for basis in ["6-31g", "cc-pvdz"]:
        b_out = out.setdefault(basis, {})
        mol_evals = b_out.setdefault("molecules", {})
        mols_ref = s5_thermo[basis]["molecules"]
        rxns_ref = s5_thermo[basis]["reaction_energies"]

        print(f"  [1-{basis}] Evaluating 11 equilibrium molecules (cached={len(mol_evals)}/11)...", flush=True)
        # Sort largest AO count first so 2 workers load-balance cleanly (e.g. CO2 + CH4 first)
        species_sorted = sorted(RXN_SPECIES, key=lambda s: -int(mols_ref[s]["spec"]["nao"]))
        tasks = [(sp, mols_ref[sp]["spec"], {}) for sp in species_sorted]
        eval_named_specs_cached(tasks, mol_evals, full_results, max_workers=2, log_prefix=f"    [{basis}] ")

        for sp in RXN_SPECIES:
            mol_evals[sp]["reference_energies"] = mols_ref[sp]["energies"]

        all_methods = ["PGFT", "RHF", "R_B3LYP", "R_PBE0", "R_MP2", "R_CCSD"]
        abs_errs_ccsd_t: Dict[str, List[float]] = {m: [] for m in all_methods}
        abs_errs_fci: Dict[str, List[float]] = {m: [] for m in all_methods + ["R_CCSD_T"]}
        reactions_out: Dict[str, Any] = {}

        for rlabel, stoich in REACTIONS_DEF:
            rref = rxns_ref[rlabel]
            gold_ccsd_t = float(rref["R_CCSD_T"]["dE_kcal_mol"])
            gold_fci = float(rref["Exact_FCI"]["dE_kcal_mol"]) if "Exact_FCI" in rref else None

            pgft_kcal = sum(c * mol_evals[sp]["e_tot"] for sp, c in stoich.items()) * HARTREE_TO_KCAL
            methods_kcal: Dict[str, float] = {"PGFT": float(pgft_kcal)}
            for m in REF_METHODS_CLOSED:
                methods_kcal[m] = float(rref[m]["dE_kcal_mol"])
            if gold_fci is not None:
                methods_kcal["Exact_FCI"] = gold_fci

            err_vs_ccsd_t: Dict[str, float] = {}
            for m in all_methods:
                err = methods_kcal[m] - gold_ccsd_t
                err_vs_ccsd_t[m] = float(err)
                abs_errs_ccsd_t[m].append(abs(err))

            err_vs_fci: Optional[Dict[str, float]] = None
            if gold_fci is not None:
                err_vs_fci = {}
                for m in all_methods + ["R_CCSD_T"]:
                    ef = methods_kcal[m] - gold_fci
                    err_vs_fci[m] = float(ef)
                    abs_errs_fci[m].append(abs(ef))

            reactions_out[rlabel] = {
                "stoichiometry": stoich,
                "R_CCSD_T_kcal_mol": gold_ccsd_t,
                "Exact_FCI_kcal_mol": gold_fci,
                "dE_kcal_mol": methods_kcal,
                "error_vs_R_CCSD_T_kcal_mol": err_vs_ccsd_t,
                "error_vs_Exact_FCI_kcal_mol": err_vs_fci,
            }

        mae_ccsd_t = {m: float(np.mean(v)) for m, v in abs_errs_ccsd_t.items()}
        mae_fci = {m: float(np.mean(v)) for m, v in abs_errs_fci.items() if len(v) > 0}
        b_out["reactions"] = reactions_out
        b_out["MAE_vs_R_CCSD_T_kcal_mol"] = mae_ccsd_t
        b_out["MAE_vs_Exact_FCI_available_kcal_mol"] = mae_fci
        save_checkpoint(full_results)
        print(
            f"  -> [{basis}] 5-Reaction MAE vs CCSD(T) (kcal/mol): "
            f"PGFT={mae_ccsd_t['PGFT']:.2f}, RHF={mae_ccsd_t['RHF']:.2f}, "
            f"B3LYP={mae_ccsd_t['R_B3LYP']:.2f}, PBE0={mae_ccsd_t['R_PBE0']:.2f}, MP2={mae_ccsd_t['R_MP2']:.2f}",
            flush=True,
        )

    out["complete"] = True
    out["wall_time_s"] = float(time.perf_counter() - t_sec)
    save_checkpoint(full_results)
    return out


# =============================================================================
# Section 2: Barrier Heights (`cc-pvdz` & `6-31g`)
# =============================================================================

def run_section_2_barriers(ref: Dict[str, Any], full_results: Dict[str, Any]) -> Dict[str, Any]:
    print("[Section 2] Evaluating Reaction & Conformational Barrier Heights...", flush=True)
    t_sec = time.perf_counter()
    s5 = ref["section_5_equilibrium_chemistry"]
    bref = s5["barriers"]
    sec1 = full_results["section_1_equilibrium_reactions"]

    barriers_out = full_results.setdefault("section_2_barriers", {})
    raw_cache = barriers_out.setdefault("_species_cache", {})

    # Batch all 10 barrier-specific species (largest NAO first for optimal 2-worker balance!)
    barrier_specs = [
        ("C2H6_staggered_ccpvdz", bref["C2H6_internal_rotation"]["cc-pvdz"]["staggered"]["spec"], {}),
        ("C2H6_eclipsed_ccpvdz", bref["C2H6_internal_rotation"]["cc-pvdz"]["eclipsed"]["spec"], {}),
        ("C2H6_staggered_631g", bref["C2H6_internal_rotation"]["6-31g"]["staggered"]["spec"], {}),
        ("C2H6_eclipsed_631g", bref["C2H6_internal_rotation"]["6-31g"]["eclipsed"]["spec"], {}),
        ("NH3_ts_ccpvdz", bref["NH3_planar_inversion"]["cc-pvdz"]["ts"]["spec"], {}),
        ("H2O_ts_ccpvdz", bref["H2O_linear_inversion"]["cc-pvdz"]["ts"]["spec"], {}),
        ("NH3_ts_631g", bref["NH3_planar_inversion"]["6-31g"]["ts"]["spec"], {}),
        ("H3_ts_ccpvdz", bref["H_plus_H2_collinear_exchange_ccpvdz"]["H3_ts"]["spec"], {}),
        ("H2O_ts_631g", bref["H2O_linear_inversion"]["6-31g"]["ts"]["spec"], {}),
        ("H_atom_ccpvdz", s5["thermochemistry"]["cc-pvdz"]["atoms"]["H"]["spec"], {}),
    ]
    eval_named_specs_cached(barrier_specs, raw_cache, full_results, max_workers=2, log_prefix="    [barriers] ")

    # 1. Collinear H + H2 -> [H3]^‡ (cc-pvdz)
    h3_ref = bref["H_plus_H2_collinear_exchange_ccpvdz"]
    ev_h_cc = raw_cache["H_atom_ccpvdz"]
    ev_h2_cc = sec1["cc-pvdz"]["molecules"]["H2"]
    ev_h3_ts = raw_cache["H3_ts_ccpvdz"]

    pgft_h3_kcal = (ev_h3_ts["e_tot"] - (ev_h_cc["e_tot"] + ev_h2_cc["e_tot"])) * HARTREE_TO_KCAL
    h3_vals = {"PGFT": float(pgft_h3_kcal)}
    for k, v in h3_ref["barrier_kcal_mol"].items():
        h3_vals[k] = float(v)
    fci_h3 = float(h3_ref["barrier_kcal_mol"]["Exact_FCI"])
    ccsd_t_h3 = float(h3_ref["barrier_kcal_mol"]["CCSD_T"])
    barriers_out["H_plus_H2_collinear_ccpvdz"] = {
        "basis": "cc-pvdz",
        "gold_method": "Exact_FCI",
        "gold_kcal_mol": fci_h3,
        "CCSD_T_kcal_mol": ccsd_t_h3,
        "barriers_kcal_mol": h3_vals,
        "errors_vs_Exact_FCI_kcal_mol": {k: float(v - fci_h3) for k, v in h3_vals.items()},
        "errors_vs_CCSD_T_kcal_mol": {k: float(v - ccsd_t_h3) for k, v in h3_vals.items()},
        "species_pgft": {"H": ev_h_cc, "H3_ts": ev_h3_ts},
    }
    print(f"    H+H2 TS (cc-pvdz): PGFT={pgft_h3_kcal:.2f} vs Exact_FCI={fci_h3:.2f} kcal/mol", flush=True)

    # 2. NH3 planar inversion (6-31g & cc-pvdz)
    barriers_out["NH3_planar_inversion"] = {}
    for basis, tag in [("6-31g", "631g"), ("cc-pvdz", "ccpvdz")]:
        nref = bref["NH3_planar_inversion"][basis]
        ev_min = sec1[basis]["molecules"]["NH3"]
        ev_ts = raw_cache[f"NH3_ts_{tag}"]
        pgft_b = (ev_ts["e_tot"] - ev_min["e_tot"]) * HARTREE_TO_KCAL
        vals = {"PGFT": float(pgft_b), **{k: float(v) for k, v in nref["barrier_kcal_mol"].items()}}
        gold = float(nref["barrier_kcal_mol"]["R_CCSD_T"])
        barriers_out["NH3_planar_inversion"][basis] = {
            "gold_method": "R_CCSD_T",
            "gold_kcal_mol": gold,
            "barriers_kcal_mol": vals,
            "errors_vs_gold_kcal_mol": {k: float(v - gold) for k, v in vals.items()},
            "ts_pgft": ev_ts,
        }
        print(f"    NH3 inv ({basis:7s}): PGFT={pgft_b:.2f} vs CCSD(T)={gold:.2f} kcal/mol", flush=True)

    # 3. H2O linear inversion (6-31g Exact FCI = 29.06 kcal/mol & cc-pvdz)
    barriers_out["H2O_linear_inversion"] = {}
    for basis, tag in [("6-31g", "631g"), ("cc-pvdz", "ccpvdz")]:
        href = bref["H2O_linear_inversion"][basis]
        ev_min = sec1[basis]["molecules"]["H2O"]
        ev_ts = raw_cache[f"H2O_ts_{tag}"]
        pgft_b = (ev_ts["e_tot"] - ev_min["e_tot"]) * HARTREE_TO_KCAL
        vals = {"PGFT": float(pgft_b), **{k: float(v) for k, v in href["barrier_kcal_mol"].items()}}
        gold_key = "Exact_FCI" if "Exact_FCI" in vals else "R_CCSD_T"
        gold = vals[gold_key]
        ccsd_t_val = vals["R_CCSD_T"]
        barriers_out["H2O_linear_inversion"][basis] = {
            "gold_method": gold_key,
            "gold_kcal_mol": gold,
            "R_CCSD_T_kcal_mol": ccsd_t_val,
            "barriers_kcal_mol": vals,
            "errors_vs_gold_kcal_mol": {k: float(v - gold) for k, v in vals.items()},
            "errors_vs_R_CCSD_T_kcal_mol": {k: float(v - ccsd_t_val) for k, v in vals.items()},
            "ts_pgft": ev_ts,
        }
        print(f"    H2O inv ({basis:7s}): PGFT={pgft_b:.2f} vs {gold_key}={gold:.2f} kcal/mol", flush=True)

    # 4. C2H6 internal rotation barrier (6-31g & cc-pvdz)
    barriers_out["C2H6_internal_rotation"] = {}
    for basis, tag in [("6-31g", "631g"), ("cc-pvdz", "ccpvdz")]:
        cref = bref["C2H6_internal_rotation"][basis]
        ev_stag = raw_cache[f"C2H6_staggered_{tag}"]
        ev_eclip = raw_cache[f"C2H6_eclipsed_{tag}"]
        pgft_b = (ev_eclip["e_tot"] - ev_stag["e_tot"]) * HARTREE_TO_KCAL
        vals = {"PGFT": float(pgft_b), **{k: float(v) for k, v in cref["barrier_kcal_mol"].items()}}
        gold = float(cref["barrier_kcal_mol"]["R_CCSD_T"])
        barriers_out["C2H6_internal_rotation"][basis] = {
            "gold_method": "R_CCSD_T",
            "gold_kcal_mol": gold,
            "barriers_kcal_mol": vals,
            "errors_vs_gold_kcal_mol": {k: float(v - gold) for k, v in vals.items()},
            "staggered_pgft": ev_stag,
            "eclipsed_pgft": ev_eclip,
        }
        print(f"    C2H6 rot ({basis:7s}): PGFT={pgft_b:.2f} vs CCSD(T)={gold:.2f} kcal/mol", flush=True)

    family_map = {
        "PGFT": ("PGFT", "PGFT"),
        "RHF": ("ROHF", "RHF"),
        "R_B3LYP": ("B3LYP", "R_B3LYP"),
        "R_PBE0": ("PBE0", "R_PBE0"),
        "R_MP2": ("MP2", "R_MP2"),
        "R_CCSD": ("CCSD", "R_CCSD"),
    }
    mae_ccpvdz_gold: Dict[str, float] = {}
    mae_ccpvdz_ccsd_t: Dict[str, float] = {}
    for fam, (open_k, closed_k) in family_map.items():
        errs_gold = [
            abs(barriers_out["H_plus_H2_collinear_ccpvdz"]["errors_vs_Exact_FCI_kcal_mol"][open_k]),
            abs(barriers_out["NH3_planar_inversion"]["cc-pvdz"]["errors_vs_gold_kcal_mol"][closed_k]),
            abs(barriers_out["H2O_linear_inversion"]["cc-pvdz"]["errors_vs_gold_kcal_mol"][closed_k]),
            abs(barriers_out["C2H6_internal_rotation"]["cc-pvdz"]["errors_vs_gold_kcal_mol"][closed_k]),
        ]
        errs_cc = [
            abs(barriers_out["H_plus_H2_collinear_ccpvdz"]["errors_vs_CCSD_T_kcal_mol"][open_k]),
            abs(barriers_out["NH3_planar_inversion"]["cc-pvdz"]["errors_vs_gold_kcal_mol"][closed_k]),
            abs(barriers_out["H2O_linear_inversion"]["cc-pvdz"]["errors_vs_R_CCSD_T_kcal_mol"][closed_k]),
            abs(barriers_out["C2H6_internal_rotation"]["cc-pvdz"]["errors_vs_gold_kcal_mol"][closed_k]),
        ]
        mae_ccpvdz_gold[fam] = float(np.mean(errs_gold))
        mae_ccpvdz_ccsd_t[fam] = float(np.mean(errs_cc))

    barriers_out["MAE_ccpvdz_4barriers_vs_gold_kcal_mol"] = mae_ccpvdz_gold
    barriers_out["MAE_ccpvdz_4barriers_vs_CCSD_T_kcal_mol"] = mae_ccpvdz_ccsd_t
    barriers_out["complete"] = True
    barriers_out["wall_time_s"] = float(time.perf_counter() - t_sec)
    save_checkpoint(full_results)
    print(
        f"  -> [cc-pvdz] 4-Barrier MAE vs Gold (kcal/mol): "
        f"PGFT={mae_ccpvdz_gold['PGFT']:.2f}, RHF/ROHF={mae_ccpvdz_gold['RHF']:.2f}, "
        f"B3LYP={mae_ccpvdz_gold['R_B3LYP']:.2f}, PBE0={mae_ccpvdz_gold['R_PBE0']:.2f}, MP2={mae_ccpvdz_gold['R_MP2']:.2f}",
        flush=True,
    )
    return barriers_out


# =============================================================================
# Section 3: He2 Van der Waals Dispersion Curve (`aug-cc-pvdz`, Counterpoise)
# =============================================================================

def _eval_he2_point(R: float) -> Dict[str, Any]:
    he_probe = gto.M(atom="He 0 0 0", basis="aug-cc-pvdz", verbose=0)
    aux_he = df.make_auxbasis(he_probe)["He"]
    basis_cp = {"He": "aug-cc-pvdz", "X-He": gto.basis.load("aug-cc-pvdz", "He")}
    aux_cp = {"He": aux_he, "X-He": aux_he}

    def run_exact_ref_cp(mol_m: gto.Mole) -> Dict[str, float]:
        mf = scf.RHF(mol_m)
        mf.conv_tol = 1e-12
        mf.max_cycle = 150
        mf = gr.safe_scf_kernel(mf)
        m2 = mp.MP2(mf).run()
        mycc = cc.CCSD(mf)
        mycc.conv_tol = 1e-11
        mycc.conv_tol_normt = 1e-10
        mycc.kernel()
        et = mycc.ccsd_t()
        return {
            "RHF": float(mf.e_tot),
            "MP2": float(m2.e_tot),
            "CCSD": float(mycc.e_tot),
            "CCSD_T": float(mycc.e_tot + et),
        }

    mol_ab = gto.M(atom=f"He 0 0 0; He 0 0 {R:.6f}", basis=basis_cp, unit="Angstrom", verbose=0)
    mol_a = gto.M(atom=f"He 0 0 0; X-He 0 0 {R:.6f}", basis=basis_cp, unit="Angstrom", verbose=0)
    mol_b = gto.M(atom=f"X-He 0 0 0; He 0 0 {R:.6f}", basis=basis_cp, unit="Angstrom", verbose=0)

    thc_ab = build_isdf_thc(mol_ab, auxbasis=aux_cp)
    thc_a = build_thc_on_shared_pivots(mol_a, thc_ab.pts, aux_cp)
    thc_b = build_thc_on_shared_pivots(mol_b, thc_ab.pts, aux_cp)

    mf_ab = gr.safe_scf_kernel(scf.RHF(mol_ab).set(conv_tol=1e-12))
    mf_a = gr.safe_scf_kernel(scf.RHF(mol_a).set(conv_tol=1e-12))
    mf_b = gr.safe_scf_kernel(scf.RHF(mol_b).set(conv_tol=1e-12))

    res_ab = PGFTSolver(mol_ab, thc_data=thc_ab).kernel(mf_init=mf_ab)
    res_a = PGFTSolver(mol_a, thc_data=thc_a).kernel(mf_init=mf_a)
    res_b = PGFTSolver(mol_b, thc_data=thc_b).kernel(mf_init=mf_b)

    de_pgft_uha = float((res_ab.e_tot - res_a.e_tot - res_b.e_tot) * 1e6)
    de_stat_uha = float((res_ab.e_stat - res_a.e_stat - res_b.e_stat) * 1e6)
    de_dyn_uha = float((res_ab.e_dyn - res_a.e_dyn - res_b.e_dyn) * 1e6)
    asym_ab_uha = float((res_a.e_tot - res_b.e_tot) * 1e6)

    ref_ab = run_exact_ref_cp(mol_ab)
    ref_a = run_exact_ref_cp(mol_a)
    ref_b = run_exact_ref_cp(mol_b)

    interaction_uha = {
        "PGFT": de_pgft_uha,
        "PGFT_stat": de_stat_uha,
        "PGFT_dyn": de_dyn_uha,
        "RHF": float((ref_ab["RHF"] - ref_a["RHF"] - ref_b["RHF"]) * 1e6),
        "MP2": float((ref_ab["MP2"] - ref_a["MP2"] - ref_b["MP2"]) * 1e6),
        "CCSD": float((ref_ab["CCSD"] - ref_a["CCSD"] - ref_b["CCSD"]) * 1e6),
        "CCSD_T": float((ref_ab["CCSD_T"] - ref_a["CCSD_T"] - ref_b["CCSD_T"]) * 1e6),
    }

    return {
        "R_angstrom": float(R),
        "n_thc": int(res_ab.n_thc),
        "n_aux": int(res_ab.n_aux),
        "interaction_energy_uHa": interaction_uha,
        "monomer_ghost_symmetry_diff_uHa": asym_ab_uha,
        "energies_ha": {
            "PGFT": {"AB": float(res_ab.e_tot), "A_ghostB": float(res_a.e_tot), "ghostA_B": float(res_b.e_tot)},
            "CCSD_T": {"AB": ref_ab["CCSD_T"], "A_ghostB": ref_a["CCSD_T"], "ghostA_B": ref_b["CCSD_T"]},
        },
    }


def run_section_3_he2_dispersion(full_results: Dict[str, Any]) -> Dict[str, Any]:
    print("[Section 3] Evaluating Counterpoise He2 Van der Waals Curve (aug-cc-pvdz)...", flush=True)
    t_sec = time.perf_counter()

    r_grid = [2.6, 3.0, 3.5, 4.0, 6.0]
    sec3 = full_results.setdefault("section_3_he2_dispersion", {})
    pts_by_r = {float(p["R_angstrom"]): p for p in sec3.get("points", [])}
    todo_r = [r for r in r_grid if r not in pts_by_r]

    if todo_r:
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
            futs = {pool.submit(_eval_he2_point, r): r for r in todo_r}
            for fut in concurrent.futures.as_completed(futs):
                pt = fut.result()
                pts_by_r[float(pt["R_angstrom"])] = pt
                sec3["points"] = [pts_by_r[r] for r in r_grid if r in pts_by_r]
                save_checkpoint(full_results)
                iu = pt["interaction_energy_uHa"]
                print(
                    f"    R={pt['R_angstrom']:.1f} A | PGFT={iu['PGFT']:+7.2f} uHa, "
                    f"CCSD(T)={iu['CCSD_T']:+7.2f} uHa, MP2={iu['MP2']:+7.2f} uHa, "
                    f"RHF={iu['RHF']:+7.2f} uHa (|E_A(B)-E_(A)B|={abs(pt['monomer_ghost_symmetry_diff_uHa']):.4f} uHa)",
                    flush=True,
                )

    points = [pts_by_r[r] for r in r_grid]
    sec3.update({
        "basis": "aug-cc-pvdz",
        "counterpoise": True,
        "shared_dimer_thc_pivots": True,
        "points": points,
        "max_ghost_monomer_asymmetry_uHa": float(max(abs(p["monomer_ghost_symmetry_diff_uHa"]) for p in points)),
        "complete": True,
        "wall_time_s": float(time.perf_counter() - t_sec),
    })
    save_checkpoint(full_results)
    return sec3


# =============================================================================
# Section 4: Size-Consistency, Open-Shell / Heterogeneous Additivity & SO(3)
# =============================================================================

def _format_additivity(ev_super: Dict[str, Any], ev_parts: List[Tuple[float, Dict[str, Any]]]) -> Dict[str, Any]:
    e_tot_sum = sum(w * p["e_tot"] for w, p in ev_parts)
    e_stat_sum = sum(w * p["e_stat"] for w, p in ev_parts)
    e_dyn_sum = sum(w * p["e_dyn"] for w, p in ev_parts)
    de_tot = ev_super["e_tot"] - e_tot_sum
    de_stat = ev_super["e_stat"] - e_stat_sum
    de_dyn = ev_super["e_dyn"] - e_dyn_sum
    return {
        "e_supersystem_ha": float(ev_super["e_tot"]),
        "e_monomer_sum_ha": float(e_tot_sum),
        "dE_tot_ha": float(de_tot),
        "dE_tot_mHa": float(de_tot * 1e3),
        "dE_tot_uHa": float(de_tot * 1e6),
        "dE_tot_kcal_mol": float(de_tot * HARTREE_TO_KCAL),
        "dE_stat_uHa": float(de_stat * 1e6),
        "dE_dyn_uHa": float(de_dyn * 1e6),
        "supersystem": ev_super,
    }


def run_section_4_size_consistency_and_so3(ref: Dict[str, Any], full_results: Dict[str, Any]) -> Dict[str, Any]:
    print("[Section 4] Evaluating Size-Consistency, Open-Shell Additivity & SO(3) Invariance...", flush=True)
    t_sec = time.perf_counter()
    s6 = ref["section_6_size_consistency_extensivity"]
    sec1 = full_results["section_1_equilibrium_reactions"]
    sec2 = full_results["section_2_barriers"]

    sec4 = full_results.setdefault("section_4_size_consistency_and_so3", {})
    raw_cache = sec4.setdefault("_species_cache", {})

    # Construct rotated H2O spec (Euler 37, 53, 71 deg)
    mol_h2o_0 = gr.reconstruct_mol_from_spec(
        ref["section_5_equilibrium_chemistry"]["thermochemistry"]["6-31g"]["molecules"]["H2O"]["spec"]
    )
    coords0 = mol_h2o_0.atom_coords(unit="Angstrom")
    syms = [mol_h2o_0.atom_symbol(i) for i in range(mol_h2o_0.natm)]
    a, b, c = math.radians(37.0), math.radians(53.0), math.radians(71.0)
    Rz1 = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[math.cos(b), 0.0, math.sin(b)], [0.0, 1.0, 0.0], [-math.sin(b), 0.0, math.cos(b)]])
    Rz2 = np.array([[math.cos(c), -math.sin(c), 0.0], [math.sin(c), math.cos(c), 0.0], [0.0, 0.0, 1.0]])
    R_rot = Rz1 @ Ry @ Rz2
    coords_rot = coords0 @ R_rot.T
    atom_rot = "; ".join(f"{s} {xyz[0]:.12f} {xyz[1]:.12f} {xyz[2]:.12f}" for s, xyz in zip(syms, coords_rot))

    sc_specs = [
        ("Test_B_super", s6["Test_B_stretched_H2_dimer_100A_ccpvdz"]["supersystem"]["spec"], {}),
        ("Test_C_super", s6["Test_C_stretched_H2_plus_He_100A_ccpvdz"]["supersystem"]["spec"], {}),
        ("Test_2B_super", gr.mol_spec(gr.build_mol("H 0 0 0; H 0 0 3.2; H 100.0 0 0", basis="cc-pvdz", spin=1)), {}),
        ("Test_E1_super", gr.mol_spec(gr.build_mol("H 0 0 0; H 0 0 3.2; Li 100.0 0 0; H 100.0 0 1.5949", basis="6-31g")), {}),
        ("H2O_rot_631g", gr.mol_spec(gr.build_mol(atom_rot, basis="6-31g")), {}),
        ("Test_A_super", s6["Test_A_He_He_100A_ccpvdz"]["supersystem"]["spec"], {}),
        ("H2_3p2A_ccpvdz", s6["monomers_ccpvdz"]["H2_R3p2A"]["spec"], {}),
        ("He_ccpvdz", s6["monomers_ccpvdz"]["He"]["spec"], {}),
        ("H2_3p2A_631g", gr.mol_spec(gr.build_mol("H 0 0 0; H 0 0 3.2", basis="6-31g")), {}),
    ]
    eval_named_specs_cached(sc_specs, raw_cache, full_results, max_workers=2, log_prefix="    [sc] ")

    ev_he = raw_cache["He_ccpvdz"]
    ev_h2_32_cc = raw_cache["H2_3p2A_ccpvdz"]
    ev_h2_32_631 = raw_cache["H2_3p2A_631g"]
    ev_h_cc = sec2["H_plus_H2_collinear_ccpvdz"]["species_pgft"]["H"]
    ev_lih_631 = sec1["6-31g"]["molecules"]["LiH"]
    ev_h2o_0 = sec1["6-31g"]["molecules"]["H2O"]
    ev_h2o_rot = raw_cache["H2O_rot_631g"]

    test_a = _format_additivity(raw_cache["Test_A_super"], [(2.0, ev_he)])
    test_a["reference_CISD_error_mHa"] = float(s6["Test_A_He_He_100A_ccpvdz"]["size_consistency_error_ha"]["R_CISD"] * 1e3)

    test_b = _format_additivity(raw_cache["Test_B_super"], [(2.0, ev_h2_32_cc)])
    test_b["reference_CISD_error_mHa"] = float(
        s6["Test_B_stretched_H2_dimer_100A_ccpvdz"]["size_consistency_error_ha"]["R_CISD"] * 1e3
    )

    test_c = _format_additivity(raw_cache["Test_C_super"], [(1.0, ev_h2_32_cc), (1.0, ev_he)])
    test_c["reference_CISD_error_mHa"] = float(
        s6["Test_C_stretched_H2_plus_He_100A_ccpvdz"]["size_consistency_error_ha"]["R_CISD"] * 1e3
    )

    test_2b = _format_additivity(raw_cache["Test_2B_super"], [(1.0, ev_h2_32_cc), (1.0, ev_h_cc)])
    fci_h2_32 = [
        p["energies"]["Exact_FCI"]
        for p in ref["section_2_two_electron_correlation"]["H2_dissociation_curve"]
        if p["R_angstrom"] == 3.2
    ][0]
    fci_h = ref["section_1_one_electron_sie"]["H_atom"]["energies"]["Exact_FCI"]
    test_2b["Exact_FCI_sum_ha"] = float(fci_h2_32 + fci_h)
    test_2b["error_vs_Exact_FCI_kcal_mol"] = float(
        (raw_cache["Test_2B_super"]["e_tot"] - (fci_h2_32 + fci_h)) * HARTREE_TO_KCAL
    )

    test_e1 = _format_additivity(raw_cache["Test_E1_super"], [(1.0, ev_h2_32_631), (1.0, ev_lih_631)])
    so3_diff_uha = float((ev_h2o_rot["e_tot"] - ev_h2o_0["e_tot"]) * 1e6)

    for label, res_d in [
        ("Test A: He...He (100 A, cc-pvdz)", test_a),
        ("Test B: 2x H2(3.2 A) (100 A, cc-pvdz)", test_b),
        ("Test C: H2(3.2 A)...He (100 A, cc-pvdz)", test_c),
        ("Test 2B: H2(3.2 A)...H (100 A, cc-pvdz, S=1/2)", test_2b),
        ("Test E1: H2(3.2 A)...LiH(eq) (100 A, 6-31g)", test_e1),
    ]:
        print(
            f"    {label:46s}: dE = {res_d['dE_tot_uHa']:+8.3f} uHa "
            f"({res_d['dE_tot_kcal_mol']:+8.5f} kcal/mol)",
            flush=True,
        )
    print(f"    {'SO(3) H2O (37, 53, 71 deg, 6-31g)':46s}: dE = {so3_diff_uha:+8.4f} uHa", flush=True)

    sec4.update({
        "monomers": {
            "He_ccpvdz": ev_he,
            "H2_3p2A_ccpvdz": ev_h2_32_cc,
            "H2_3p2A_631g": ev_h2_32_631,
        },
        "Test_A_He_He_100A_ccpvdz": test_a,
        "Test_B_2x_H2_3p2A_100A_ccpvdz": test_b,
        "Test_C_H2_3p2A_plus_He_100A_ccpvdz": test_c,
        "Test_2B_H2_3p2A_plus_H_100A_ccpvdz": test_2b,
        "Test_E1_H2_3p2A_plus_LiH_eq_100A_631g": test_e1,
        "SO3_rigid_rotation_H2O_631g": {
            "euler_angles_deg": [37.0, 53.0, 71.0],
            "e_unrotated_ha": float(ev_h2o_0["e_tot"]),
            "e_rotated_ha": float(ev_h2o_rot["e_tot"]),
            "dE_tot_ha": float(ev_h2o_rot["e_tot"] - ev_h2o_0["e_tot"]),
            "dE_tot_uHa": so3_diff_uha,
            "dE_stat_uHa": float((ev_h2o_rot["e_stat"] - ev_h2o_0["e_stat"]) * 1e6),
            "dE_dyn_uHa": float((ev_h2o_rot["e_dyn"] - ev_h2o_0["e_dyn"]) * 1e6),
            "dE_RHF_uHa": float((ev_h2o_rot["e_hf"] - ev_h2o_0["e_hf"]) * 1e6),
            "rotated_pgft": ev_h2o_rot,
        },
        "complete": True,
        "wall_time_s": float(time.perf_counter() - t_sec),
    })
    save_checkpoint(full_results)
    return sec4


# =============================================================================
# Section 5: Empirical O(N^3) Scaling & Size-Extensivity on (H2)_k Chains
# =============================================================================

def build_h2_chain_mol(k: int, r_intra: float = 0.74, spacing: float = 3.0, basis: str = "6-31g") -> gto.Mole:
    coords = []
    total_len = (k - 1) * spacing + r_intra
    z_shift = -0.5 * total_len
    for j in range(k):
        z0 = z_shift + spacing * j
        coords.append(f"H 0.0 0.0 {z0:.8f}")
        coords.append(f"H 0.0 0.0 {z0 + r_intra:.8f}")
    return gr.build_mol("; ".join(coords), basis=basis, symmetry=False)


def _fit_power_law(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2)) + 1e-30
    r2 = 1.0 - ss_res / ss_tot
    return {
        "alpha": float(slope),
        "prefactor": float(np.exp(intercept)),
        "r2": float(r2),
    }


def run_section_5_scaling_and_extensivity(
    full_results: Dict[str, Any],
    checkpoint_path: str = OUT_JSON_PATH,
) -> Dict[str, Any]:
    print("[Section 5] Running Empirical O(N^3) Scaling & Extensivity on (H2)_k (k = 2..32)...", flush=True)
    t_sec = time.perf_counter()

    k_list = [2, 4, 8, 12, 16, 24, 32]
    existing_sec5 = full_results.get("section_5_scaling_and_extensivity", {})
    existing_by_k = {int(pt["k"]): pt for pt in existing_sec5.get("chain_points", [])}

    if 2 not in existing_by_k:
        _ = eval_pgft_mol(build_h2_chain_mol(2), c_thc=10.0, n_rad=20)

    chain_points: List[Dict[str, Any]] = []
    for k in k_list:
        if k in existing_by_k:
            pt = existing_by_k[k]
            chain_points.append(pt)
            print(
                f"    [cached] k={k:2d} (N_AO={pt['nao']:3d}, N_aux={pt['n_aux']:4d}) | "
                f"tot={pt['time_tot_s']:.3f}s, dyn={pt['time_dyn_s']:.3f}s | E_corr/k={pt['e_corr_per_monomer_ha']:.6f} Ha",
                flush=True,
            )
            continue

        mol_k = build_h2_chain_mol(k, r_intra=0.74, spacing=3.0, basis="6-31g")
        ev_k = eval_pgft_mol(mol_k, c_thc=10.0, n_rad=20)
        pt = {
            "k": int(k),
            "nao": int(ev_k["nao"]),
            "n_aux": int(ev_k["n_aux"]),
            "n_thc": int(ev_k["n_thc"]),
            "e_tot_ha": float(ev_k["e_tot"]),
            "e_stat_ha": float(ev_k["e_stat"]),
            "e_dyn_ha": float(ev_k["e_dyn"]),
            "e_hf_ha": float(ev_k["e_hf"]),
            "e_corr_ha": float(ev_k["e_corr"]),
            "e_corr_per_monomer_ha": float(ev_k["e_corr"] / k),
            "e_stat_corr_per_monomer_ha": float((ev_k["e_stat"] - ev_k["e_hf"]) / k),
            "e_dyn_per_monomer_ha": float(ev_k["e_dyn"] / k),
            "e_tot_per_monomer_ha": float(ev_k["e_tot"] / k),
            "time_thc_s": float(ev_k["time_thc_s"]),
            "time_stat_s": float(ev_k["time_stat_s"]),
            "time_dyn_s": float(ev_k["time_dyn_s"]),
            "time_tot_s": float(ev_k["time_tot_s"]),
        }
        chain_points.append(pt)
        print(
            f"    k={k:2d} (N_AO={pt['nao']:3d}, N_aux={pt['n_aux']:4d}, N_THC={pt['n_thc']:3d}) | "
            f"thc={pt['time_thc_s']:.3f}s stat={pt['time_stat_s']:.3f}s dyn={pt['time_dyn_s']:.3f}s "
            f"tot={pt['time_tot_s']:.3f}s | E_corr/k={pt['e_corr_per_monomer_ha']:.6f} Ha",
            flush=True,
        )

        full_results["section_5_scaling_and_extensivity"] = {
            "basis": "6-31g",
            "r_intra_angstrom": 0.74,
            "spacing_angstrom": 3.0,
            "c_thc": 10.0,
            "n_rad": 20,
            "chain_points": chain_points,
        }
        save_checkpoint(full_results, checkpoint_path)

    asym_pts = [p for p in chain_points if p["k"] >= 8]
    nao_asym = np.array([p["nao"] for p in asym_pts], dtype=float)

    fits_k_ge_8 = {
        "dyn": _fit_power_law(nao_asym, np.array([p["time_dyn_s"] for p in asym_pts])),
        "thc": _fit_power_law(nao_asym, np.array([p["time_thc_s"] for p in asym_pts])),
        "stat": _fit_power_law(nao_asym, np.array([p["time_stat_s"] for p in asym_pts])),
        "tot": _fit_power_law(nao_asym, np.array([p["time_tot_s"] for p in asym_pts])),
    }

    ecorr_k_ge_8 = [p["e_corr_per_monomer_ha"] for p in asym_pts]
    ecorr_drift_k16_to_32_uha = float(
        (chain_points[-1]["e_corr_per_monomer_ha"] - chain_points[-3]["e_corr_per_monomer_ha"]) * 1e6
    )
    ecorr_drift_k24_to_32_uha = float(
        (chain_points[-1]["e_corr_per_monomer_ha"] - chain_points[-2]["e_corr_per_monomer_ha"]) * 1e6
    )

    print(
        f"  -> Scaling exponents (k >= 8, N_AO = 32..128): "
        f"alpha_dyn = {fits_k_ge_8['dyn']['alpha']:.2f} (R^2={fits_k_ge_8['dyn']['r2']:.4f}), "
        f"alpha_tot = {fits_k_ge_8['tot']['alpha']:.2f} (R^2={fits_k_ge_8['tot']['r2']:.4f}), "
        f"alpha_thc = {fits_k_ge_8['thc']['alpha']:.2f}, alpha_stat = {fits_k_ge_8['stat']['alpha']:.2f}",
        flush=True,
    )
    print(
        f"  -> Extensivity E_corr/k: k=16..32 drift = {ecorr_drift_k16_to_32_uha:+.2f} uHa, "
        f"k=24..32 drift = {ecorr_drift_k24_to_32_uha:+.2f} uHa",
        flush=True,
    )

    out_s5 = {
        "basis": "6-31g",
        "r_intra_angstrom": 0.74,
        "spacing_angstrom": 3.0,
        "c_thc": 10.0,
        "n_rad": 20,
        "chain_points": chain_points,
        "power_law_fits_k_ge_8": fits_k_ge_8,
        "extensivity_summary": {
            "e_corr_per_monomer_k8_ha": float(ecorr_k_ge_8[0]),
            "e_corr_per_monomer_k32_ha": float(chain_points[-1]["e_corr_per_monomer_ha"]),
            "drift_k16_to_k32_uHa": ecorr_drift_k16_to_32_uha,
            "drift_k24_to_k32_uHa": ecorr_drift_k24_to_32_uha,
            "std_k12_to_k32_uHa": float(
                np.std([p["e_corr_per_monomer_ha"] for p in chain_points if p["k"] >= 12]) * 1e6
            ),
        },
        "complete": True,
        "wall_time_s": float(time.perf_counter() - t_sec),
    }
    full_results["section_5_scaling_and_extensivity"] = out_s5
    save_checkpoint(full_results, checkpoint_path)
    return out_s5


# =============================================================================
# Figure 3 Generation (300 DPI, 1x3 Publication Panel)
# =============================================================================

def generate_figure_3(results: Dict[str, Any], out_path: str = FIG3_PATH) -> None:
    print(f"[Figure 3] Generating 1x3 publication figure -> {out_path}...", flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "legend.fontsize": 8.2,
        "axes.linewidth": 0.9,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.1), dpi=300)
    fig.subplots_adjust(left=0.052, right=0.985, bottom=0.17, top=0.88, wspace=0.26)

    colors = {
        "PGFT": "#D55E00",
        "RHF": "#7F7F7F",
        "R_B3LYP": "#009E73",
        "R_PBE0": "#56B4E9",
        "R_MP2": "#CC79A7",
        "CCSD_T": "#111111",
    }

    # Panel (a): cc-pVDZ Reaction MAE & Barrier Heights
    ax_a = axes[0]
    sec1_cc = results["section_1_equilibrium_reactions"]["cc-pvdz"]
    sec2 = results["section_2_barriers"]

    categories = [
        "5-Rxn\nMAE",
        "4-Barrier\nMAE",
        r"$\mathrm{H+H_2}$ TS" + "\n|Error|",
        r"$\mathrm{H_2O}$ Inv." + "\n|Error|",
        r"$\mathrm{NH_3}$ Inv." + "\n|Error|",
    ]
    method_specs = [
        ("PGFT", "PGFT", "PGFT", "PGFT"),
        ("RHF", "RHF/ROHF", "ROHF", "RHF"),
        ("R_B3LYP", "B3LYP", "B3LYP", "R_B3LYP"),
        ("R_PBE0", "PBE0", "PBE0", "R_PBE0"),
        ("R_MP2", "MP2", "MP2", "R_MP2"),
    ]

    x_cat = np.arange(len(categories), dtype=float)
    width = 0.155
    offsets = (np.arange(len(method_specs)) - 2.0) * width

    max_bar_val = 0.0
    for idx, (m_key, m_label, open_k, closed_k) in enumerate(method_specs):
        vals = [
            sec1_cc["MAE_vs_R_CCSD_T_kcal_mol"][m_key],
            sec2["MAE_ccpvdz_4barriers_vs_gold_kcal_mol"][m_key],
            abs(sec2["H_plus_H2_collinear_ccpvdz"]["errors_vs_Exact_FCI_kcal_mol"][open_k]),
            abs(sec2["H2O_linear_inversion"]["cc-pvdz"]["errors_vs_gold_kcal_mol"][closed_k]),
            abs(sec2["NH3_planar_inversion"]["cc-pvdz"]["errors_vs_gold_kcal_mol"][closed_k]),
        ]
        max_bar_val = max(max_bar_val, max(vals))
        edge_lw = 1.0 if m_key == "PGFT" else 0.4
        edge_col = "#662200" if m_key == "PGFT" else "#333333"
        bars = ax_a.bar(
            x_cat + offsets[idx],
            vals,
            width=width * 0.92,
            label=m_label,
            color=colors[m_key],
            edgecolor=edge_col,
            linewidth=edge_lw,
            zorder=3,
        )
        if m_key == "PGFT":
            for b, v in zip(bars, vals):
                ax_a.text(
                    b.get_x() + 0.5 * b.get_width(),
                    v + 0.35,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.3,
                    fontweight="bold",
                    color=colors["PGFT"],
                )

    ax_a.set_xticks(x_cat)
    ax_a.set_xticklabels(categories)
    ax_a.set_xlim(-0.56, len(categories) - 0.44)
    ax_a.set_ylabel("Error vs CCSD(T) / FCI (kcal/mol)")
    ax_a.set_title("(a) cc-pVDZ Thermochemistry & Barrier Errors", fontweight="bold", pad=8)
    ax_a.set_ylim(0.0, min(28.0, max_bar_val * 1.18))
    ax_a.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax_a.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.92)

    # Panel (b): He2 Van der Waals Dispersion Curve (aug-cc-pVDZ, Counterpoise)
    ax_b = axes[1]
    he2_pts = results["section_3_he2_dispersion"]["points"]
    r_vals = np.array([p["R_angstrom"] for p in he2_pts], dtype=float)
    y_cc = np.array([p["interaction_energy_uHa"]["CCSD_T"] for p in he2_pts], dtype=float)
    y_mp2 = np.array([p["interaction_energy_uHa"]["MP2"] for p in he2_pts], dtype=float)
    y_pgft = np.array([p["interaction_energy_uHa"]["PGFT"] for p in he2_pts], dtype=float)
    y_rhf = np.array([p["interaction_energy_uHa"]["RHF"] for p in he2_pts], dtype=float)

    ax_b.axhline(0.0, color="#444444", linestyle=":", linewidth=1.0, zorder=1)
    ax_b.plot(r_vals, y_cc, "o-", color=colors["CCSD_T"], lw=2.0, ms=5.8, label="CCSD(T) (CP)", zorder=4)
    ax_b.plot(r_vals, y_mp2, "s--", color=colors["R_MP2"], lw=1.6, ms=5.0, label="MP2 (CP)", zorder=3)
    ax_b.plot(r_vals, y_pgft, "D-", color=colors["PGFT"], lw=2.2, ms=5.8, label="PGFT (shared THC)", zorder=5)
    ax_b.plot(r_vals, y_rhf, "^-.", color=colors["RHF"], lw=1.5, ms=5.0, label="RHF (CP, repulsive)", zorder=2)

    y_min_plot = min(float(np.min(y_cc)), float(np.min(y_pgft))) - 4.0
    ax_b.set_ylim(y_min_plot, 34.0)
    ax_b.set_xlim(2.5, 6.15)
    ax_b.annotate(
        f"R = 2.6 Å repulsive wall:\nPGFT +{y_pgft[0]:.0f}, CCSD(T) +{y_cc[0]:.0f} μHa",
        xy=(2.62, 33.2),
        xytext=(3.18, 24.5),
        arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 0.9},
        fontsize=7.8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "#FAFAFA", "ec": "#CCCCCC", "alpha": 0.92},
    )

    ax_b.set_xlabel(r"Internuclear Separation $R(\mathrm{He}-\mathrm{He})$ (Å)")
    ax_b.set_ylabel(r"Counterpoise $\Delta E(R)$ ($\mu\mathrm{Ha}$)")
    ax_b.set_title(r"(b) $\mathrm{He_2}$ Dispersion Curve (aug-cc-pVDZ)", fontweight="bold", pad=8)
    ax_b.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax_b.legend(loc="lower right", frameon=True, framealpha=0.92)

    # Panel (c): Log-Log Wall-Time Scaling & Size-Extensivity Inset
    ax_c = axes[2]
    sec5 = results["section_5_scaling_and_extensivity"]
    cpts = sec5["chain_points"]
    fits = sec5["power_law_fits_k_ge_8"]

    nao_arr = np.array([p["nao"] for p in cpts], dtype=float)
    k_arr = np.array([p["k"] for p in cpts], dtype=float)
    t_tot = np.array([p["time_tot_s"] for p in cpts], dtype=float)
    t_stat = np.array([p["time_stat_s"] for p in cpts], dtype=float)
    t_dyn = np.array([p["time_dyn_s"] for p in cpts], dtype=float)
    t_thc = np.array([p["time_thc_s"] for p in cpts], dtype=float)
    ecorr_k = np.array([p["e_corr_per_monomer_ha"] * 1e3 for p in cpts], dtype=float)

    a_tot = fits["tot"]["alpha"]
    a_dyn = fits["dyn"]["alpha"]
    a_thc = fits["thc"]["alpha"]

    ax_c.loglog(nao_arr, t_tot, "o-", color=colors["PGFT"], lw=2.2, ms=5.8,
                label=rf"Total ($\alpha={a_tot:.2f}$)")
    ax_c.loglog(nao_arr, t_stat, "s--", color="#0072B2", lw=1.6, ms=4.8,
                label=rf"Static SCF ($\alpha={fits['stat']['alpha']:.2f}$)")
    ax_c.loglog(nao_arr, t_dyn, "D-", color=colors["R_B3LYP"], lw=1.8, ms=4.8,
                label=rf"Dynamic ACFD ($\alpha={a_dyn:.2f}$)")
    ax_c.loglog(nao_arr, t_thc, "^:", color=colors["R_MP2"], lw=1.5, ms=4.8,
                label=rf"ISDF-THC ($\alpha={a_thc:.2f}$)")

    ref_x = np.geomspace(24.0, 136.0, 50)
    idx_ref = [i for i, p in enumerate(cpts) if p["k"] == 16][0]
    ref_y = (t_tot[idx_ref] * 1.45) * ((ref_x / nao_arr[idx_ref]) ** 3)
    ax_c.loglog(ref_x, ref_y, "--", color="#222222", lw=1.3, label=r"Exact $\mathcal{O}(N_{\mathrm{AO}}^3)$ ref.")

    ax_c.set_ylim(1.2e-2, 2.4e3)
    ax_c.set_xticks([8, 16, 32, 64, 128])
    ax_c.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_c.set_xlabel(r"Basis Dimension $N_{\mathrm{AO}}$ along $(\mathrm{H_2})_k$ ($k = 2\dots 32$)")
    ax_c.set_ylabel("Wall Time (s)")
    ax_c.set_title(r"(c) Empirical $\mathcal{O}(N^3)$ Scaling & Extensivity", fontweight="bold", pad=8)
    ax_c.grid(True, which="both", linestyle="--", alpha=0.30, zorder=0)
    ax_c.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=7.6)

    ax_ins = ax_c.inset_axes([0.52, 0.10, 0.45, 0.25])
    ax_ins.patch.set_alpha(0.95)
    ecorr_asym = ecorr_k[-1]
    dev_uha = (ecorr_k - ecorr_asym) * 1e3
    ax_ins.axhline(0.0, color="#444444", ls=":", lw=0.8)
    ax_ins.plot(k_arr[2:], dev_uha[2:], "o-", color=colors["PGFT"], lw=1.5, ms=3.8)
    ax_ins.set_title(r"$\Delta(E_{\mathrm{corr}}/k)$ ($\mu\mathrm{Ha}$, $k\geq 8$)", fontsize=7.2, pad=2.0)
    ax_ins.set_xlabel(r"$k$ monomers", fontsize=7.0, labelpad=1.0)
    ax_ins.tick_params(labelsize=6.6)
    max_dev = max(5.0, float(np.max(np.abs(dev_uha[2:]))) * 1.30)
    ax_ins.set_ylim(-max_dev, max_dev)
    ax_ins.grid(True, ls="--", alpha=0.3)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  -> Saved {out_path} successfully!", flush=True)


# =============================================================================
# Main Driver
# =============================================================================

def main() -> None:
    t_start = time.perf_counter()
    rebuild = "--rebuild" in sys.argv
    plot_only = "--plot-only" in sys.argv
    skip_sec5 = "--skip-sec5" in sys.argv

    with open(REF_JSON_PATH, "r", encoding="utf-8") as f:
        ref = json.load(f)

    if (not rebuild) and os.path.exists(OUT_JSON_PATH):
        with open(OUT_JSON_PATH, "r", encoding="utf-8") as f:
            results: Dict[str, Any] = json.load(f)
    else:
        results = {
            "metadata": {
                "theory": "Pauli-Purified Gor'kov Fluctuation Theory (PGFT)",
                "reference_json": REF_JSON_PATH,
                "zero_empirical_parameters": True,
                "zero_dft_grids": True,
            }
        }

    if not plot_only:
        if rebuild or not results.get("section_1_equilibrium_reactions", {}).get("complete"):
            run_section_1_equilibrium_reactions(ref, results)
        else:
            print("[Checkpoint] Loaded section_1_equilibrium_reactions", flush=True)

        if rebuild or not results.get("section_2_barriers", {}).get("complete"):
            run_section_2_barriers(ref, results)
        else:
            print("[Checkpoint] Loaded section_2_barriers", flush=True)

        if rebuild or not results.get("section_3_he2_dispersion", {}).get("complete"):
            run_section_3_he2_dispersion(results)
        else:
            print("[Checkpoint] Loaded section_3_he2_dispersion", flush=True)

        if rebuild or not results.get("section_4_size_consistency_and_so3", {}).get("complete"):
            run_section_4_size_consistency_and_so3(ref, results)
        else:
            print("[Checkpoint] Loaded section_4_size_consistency_and_so3", flush=True)

        if not skip_sec5:
            if rebuild or not results.get("section_5_scaling_and_extensivity", {}).get("complete"):
                run_section_5_scaling_and_extensivity(results)
            else:
                print("[Checkpoint] Loaded section_5_scaling_and_extensivity", flush=True)

        results["metadata"]["last_updated_wall_time_s"] = float(time.perf_counter() - t_start)
        save_checkpoint(results)

    if not skip_sec5:
        generate_figure_3(results)
    print(f"ALL DONE in {time.perf_counter() - t_start:.2f}s!", flush=True)


if __name__ == "__main__":
    main()
