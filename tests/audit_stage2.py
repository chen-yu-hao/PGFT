#!/usr/bin/env python3
"""
Stage 2 Adversarial Comparative Audit of Proposal A (GPS-ACFD) and Proposal B (AESF)
====================================================================================
Author: Reviewer #2 (referee_two)
Outputs:
  - /home/antigravity/project/tests/stage2_audit_results.json

Executes head-to-head stress tests of GPS-ACFD (Ent & Bogo) and AESF (DF-static & pure-THC-static)
against exact FCI / CCSD(T) benchmarks in `/home/antigravity/project/tests/reference_data.json`
plus targeted Reviewer #2 mathematical trap tests.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
from pyscf import gto, scf

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROP_A_DIR = "/home/antigravity/project/proposals/theory_A"
PROP_B_DIR = "/home/antigravity/project/proposals/theory_B"
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, PROP_A_DIR)
sys.path.insert(0, PROP_B_DIR)

import gauntlet_reference as gr
from qps_acfd import GorKovPauliACFD
from aesf import run_aesf

HARTREE_TO_KCAL = gr.HARTREE_TO_KCAL
REF_JSON = os.path.join(TESTS_DIR, "reference_data.json")
OUT_JSON = os.path.join(TESTS_DIR, "stage2_audit_results.json")


def eval_all_theories(
    mol: gto.Mole,
    include_aesf_pure_thc: bool = False,
    n_seeds_a: int = 3,
) -> Dict[str, Any]:
    """
    Evaluate GPS-ACFD(Ent), GPS-ACFD(Bogo), AESF(DF-static default), and optionally
    AESF(pure-THC O(N^3) static) on the exact same PySCF Mole & HF starting orbitals.
    """
    mf = scf.RHF(mol) if mol.spin == 0 else scf.ROHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 150
    mf = gr.safe_scf_kernel(mf)

    out: Dict[str, Any] = {"RHF_or_ROHF": float(mf.e_tot)}

    # Proposal A: GPS-ACFD (Entangled & Bogo share exact same THC build)
    t0 = time.time()
    solver_a = GorKovPauliACFD(mol, kappa_mode="entangled")
    e_ent, es_ent, ed_ent, _, na_ent, nb_ent = solver_a.kernel(mf_init=mf, n_seeds=n_seeds_a)
    t_ent = time.time() - t0

    t1 = time.time()
    solver_a.kappa_mode = "bogo"
    e_bogo, es_bogo, ed_bogo, _, na_bogo, nb_bogo = solver_a.kernel(mf_init=mf, n_seeds=n_seeds_a)
    t_bogo = time.time() - t1

    out["GPS_Ent"] = {
        "e_tot": float(e_ent),
        "e_stat": float(es_ent),
        "e_dyn": float(ed_ent),
        "occ_alpha_top6": [round(float(x), 5) for x in na_ent[:6]],
        "occ_beta_top6": [round(float(x), 5) for x in nb_ent[:6]],
        "time_s": round(t_ent, 3),
    }
    out["GPS_Bogo"] = {
        "e_tot": float(e_bogo),
        "e_stat": float(es_bogo),
        "e_dyn": float(ed_bogo),
        "occ_alpha_top6": [round(float(x), 5) for x in na_bogo[:6]],
        "occ_beta_top6": [round(float(x), 5) for x in nb_bogo[:6]],
        "time_s": round(t_bogo, 3),
    }

    # Proposal B: AESF default (use_pure_thc_static=False -> O(N^4) DF static!)
    rb_df = run_aesf(mol, mf=mf, use_pure_thc_static=False)
    out["AESF_DF"] = {
        "e_tot": float(rb_df["e_tot"]),
        "e_stat": float(rb_df["e_stat"]),
        "e_dyn": float(rb_df["e_dyn"]),
        "f_occ_top6": [round(float(x), 5) for x in rb_df["f_occ"][:6]],
        "time_s": round(float(rb_df["wall_time"]), 3),
    }

    if include_aesf_pure_thc:
        rb_thc = run_aesf(mol, mf=mf, use_pure_thc_static=True)
        out["AESF_THC"] = {
            "e_tot": float(rb_thc["e_tot"]),
            "e_stat": float(rb_thc["e_stat"]),
            "e_dyn": float(rb_thc["e_dyn"]),
            "f_occ_top6": [round(float(x), 5) for x in rb_thc["f_occ"][:6]],
            "time_s": round(float(rb_thc["wall_time"]), 3),
        }

    return out


# =============================================================================
# Audit 1: THC Static Fidelity (O(N^4) DF-Static vs O(N^3) THC-Static) & SO(3) Rotation
# =============================================================================

def run_audit_1(ref: Dict[str, Any]) -> Dict[str, Any]:
    print("[Audit 1] Testing O(N^3) THC Static Fidelity & 3D Rotational Invariance...", flush=True)
    mols_631g = ref["section_5_equilibrium_chemistry"]["thermochemistry"]["6-31g"]["molecules"]
    thc_comparison = {}
    for mname in ["H2", "LiH", "H2O", "N2", "CO2"]:
        mol = gr.reconstruct_mol_from_spec(mols_631g[mname]["spec"])
        ev = eval_all_theories(mol, include_aesf_pure_thc=True)
        ev["Exact_RHF"] = mols_631g[mname]["energies"]["RHF"]
        ev["AESF_THC_minus_DF_kcal_mol"] = (
            ev["AESF_THC"]["e_tot"] - ev["AESF_DF"]["e_tot"]
        ) * HARTREE_TO_KCAL
        ev["AESF_THC_stat_minus_RHF_kcal_mol"] = (
            ev["AESF_THC"]["e_stat"] - ev["Exact_RHF"]
        ) * HARTREE_TO_KCAL
        ev["GPS_Ent_stat_minus_RHF_kcal_mol"] = (
            ev["GPS_Ent"]["e_stat"] - ev["Exact_RHF"]
        ) * HARTREE_TO_KCAL
        thc_comparison[mname] = ev

    # 3D Rigid SO(3) Rotational Invariance Test on H2O (6-31G)
    mol_h2o = gr.reconstruct_mol_from_spec(mols_631g["H2O"]["spec"])
    coords0 = mol_h2o.atom_coords(unit="Angstrom")
    syms = [mol_h2o.atom_symbol(i) for i in range(mol_h2o.natm)]

    # Orthogonal 3D Euler rotation (37 deg, 53 deg, 71 deg)
    a, b, c = math.radians(37.0), math.radians(53.0), math.radians(71.0)
    Rz1 = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz2 = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    R_rot = Rz1 @ Ry @ Rz2
    coords_rot = coords0 @ R_rot.T
    atom_rot = "; ".join(
        f"{s} {xyz[0]:.12f} {xyz[1]:.12f} {xyz[2]:.12f}" for s, xyz in zip(syms, coords_rot)
    )
    mol_h2o_rot = gr.build_mol(atom_rot, basis="6-31g")

    ev_0 = thc_comparison["H2O"]
    ev_rot = eval_all_theories(mol_h2o_rot, include_aesf_pure_thc=True)
    rot_diff_uHa = {
        k: (ev_rot[k]["e_tot"] - ev_0[k]["e_tot"]) * 1e6
        for k in ["GPS_Ent", "GPS_Bogo", "AESF_DF", "AESF_THC"]
    }
    rot_diff_uHa["RHF"] = (ev_rot["RHF_or_ROHF"] - ev_0["RHF_or_ROHF"]) * 1e6

    return {
        "thc_vs_df_static": thc_comparison,
        "so3_rotation_invariance_H2O_631g_uHa": rot_diff_uHa,
    }


# =============================================================================
# Audit 2: 1-Electron & 3-Electron Anti-Cheat Open-Shell Torture (cc-pVDZ)
# =============================================================================

def run_audit_2(ref: Dict[str, Any]) -> Dict[str, Any]:
    print("[Audit 2] Running 1e & 3e Anti-Cheat Open-Shell Torture Tests (cc-pVDZ)...", flush=True)
    s1 = ref["section_1_one_electron_sie"]

    # 1e H atom and H2+ (R=1.06, 4.0, 6.0 A)
    mol_h = gr.reconstruct_mol_from_spec(s1["H_atom"]["spec"])
    ev_h = eval_all_theories(mol_h)
    ev_h["Exact_FCI"] = s1["H_atom"]["energies"]["Exact_FCI"]

    h2p_out = {}
    for pt in s1["H2_plus_curve"]:
        r = pt["R_angstrom"]
        if r in [1.06, 4.0, 6.0]:
            mol_p = gr.reconstruct_mol_from_spec(pt["spec"])
            ev_p = eval_all_theories(mol_p)
            ev_p["Exact_FCI"] = pt["energies"]["Exact_FCI"]
            h2p_out[f"R_{r}"] = ev_p

    # Critical 3e Test 2A: H2+(4.0 A) ... He(100 A) (charge=+1, spin=1, nelec=3)
    ac = s1["anti_cheat_H2plus_4A_plus_He_100A"]
    mol_super_2a = gr.reconstruct_mol_from_spec(ac["supersystem"]["spec"])
    ev_super_2a = eval_all_theories(mol_super_2a)

    mol_he = gr.build_mol("He 0 0 0", basis="cc-pvdz")
    ev_he = eval_all_theories(mol_he)
    ev_h2p_4 = h2p_out["R_4.0"]

    err_2a_ha = {
        k: ev_super_2a[k]["e_tot"] - (ev_h2p_4[k]["e_tot"] + ev_he[k]["e_tot"])
        for k in ["GPS_Ent", "GPS_Bogo", "AESF_DF"]
    }
    err_2a_vs_fci_kcal = {
        k: (ev_super_2a[k]["e_tot"] - ac["supersystem"]["energies"]["Exact_FCI"]) * HARTREE_TO_KCAL
        for k in ["GPS_Ent", "GPS_Bogo", "AESF_DF"]
    }

    # Critical 3e Test 2B: Stretched H2(3.2 A) ... H(100 A) (charge=0, spin=1, nelec=3)
    # Exposes whether spin>0 disables multireference static entanglement on H2!
    mol_h2_32 = gr.build_mol("H 0 0 0; H 0 0 3.2", basis="cc-pvdz", charge=0, spin=0)
    ev_h2_32 = eval_all_theories(mol_h2_32)
    fci_h2_32 = [
        p["energies"]["Exact_FCI"]
        for p in ref["section_2_two_electron_correlation"]["H2_dissociation_curve"]
        if p["R_angstrom"] == 3.2
    ][0]

    mol_super_2b = gr.build_mol(
        "H 0 0 0; H 0 0 3.2; H 100.0 0 0", basis="cc-pvdz", charge=0, spin=1
    )
    ev_super_2b = eval_all_theories(mol_super_2b)
    fci_super_2b = fci_h2_32 + ev_h["Exact_FCI"]

    err_2b_additivity_kcal = {
        k: (ev_super_2b[k]["e_tot"] - (ev_h2_32[k]["e_tot"] + ev_h[k]["e_tot"])) * HARTREE_TO_KCAL
        for k in ["GPS_Ent", "GPS_Bogo", "AESF_DF"]
    }
    err_2b_vs_fci_kcal = {
        k: (ev_super_2b[k]["e_tot"] - fci_super_2b) * HARTREE_TO_KCAL
        for k in ["GPS_Ent", "GPS_Bogo", "AESF_DF"]
    }

    return {
        "H_atom": ev_h,
        "H2_plus_selected": h2p_out,
        "Test_2A_H2plus_4A_He_100A": {
            "supersystem": ev_super_2a,
            "Exact_FCI": ac["supersystem"]["energies"]["Exact_FCI"],
            "additivity_error_ha": err_2a_ha,
            "error_vs_fci_kcal_mol": err_2a_vs_fci_kcal,
        },
        "Test_2B_H2_3p2A_plus_H_100A": {
            "supersystem": ev_super_2b,
            "H2_3p2A_monomer": ev_h2_32,
            "Exact_FCI_sum": fci_super_2b,
            "additivity_error_kcal_mol": err_2b_additivity_kcal,
            "error_vs_fci_kcal_mol": err_2b_vs_fci_kcal,
        },
    }


# =============================================================================
# Audit 3: Multibond, Heteronuclear, & Diradical Curves (HF, H2O, N2, H4)
# =============================================================================

def run_audit_3(ref: Dict[str, Any]) -> Dict[str, Any]:
    print("[Audit 3] Running HF, H2O, N2, and H4 Multireference Curves...", flush=True)
    s4 = ref["section_4_multibond_breaking"]
    atoms_631g = ref["section_5_equilibrium_chemistry"]["thermochemistry"]["6-31g"]["atoms"]

    # Isolated open-shell atoms in 6-31G & STO-3G for asymptotic consistency checks
    ev_h_631g = eval_all_theories(gr.reconstruct_mol_from_spec(atoms_631g["H"]["spec"]))
    ev_f_631g = eval_all_theories(gr.reconstruct_mol_from_spec(atoms_631g["F"]["spec"]))
    ev_o_631g = eval_all_theories(gr.reconstruct_mol_from_spec(atoms_631g["O"]["spec"]))
    ev_n_631g = eval_all_theories(gr.reconstruct_mol_from_spec(atoms_631g["N"]["spec"]))
    ev_n_sto3g = eval_all_theories(gr.build_mol("N 0 0 0", basis="sto-3g", spin=3))

    # 3a. HF curve (6-31G Exact FCI) + R=8.0 A asymptotic point
    print("  [3a] HF dissociation curve (6-31G)...", flush=True)
    hf_pts = []
    for pt in s4["HF_stretch_631g"]["points"]:
        mol = gr.reconstruct_mol_from_spec(pt["spec"])
        ev = eval_all_theories(mol)
        ev["R_angstrom"] = pt["R_angstrom"]
        ev["Exact_FCI"] = pt["energies"]["Exact_FCI"]
        ev["R_B3LYP"] = pt["energies"]["R_B3LYP"]
        ev["R_CCSD_T"] = pt["energies"]["R_CCSD_T"]
        ev["errors_vs_fci_kcal_mol"] = {
            k: (ev[k]["e_tot"] - pt["energies"]["Exact_FCI"]) * HARTREE_TO_KCAL
            for k in ["GPS_Ent", "GPS_Bogo", "AESF_DF"]
        }
        hf_pts.append(ev)
    ev_hf_8a = eval_all_theories(gr.build_mol("F 0 0 0; H 0 0 8.0", basis="6-31g"))

    # 3b. H2O symmetric double-bond stretch (6-31G Exact FCI)
    print("  [3b] H2O symmetric stretch (6-31G)...", flush=True)
    h2o_pts = []
    for pt in s4["H2O_symmetric_stretch_631g"]["points"]:
        mol = gr.reconstruct_mol_from_spec(pt["spec"])
        ev = eval_all_theories(mol)
        ev["R_OH_angstrom"] = pt["R_OH_angstrom"]
        ev["Exact_FCI"] = pt["energies"]["Exact_FCI"]
        ev["R_B3LYP"] = pt["energies"]["R_B3LYP"]
        ev["R_CCSD_T"] = pt["energies"]["R_CCSD_T"]
        ev["errors_vs_fci_kcal_mol"] = {
            k: (ev[k]["e_tot"] - pt["energies"]["Exact_FCI"]) * HARTREE_TO_KCAL
            for k in ["GPS_Ent", "GPS_Bogo", "AESF_DF"]
        }
        h2o_pts.append(ev)

    # 3c. N2 triple-bond stretch (STO-3G Exact FCI & 6-31G NEVPT2 + 2N(^4S) limit)
    print("  [3c] N2 triple-bond stretch (STO-3G & 6-31G)...", flush=True)
    n2_sto_pts = []
    for pt in s4["N2_triple_bond_sto3g"]["points"]:
        mol = gr.reconstruct_mol_from_spec(pt["spec"])
        ev = eval_all_theories(mol)
        ev["R_angstrom"] = pt["R_angstrom"]
        ev["Exact_FCI"] = pt["energies"]["Exact_FCI"]
        ev["errors_vs_fci_kcal_mol"] = {
            k: (ev[k]["e_tot"] - pt["energies"]["Exact_FCI"]) * HARTREE_TO_KCAL
            for k in ["GPS_Ent", "GPS_Bogo", "AESF_DF"]
        }
        n2_sto_pts.append(ev)

    n2_631g_pts = []
    for pt in s4["N2_triple_bond_631g"]["points"]:
        mol = gr.reconstruct_mol_from_spec(pt["spec"])
        ev = eval_all_theories(mol)
        ev["R_angstrom"] = pt["R_angstrom"]
        ev["CASSCF_10e8o_NEVPT2"] = pt["energies"]["CASSCF_10e8o_NEVPT2"]
        ev["R_B3LYP"] = pt["energies"]["R_B3LYP"]
        ev["R_CCSD_T"] = pt["energies"]["R_CCSD_T"]
        n2_631g_pts.append(ev)
    ev_n2_8a_631g = eval_all_theories(gr.build_mol("N 0 0 0; N 0 0 8.0", basis="6-31g"))

    # 3d. H4 circle diradical cusp & barrier (6-31G, R = 1.738 A)
    print("  [3d] H4 circle diradical transition (6-31G)...", flush=True)
    h4_ref = ref["section_3_h4_diradical_circle"]["curves"]["6-31g"]["points"]
    h4_pts = []
    by_theta = {}
    for pt in h4_ref:
        th = pt["theta_deg"]
        if th in [70.0, 85.0, 88.0, 89.5, 90.0, 90.5, 92.0, 95.0, 110.0]:
            mol = gr.reconstruct_mol_from_spec(pt["spec"])
            ev = eval_all_theories(mol)
            ev["theta_deg"] = th
            ev["Exact_FCI"] = pt["energies"]["Exact_FCI"]
            h4_pts.append(ev)
            by_theta[th] = ev

    h4_cusp_mHa = {
        k: (
            by_theta[90.0][k]["e_tot"]
            - 0.5 * (by_theta[89.5][k]["e_tot"] + by_theta[90.5][k]["e_tot"])
        ) * 1000.0
        for k in ["GPS_Ent", "GPS_Bogo", "AESF_DF"]
    }
    h4_barrier_kcal = {
        k: (by_theta[90.0][k]["e_tot"] - by_theta[70.0][k]["e_tot"]) * HARTREE_TO_KCAL
        for k in ["GPS_Ent", "GPS_Bogo", "AESF_DF"]
    }

    return {
        "HF_631g": {
            "points": hf_pts,
            "R_8A": ev_hf_8a,
            "exact_F_plus_H_fci_ha": s4["HF_stretch_631g"]["exact_F2P_plus_H2S_asymptote_ha"],
            "own_method_F_plus_H_ha": {
                k: ev_f_631g[k]["e_tot"] + ev_h_631g[k]["e_tot"]
                for k in ["GPS_Ent", "GPS_Bogo", "AESF_DF"]
            },
        },
        "H2O_631g": {
            "points": h2o_pts,
            "exact_O_plus_2H_fci_ha": s4["H2O_symmetric_stretch_631g"]["exact_O3P_plus_2H_asymptote_ha"],
            "own_method_O_plus_2H_ha": {
                k: ev_o_631g[k]["e_tot"] + 2.0 * ev_h_631g[k]["e_tot"]
                for k in ["GPS_Ent", "GPS_Bogo", "AESF_DF"]
            },
        },
        "N2_sto3g": {
            "points": n2_sto_pts,
            "exact_2N_fci_ha": s4["N2_triple_bond_sto3g"]["exact_2N4S_asymptote_ha"],
            "own_method_2N_ha": {
                k: 2.0 * ev_n_sto3g[k]["e_tot"]
                for k in ["GPS_Ent", "GPS_Bogo", "AESF_DF"]
            },
        },
        "N2_631g": {
            "points": n2_631g_pts,
            "R_8A": ev_n2_8a_631g,
            "exact_2N_fci_ha": s4["N2_triple_bond_631g"]["exact_all_electron_2N4S_fci_asymptote_ha"],
            "own_method_2N_ha": {
                k: 2.0 * ev_n_631g[k]["e_tot"]
                for k in ["GPS_Ent", "GPS_Bogo", "AESF_DF"]
            },
            "multiplet_consistency_N2_8A_minus_2N_kcal_mol": {
                k: (ev_n2_8a_631g[k]["e_tot"] - 2.0 * ev_n_631g[k]["e_tot"]) * HARTREE_TO_KCAL
                for k in ["GPS_Ent", "GPS_Bogo", "AESF_DF"]
            },
        },
        "H4_631g": {
            "points": h4_pts,
            "cusp_mHa_at_90deg": h4_cusp_mHa,
            "fci_cusp_mHa": ref["section_3_h4_diradical_circle"]["curves"]["6-31g"]["cusp_metric_mHa_at_90deg"]["Exact_FCI"],
            "barrier_90_minus_70_kcal_mol": h4_barrier_kcal,
            "fci_barrier_kcal_mol": ref["section_3_h4_diradical_circle"]["curves"]["6-31g"]["barrier_90_minus_70_kcal_mol"]["Exact_FCI"],
        },
    }


# =============================================================================
# Audit 4: Equilibrium Reaction Energies (6-31G & cc-pVDZ) & Key Barriers
# =============================================================================

def run_audit_4(ref: Dict[str, Any]) -> Dict[str, Any]:
    print("[Audit 4] Running 5 Equilibrium Reactions & Key Barrier Heights...", flush=True)
    s5 = ref["section_5_equilibrium_chemistry"]

    reactions_def = [
        ("H2 + F2 -> 2 HF", {"HF": 2.0, "H2": -1.0, "F2": -1.0}),
        ("N2 + 3 H2 -> 2 NH3", {"NH3": 2.0, "N2": -1.0, "H2": -3.0}),
        ("CO + H2O -> CO2 + H2", {"CO2": 1.0, "H2": 1.0, "CO": -1.0, "H2O": -1.0}),
        ("CH4 + 2 H2O -> CO2 + 4 H2", {"CO2": 1.0, "H2": 4.0, "CH4": -1.0, "H2O": -2.0}),
        ("Li2 + H2 -> 2 LiH", {"LiH": 2.0, "Li2": -1.0, "H2": -1.0}),
    ]
    rxn_species = ["H2", "F2", "HF", "N2", "NH3", "CO", "H2O", "CO2", "CH4", "Li2", "LiH"]

    rxn_results = {}
    mol_evals_by_basis: Dict[str, Dict[str, Any]] = {}
    for basis in ["6-31g", "cc-pvdz"]:
        print(f"  [4-rxn] Evaluating 11 reaction molecules in {basis}...", flush=True)
        mols_ref = s5["thermochemistry"][basis]["molecules"]
        rxns_ref = s5["thermochemistry"][basis]["reaction_energies"]
        inc_thc = False
        mevals = {}
        for sp in rxn_species:
            mol = gr.reconstruct_mol_from_spec(mols_ref[sp]["spec"])
            mevals[sp] = eval_all_theories(mol, include_aesf_pure_thc=inc_thc, n_seeds_a=2)
        mol_evals_by_basis[basis] = mevals

        cand_keys = ["GPS_Ent", "GPS_Bogo", "AESF_DF"] + (["AESF_THC"] if inc_thc else [])
        per_rxn = {}
        abs_errs = {k: [] for k in cand_keys + ["RHF", "R_B3LYP", "R_PBE0", "R_MP2"]}

        for rlabel, stoich in reactions_def:
            gold_kcal = rxns_ref[rlabel]["R_CCSD_T"]["dE_kcal_mol"]
            row = {"CCSD_T_gold_kcal_mol": gold_kcal, "methods_kcal_mol": {}, "errors_kcal_mol": {}}
            for rkey in ["RHF", "R_B3LYP", "R_PBE0", "R_MP2"]:
                val = rxns_ref[rlabel][rkey]["dE_kcal_mol"]
                row["methods_kcal_mol"][rkey] = val
                row["errors_kcal_mol"][rkey] = val - gold_kcal
                abs_errs[rkey].append(abs(val - gold_kcal))
            for ckey in cand_keys:
                val = sum(c * mevals[sp][ckey]["e_tot"] for sp, c in stoich.items()) * HARTREE_TO_KCAL
                row["methods_kcal_mol"][ckey] = val
                row["errors_kcal_mol"][ckey] = val - gold_kcal
                abs_errs[ckey].append(abs(val - gold_kcal))
            per_rxn[rlabel] = row

        mae = {k: float(np.mean(v)) for k, v in abs_errs.items()}
        rxn_results[basis] = {"reactions": per_rxn, "MAE_vs_CCSD_T_kcal_mol": mae}

    # Key Barriers
    print("  [4-barriers] Evaluating H+H2 radical TS, NH3 inversion, H2O linear TS, and BeH2 C2v diradical TS...", flush=True)
    barriers_out = {}

    # 1. Collinear H + H2 -> [H...H...H]^‡ (cc-pVDZ Exact FCI = 10.15 kcal/mol!)
    h3_ref = s5["barriers"]["H_plus_H2_collinear_exchange_ccpvdz"]
    mol_h_cc = gr.reconstruct_mol_from_spec(s5["thermochemistry"]["cc-pvdz"]["atoms"]["H"]["spec"])
    ev_h_cc = eval_all_theories(mol_h_cc)
    ev_h2_cc = mol_evals_by_basis["cc-pvdz"]["H2"]
    ev_h3_ts = eval_all_theories(gr.reconstruct_mol_from_spec(h3_ref["H3_ts"]["spec"]))
    h3_cand = {
        k: (ev_h3_ts[k]["e_tot"] - (ev_h_cc[k]["e_tot"] + ev_h2_cc[k]["e_tot"])) * HARTREE_TO_KCAL
        for k in ["GPS_Ent", "GPS_Bogo", "AESF_DF"]
    }
    barriers_out["H_plus_H2_ccpvdz"] = {
        "reference_kcal_mol": h3_ref["barrier_kcal_mol"],
        "candidates_kcal_mol": h3_cand,
        "H3_ts_details": ev_h3_ts,
    }

    # 2. NH3 planar inversion (cc-pVDZ)
    nh3_ref = s5["barriers"]["NH3_planar_inversion"]["cc-pvdz"]
    ev_nh3_ts = eval_all_theories(gr.reconstruct_mol_from_spec(nh3_ref["ts"]["spec"]), n_seeds_a=2)
    ev_nh3_min = mol_evals_by_basis["cc-pvdz"]["NH3"]
    barriers_out["NH3_inversion_ccpvdz"] = {
        "reference_kcal_mol": nh3_ref["barrier_kcal_mol"],
        "candidates_kcal_mol": {
            k: (ev_nh3_ts[k]["e_tot"] - ev_nh3_min[k]["e_tot"]) * HARTREE_TO_KCAL
            for k in ["GPS_Ent", "GPS_Bogo", "AESF_DF"]
        },
    }

    # 3. H2O linear inversion (6-31G Exact FCI)
    h2o_inv_ref = s5["barriers"]["H2O_linear_inversion"]["6-31g"]
    ev_h2o_ts = eval_all_theories(gr.reconstruct_mol_from_spec(h2o_inv_ref["ts"]["spec"]), n_seeds_a=2)
    ev_h2o_min = mol_evals_by_basis["6-31g"]["H2O"]
    barriers_out["H2O_linear_inversion_631g"] = {
        "reference_kcal_mol": h2o_inv_ref["barrier_kcal_mol"],
        "candidates_kcal_mol": {
            k: (ev_h2o_ts[k]["e_tot"] - ev_h2o_min[k]["e_tot"]) * HARTREE_TO_KCAL
            for k in ["GPS_Ent", "GPS_Bogo", "AESF_DF"]
        },
    }

    # 4. BeH2 Purvis-Bartlett C2v Diradical Insertion Trajectory (6-31G Exact FCI = 70.99 kcal/mol)
    beh2_ref = s5["barriers"]["BeH2_C2v_insertion_631g"]
    beh2_pts = []
    for pt in beh2_ref["trajectory_points"]:
        ev = eval_all_theories(gr.reconstruct_mol_from_spec(pt["spec"]))
        ev["x_bohr"] = pt["x_bohr"]
        ev["Exact_FCI"] = pt["energies"]["Exact_FCI"]
        ev["RHF"] = pt["energies"]["RHF"]
        beh2_pts.append(ev)

    e_x4_cand = beh2_pts[-1]
    beh2_peak_cand = {
        k: (max(p[k]["e_tot"] for p in beh2_pts) - e_x4_cand[k]["e_tot"]) * HARTREE_TO_KCAL
        for k in ["GPS_Ent", "GPS_Bogo", "AESF_DF"]
    }
    barriers_out["BeH2_C2v_insertion_631g"] = {
        "reference_peak_barrier_kcal_mol": beh2_ref["peak_barrier_vs_x4_bohr_kcal_mol"],
        "candidates_peak_barrier_kcal_mol": beh2_peak_cand,
        "trajectory_points": beh2_pts,
    }

    return {
        "reaction_thermochemistry": rxn_results,
        "barriers": barriers_out,
    }


# =============================================================================
# Audit 5: Size-Consistency & Bogoliubov Global Chemical-Potential Mismatch
# =============================================================================

def run_audit_5(ref: Dict[str, Any]) -> Dict[str, Any]:
    print("[Audit 5] Running Size-Consistency Tests A, B, C & Heterogeneous mu-Mismatch...", flush=True)
    s6 = ref["section_6_size_consistency_extensivity"]

    ev_he = eval_all_theories(gr.reconstruct_mol_from_spec(s6["monomers_ccpvdz"]["He"]["spec"]))
    ev_h2s = eval_all_theories(gr.reconstruct_mol_from_spec(s6["monomers_ccpvdz"]["H2_R3p2A"]["spec"]))

    ev_ta = eval_all_theories(gr.reconstruct_mol_from_spec(s6["Test_A_He_He_100A_ccpvdz"]["supersystem"]["spec"]))
    ev_tb = eval_all_theories(gr.reconstruct_mol_from_spec(s6["Test_B_stretched_H2_dimer_100A_ccpvdz"]["supersystem"]["spec"]))
    ev_tc = eval_all_theories(gr.reconstruct_mol_from_spec(s6["Test_C_stretched_H2_plus_He_100A_ccpvdz"]["supersystem"]["spec"]))

    def sc_err(ev_ab, e_sum_fn):
        res = {}
        for k in ["GPS_Ent", "GPS_Bogo", "AESF_DF"]:
            de_tot = ev_ab[k]["e_tot"] - e_sum_fn(k, "e_tot")
            de_stat = ev_ab[k]["e_stat"] - e_sum_fn(k, "e_stat")
            de_dyn = ev_ab[k]["e_dyn"] - e_sum_fn(k, "e_dyn")
            res[k] = {
                "dE_tot_mHa": float(de_tot * 1000.0),
                "dE_tot_kcal_mol": float(de_tot * HARTREE_TO_KCAL),
                "dE_stat_mHa": float(de_stat * 1000.0),
                "dE_dyn_mHa": float(de_dyn * 1000.0),
            }
        return res

    err_a = sc_err(ev_ta, lambda k, f: 2.0 * ev_he[k][f])
    err_b = sc_err(ev_tb, lambda k, f: 2.0 * ev_h2s[k][f])
    err_c = sc_err(ev_tc, lambda k, f: ev_h2s[k][f] + ev_he[k][f])

    # Test E1: Stretched H2(3.2 A) + Closed-shell LiH(1.595 A) at 100 A in 6-31G
    # Tests Bogoliubov global chemical potential sign-flip when mu(H2) = -0.50 Ha < eps_HOMO(LiH) = -0.30 Ha!
    mol_h2_631 = gr.build_mol("H 0 0 0; H 0 0 3.2", basis="6-31g")
    mol_lih_631 = gr.build_mol("Li 0 0 0; H 0 0 1.5949", basis="6-31g")
    mol_e1 = gr.build_mol("H 0 0 0; H 0 0 3.2; Li 100.0 0 0; H 100.0 0 1.5949", basis="6-31g")
    ev_h2_631 = eval_all_theories(mol_h2_631)
    ev_lih_631 = eval_all_theories(mol_lih_631)
    ev_e1 = eval_all_theories(mol_e1)
    err_e1 = sc_err(ev_e1, lambda k, f: ev_h2_631[k][f] + ev_lih_631[k][f])

    # Test E2: Stretched H2(3.2 A) + Stretched HF(3.6 A) at 100 A in 6-31G
    # Tests two entangled fragments with different electronegativities / chemical potentials!
    mol_hf_631 = gr.build_mol("F 0 0 0; H 0 0 3.6", basis="6-31g")
    mol_e2 = gr.build_mol("H 0 0 0; H 0 0 3.2; F 100.0 0 0; H 100.0 0 3.6", basis="6-31g")
    ev_hf_631 = eval_all_theories(mol_hf_631)
    ev_e2 = eval_all_theories(mol_e2)
    err_e2 = sc_err(ev_e2, lambda k, f: ev_h2_631[k][f] + ev_hf_631[k][f])

    return {
        "Test_A_He_He_100A_ccpvdz": err_a,
        "Test_B_2x_H2_3p2A_100A_ccpvdz": err_b,
        "Test_C_H2_3p2A_plus_He_100A_ccpvdz": err_c,
        "Test_E1_H2_3p2A_plus_LiH_eq_100A_631g": err_e1,
        "Test_E2_H2_3p2A_plus_HF_3p6A_100A_631g": err_e2,
    }


def main() -> None:
    t0 = time.time()
    with open(REF_JSON, "r", encoding="utf-8") as f:
        ref = json.load(f)

    rebuild = "--rebuild" in sys.argv
    if (not rebuild) and os.path.exists(OUT_JSON):
        with open(OUT_JSON, "r", encoding="utf-8") as f:
            results: Dict[str, Any] = json.load(f)
    else:
        results = {}

    tasks = [
        ("audit_1_thc_and_rotation", run_audit_1),
        ("audit_2_one_and_three_electron", run_audit_2),
        ("audit_3_multireference_curves", run_audit_3),
        ("audit_4_equilibrium_and_barriers", run_audit_4),
        ("audit_5_size_consistency_and_mu_paradox", run_audit_5),
    ]

    for key, fn in tasks:
        if (not rebuild) and (key in results):
            print(f"[Checkpoint] Skipping {key}", flush=True)
            continue
        results[key] = fn(ref)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"DONE: Saved {OUT_JSON} in {time.time() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
