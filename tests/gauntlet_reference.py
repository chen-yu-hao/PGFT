#!/usr/bin/env python3
"""
Quantum Chemistry "Gauntlet of Death" Reference & Audit Suite
=============================================================
Author: Reviewer #2 (referee_two)
Output: /home/antigravity/project/tests/reference_data.json

Generates airtight, spin-verified PySCF reference benchmarks across:
  1. 1-Electron Self-Interaction Freedom (H, H2+ dissociation, H2+...He anti-cheat)
  2. Exact 2-Electron Dynamic vs Static Correlation (H-, He, Li+, He2 vdW, H2 PEC)
  3. Degenerate Diradical & Intruder-State Test (H4 circle theta = 70..110 deg)
  4. Simultaneous Multi-Bond Breaking (H2O double, N2 triple, LiH & HF heteronuclear)
  5. Single-Reference Equilibrium Chemistry (5 reactions, atomization, 5 barriers)
  6. Size-Consistency & Size-Extensivity Torture Tests (Tests A, B, C, D1, D2)

Also provides `audit_proposed_theory(energy_fn, reference_json_path)` so any
candidate O(N^3) non-DFT method `energy_fn(mol) -> float` can be audited directly
against the exact geometries and gold-standard numbers stored in `reference_data.json`.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pyscf
import scipy
from pyscf import cc, ci, dft, fci, gto, lo, mcscf, mp, mrpt, scf

HARTREE_TO_KCAL = 627.5094740631
OUTPUT_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_data.json")


# =============================================================================
# Core Solver Utilities (Spin-Pure, Root-Verified, Self-Checking)
# =============================================================================

def build_mol(
    atom: str,
    basis: str,
    charge: int = 0,
    spin: int = 0,
    unit: str = "Angstrom",
    symmetry: bool | str = False,
) -> gto.Mole:
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.unit = unit
    mol.symmetry = symmetry
    mol.verbose = 0
    mol.build()
    return mol


def mol_spec(mol: gto.Mole) -> Dict[str, Any]:
    return {
        "atom": mol.atom,
        "basis": mol.basis,
        "charge": int(mol.charge),
        "spin": int(mol.spin),
        "unit": mol.unit,
        "nelectron": int(mol.nelectron),
        "nao": int(mol.nao_nr()),
    }


def solve_singlet_fci(
    mol: gto.Mole,
    mf: scf.hf.SCF,
    wfnsym: Optional[str] = None,
    nroots: int = 1,
    fix_spin_shift: float = 0.8,
    localize_guess: bool = False,
    conv_tol: float = 1e-8,
) -> Tuple[float, float]:
    """
    Compute exact all-electron singlet (S=0) Full FCI energy and <S^2>.
    Fast path: runs unpenalized direct_spin1/direct_spin1_symm at conv_tol=1e-8
    and verifies <S^2> < 0.02 (avoiding 15x per-iteration 2-RDM fix_spin_ overhead
    and direct_spin0 ValueError traps).
    Fallback: if no pure singlet root is found, re-solves with fix_spin_.
    """
    mo_coeff = mf.mo_coeff.copy()
    if localize_guess and not mol.symmetry and mol.nelectron >= 4:
        nocc = mol.nelectron // 2
        mo_occ = lo.Boys(mol, mo_coeff[:, :nocc]).kernel()
        mo_vir = lo.Boys(mol, mo_coeff[:, nocc:]).kernel()
        mo_coeff = np.hstack([mo_occ, mo_vir])

    h1e = mo_coeff.T @ mf.get_hcore() @ mo_coeff
    eri = pyscf.ao2mo.kernel(mol, mo_coeff)
    norb = mo_coeff.shape[1]
    nelec = mol.nelec
    orbsym = (
        pyscf.symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_coeff)
        if mol.symmetry
        else None
    )

    def _run_solver(use_penalty: bool) -> Tuple[Optional[float], Optional[float]]:
        if mol.symmetry:
            base = fci.direct_spin1_symm.FCI(mol)
            base.orbsym = orbsym
            if wfnsym is not None:
                base.wfnsym = wfnsym
        else:
            base = fci.direct_spin1.FCI(mol)

        solver = (
            fci.addons.fix_spin_(base, ss=0.0, shift=fix_spin_shift)
            if use_penalty
            else base
        )
        solver.conv_tol = conv_tol
        solver.max_cycle = 150
        solver.max_space = 20
        solver.nroots = nroots

        e_all, ci_all = solver.kernel(h1e, eri, norb, nelec)
        if nroots == 1 or np.ndim(e_all) == 0:
            e_list = [float(e_all)]
            ci_list = [ci_all]
        else:
            e_list = [float(x) for x in e_all]
            ci_list = list(ci_all)

        best_e = None
        best_ss = None
        for ei, civec in zip(e_list, ci_list):
            ss, _ = fci.spin_op.spin_square(civec, norb, nelec)
            ss = float(ss)
            if ss < 0.02:
                if best_e is None or ei < best_e:
                    best_e = ei
                    best_ss = ss
        return best_e, best_ss

    best_e, best_ss = _run_solver(use_penalty=False)
    if best_e is None:
        best_e, best_ss = _run_solver(use_penalty=True)
    if best_e is None:
        raise RuntimeError("Failed to converge singlet (<S^2> < 0.02) FCI root!")

    return float(best_e + mol.energy_nuc()), float(best_ss)


def solve_open_shell_fci(mol: gto.Mole, mf: scf.hf.SCF, conv_tol: float = 1e-10) -> Tuple[float, float]:
    """Exact all-electron FCI for open-shell systems (S = spin/2)."""
    target_ss = 0.5 * mol.spin * (0.5 * mol.spin + 1.0)
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = pyscf.ao2mo.kernel(mol, mf.mo_coeff)
    norb = mf.mo_coeff.shape[1]
    nelec = mol.nelec

    base_solver = fci.direct_spin1.FCI(mol)
    solver = fci.addons.fix_spin_(base_solver, ss=target_ss, shift=0.5)
    solver.conv_tol = conv_tol
    solver.max_cycle = 200
    e_elec, civec = solver.kernel(h1e, eri, norb, nelec)
    ss, _ = fci.spin_op.spin_square(civec, norb, nelec)
    return float(e_elec + mol.energy_nuc()), float(ss)


def safe_scf_kernel(mf: scf.hf.SCF, dm0: Optional[np.ndarray] = None) -> scf.hf.SCF:
    """Run SCF/KS-DFT safely; fall back to SOSCF (Newton) if DIIS hits a singular subspace."""
    try:
        if dm0 is not None:
            mf.kernel(dm0=dm0)
        else:
            mf.kernel()
    except np.linalg.LinAlgError:
        mf.converged = False

    if not mf.converged:
        mf_soscf = mf.newton()
        mf_soscf.max_cycle = 30
        try:
            if dm0 is not None:
                mf_soscf.kernel(dm0=dm0)
            else:
                mf_soscf.kernel()
        except np.linalg.LinAlgError:
            pass
        if hasattr(mf_soscf, "undo_soscf"):
            return mf_soscf.undo_soscf()
        return mf_soscf
    return mf


def run_closed_shell_suite(
    mol: gto.Mole,
    include_fci: bool = True,
    include_cisd: bool = False,
    fci_nroots: int = 1,
    fci_localize: bool = False,
    mol_sym_for_fci: Optional[gto.Mole] = None,
    fci_wfnsym: Optional[str] = None,
    fci_conv_tol: float = 1e-8,
    check_rhf_internal_stability: bool = False,
) -> Dict[str, Any]:
    """
    Runs RHF, R-B3LYP, R-PBE0, R-MP2, (optional R-CISD), R-CCSD, R-CCSD(T),
    and (optional) spin-verified Exact All-Electron Full FCI on a closed-shell singlet.
    """
    res: Dict[str, Any] = {"spec": mol_spec(mol), "energies": {}, "converged": {}}

    # 1. RHF
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 150
    mf = safe_scf_kernel(mf)
    if check_rhf_internal_stability:
        mo_i, _, stable_i, _ = mf.stability(return_status=True, verbose=0)
        if not stable_i:
            dm_new = mf.make_rdm1(mo_i, mf.mo_occ)
            mf_retry = scf.RHF(mol)
            mf_retry.conv_tol = 1e-10
            mf_retry.max_cycle = 150
            mf = safe_scf_kernel(mf_retry, dm0=dm_new)
    res["energies"]["RHF"] = float(mf.e_tot)
    res["converged"]["RHF"] = bool(mf.converged)

    # HOMO-LUMO gap diagnostic (detects degeneracies/diradicals)
    nocc = mol.nelectron // 2
    if nocc < len(mf.mo_energy):
        res["homo_lumo_gap_ha"] = float(mf.mo_energy[nocc] - mf.mo_energy[nocc - 1])
    else:
        res["homo_lumo_gap_ha"] = None

    # 2. R-B3LYP & R-PBE0
    dm_rhf = mf.make_rdm1()
    for label, xc in [("R_B3LYP", "b3lyp"), ("R_PBE0", "pbe0")]:
        rks = dft.RKS(mol, xc=xc)
        rks.grids.level = 3
        rks.conv_tol = 1e-9
        rks.max_cycle = 150
        rks = safe_scf_kernel(rks, dm0=dm_rhf)
        res["energies"][label] = float(rks.e_tot)
        res["converged"][label] = bool(rks.converged)

    # 3. R-MP2
    mymp2 = mp.MP2(mf)
    mymp2.kernel()
    res["energies"]["R_MP2"] = float(mymp2.e_tot)

    # 4. Optional R-CISD (for size-consistency/extensivity failure demonstration)
    if include_cisd:
        myci = ci.RCISD(mf)
        myci.conv_tol = 1e-9
        myci.max_cycle = 120
        myci.kernel()
        res["energies"]["R_CISD"] = float(myci.e_tot)
        res["converged"]["R_CISD"] = bool(myci.converged)

    # 5. R-CCSD & R-CCSD(T)
    mycc = cc.RCCSD(mf)
    mycc.conv_tol = 1e-9
    mycc.max_cycle = 80
    mycc.kernel()
    res["converged"]["R_CCSD"] = bool(mycc.converged)
    if mycc.converged:
        res["energies"]["R_CCSD"] = float(mycc.e_tot)
        if mol.nelectron == 2:
            # Exact zero triples for 2 electrons
            res["energies"]["R_CCSD_T"] = float(mycc.e_tot)
        else:
            et = float(mycc.ccsd_t())
            res["energies"]["R_CCSD_T"] = float(mycc.e_tot + et)
    else:
        res["energies"]["R_CCSD"] = None
        res["energies"]["R_CCSD_T"] = None
        res["unconverged_R_CCSD_last_e"] = float(mycc.e_tot)

    # 6. Exact All-Electron Full FCI
    if include_fci:
        if mol_sym_for_fci is not None:
            mf_sym = scf.RHF(mol_sym_for_fci)
            mf_sym.conv_tol = 1e-10
            mf_sym.max_cycle = 150
            mf_sym = safe_scf_kernel(mf_sym)
            efci, s2 = solve_singlet_fci(
                mol_sym_for_fci,
                mf_sym,
                wfnsym=fci_wfnsym,
                nroots=fci_nroots,
                conv_tol=fci_conv_tol,
            )
        else:
            efci, s2 = solve_singlet_fci(
                mol,
                mf,
                nroots=fci_nroots,
                localize_guess=fci_localize,
                conv_tol=fci_conv_tol,
            )
        res["energies"]["Exact_FCI"] = efci
        res["fci_spin_square"] = s2

    return res


def run_open_shell_suite(
    mol: gto.Mole, include_fci: bool = True, include_dft: bool = True
) -> Dict[str, Any]:
    """
    Runs ROHF, UHF, UKS-B3LYP, UKS-PBE0, UMP2, UCCSD, UCCSD(T),
    and (optional) Exact All-Electron Full FCI on open-shell systems.
    """
    res: Dict[str, Any] = {"spec": mol_spec(mol), "energies": {}, "converged": {}}

    mf_ro = scf.ROHF(mol)
    mf_ro.conv_tol = 1e-10
    mf_ro.max_cycle = 150
    mf_ro = safe_scf_kernel(mf_ro)
    res["energies"]["ROHF"] = float(mf_ro.e_tot)
    res["converged"]["ROHF"] = bool(mf_ro.converged)

    mf_u = scf.UHF(mol)
    mf_u.conv_tol = 1e-10
    mf_u.max_cycle = 150
    mf_u = safe_scf_kernel(mf_u, dm0=mf_ro.make_rdm1())
    res["energies"]["UHF"] = float(mf_u.e_tot)
    res["uhf_spin_square"] = float(mf_u.spin_square()[0])

    if include_dft:
        dm_u = mf_u.make_rdm1()
        for label, xc in [("B3LYP", "b3lyp"), ("PBE0", "pbe0")]:
            uks = dft.UKS(mol, xc=xc)
            uks.grids.level = 3
            uks.conv_tol = 1e-9
            uks.max_cycle = 150
            uks = safe_scf_kernel(uks, dm0=dm_u)
            res["energies"][f"UKS_{label}"] = float(uks.e_tot)
            res["converged"][f"UKS_{label}"] = bool(uks.converged)

    if mol.nelectron == 1:
        # 1-electron exact wave-function methods have identically zero correlation energy
        res["energies"]["UMP2"] = float(mf_ro.e_tot)
        res["energies"]["UCCSD"] = float(mf_ro.e_tot)
        res["energies"]["UCCSD_T"] = float(mf_ro.e_tot)
    else:
        ump2 = mp.UMP2(mf_u)
        ump2.kernel()
        res["energies"]["UMP2"] = float(ump2.e_tot)

        ucc = cc.UCCSD(mf_u)
        ucc.conv_tol = 1e-9
        ucc.max_cycle = 80
        ucc.kernel()
        res["converged"]["UCCSD"] = bool(ucc.converged)
        if ucc.converged:
            res["energies"]["UCCSD"] = float(ucc.e_tot)
            if mol.nelectron >= 3 and sum(mol.nelec) >= 3:
                et = float(ucc.ccsd_t())
                res["energies"]["UCCSD_T"] = float(ucc.e_tot + et)
            else:
                res["energies"]["UCCSD_T"] = float(ucc.e_tot)
        else:
            res["energies"]["UCCSD"] = None
            res["energies"]["UCCSD_T"] = None

    if include_fci:
        efci, s2 = solve_open_shell_fci(mol, mf_ro)
        res["energies"]["Exact_FCI"] = efci
        res["fci_spin_square"] = s2

    return res


# =============================================================================
# Section 1: 1-Electron Self-Interaction Test (cc-pVDZ)
# =============================================================================

def generate_section_1() -> Dict[str, Any]:
    print("[Section 1] Running 1-Electron Self-Interaction Tests (cc-pVDZ)...", flush=True)
    t0 = time.time()
    basis = "cc-pvdz"

    mol_h = build_mol("H 0 0 0", basis=basis, charge=0, spin=1)
    h_res = run_open_shell_suite(mol_h, include_fci=True)
    assert abs(h_res["energies"]["Exact_FCI"] - h_res["energies"]["ROHF"]) < 1e-11

    r_grid = [0.7, 1.06, 1.5, 2.5, 4.0, 6.0]
    h2p_curve = []
    e_h_exact = h_res["energies"]["Exact_FCI"]
    for r in r_grid:
        mol = build_mol(f"H 0 0 0; H 0 0 {r}", basis=basis, charge=1, spin=1)
        pt = run_open_shell_suite(mol, include_fci=True)
        assert abs(pt["energies"]["Exact_FCI"] - pt["energies"]["ROHF"]) < 1e-11
        pt["R_angstrom"] = r
        pt["binding_kcal_mol"] = {
            k: (v - e_h_exact) * HARTREE_TO_KCAL
            for k, v in pt["energies"].items()
            if v is not None
        }
        pt["sie_error_kcal_mol"] = {
            k: (v - pt["energies"]["Exact_FCI"]) * HARTREE_TO_KCAL
            for k, v in pt["energies"].items()
            if v is not None
        }
        h2p_curve.append(pt)

    # Anti-cheat 3-electron supersystem: H2+(R=4.0 A) + He at 100 A
    mol_he = build_mol("He 0 0 0", basis=basis, charge=0, spin=0)
    he_res = run_closed_shell_suite(mol_he, include_fci=True)
    mol_h2p_4 = build_mol("H 0 0 0; H 0 0 4.0", basis=basis, charge=1, spin=1)
    h2p_4_exact = run_open_shell_suite(mol_h2p_4, include_fci=True)["energies"]["Exact_FCI"]

    mol_super = build_mol(
        "H 0 0 0; H 0 0 4.0; He 100.0 0 0", basis=basis, charge=1, spin=1
    )
    super_res = run_open_shell_suite(mol_super, include_fci=True, include_dft=False)
    super_sep_check = (
        super_res["energies"]["Exact_FCI"]
        - he_res["energies"]["Exact_FCI"]
        - h2p_4_exact
    )
    assert abs(super_sep_check) < 1e-8, f"3e supersystem FCI mismatch: {super_sep_check}"

    print(f"  -> Section 1 completed in {time.time() - t0:.2f}s", flush=True)
    return {
        "basis": basis,
        "H_atom": h_res,
        "H2_plus_curve": h2p_curve,
        "anti_cheat_H2plus_4A_plus_He_100A": {
            "supersystem": super_res,
            "E_H2plus_4A_exact": h2p_4_exact,
            "E_He_exact": he_res["energies"]["Exact_FCI"],
            "fci_additivity_error_ha": float(super_sep_check),
        },
    }


# =============================================================================
# Section 2: Exact 2-Electron Dynamic vs Static Correlation (cc-pVDZ)
# =============================================================================

def generate_section_2() -> Dict[str, Any]:
    print("[Section 2] Running Exact 2-Electron Correlation Suite (cc-pVDZ)...", flush=True)
    t0 = time.time()
    basis = "cc-pvdz"

    # 2a. Isoelectronic 2-electron ions: H-, He, Li+
    ions = {}
    for label, sym, chg in [("H_minus", "H", -1), ("He", "He", 0), ("Li_plus", "Li", 1)]:
        mol = build_mol(f"{sym} 0 0 0", basis=basis, charge=chg, spin=0)
        res = run_closed_shell_suite(mol, include_fci=True)
        assert abs(res["energies"]["R_CCSD"] - res["energies"]["Exact_FCI"]) < 1e-8
        res["exact_correlation_energy_ha"] = float(
            res["energies"]["Exact_FCI"] - res["energies"]["RHF"]
        )
        ions[label] = res

    # 2b. He2 van der Waals dispersion curve (cc-pVDZ)
    he2_curve = []
    e_he_fci = ions["He"]["energies"]["Exact_FCI"]
    for r in [2.6, 3.0, 3.5, 5.0, 100.0]:
        mol = build_mol(f"He 0 0 0; He 0 0 {r}", basis=basis, charge=0, spin=0)
        pt = run_closed_shell_suite(mol, include_fci=True)
        pt["R_angstrom"] = r
        pt["interaction_energy_uHa"] = {
            "RHF": (pt["energies"]["RHF"] - 2 * ions["He"]["energies"]["RHF"]) * 1e6,
            "Exact_FCI": (pt["energies"]["Exact_FCI"] - 2 * e_he_fci) * 1e6,
            "R_CCSD_T": (pt["energies"]["R_CCSD_T"] - 2 * ions["He"]["energies"]["R_CCSD_T"]) * 1e6,
        }
        he2_curve.append(pt)

    # 2c. H2 full potential energy curve (cc-pVDZ)
    mol_h = build_mol("H 0 0 0", basis=basis, charge=0, spin=1)
    e_2h_exact = 2.0 * solve_open_shell_fci(mol_h, scf.ROHF(mol_h).run())[0]

    h2_r_grid = [0.5, 0.74, 1.0, 1.4, 1.8, 2.4, 3.2, 4.5]
    h2_curve = []
    for r in h2_r_grid:
        mol = build_mol(f"H 0 0 0; H 0 0 {r}", basis=basis)
        pt = run_closed_shell_suite(mol, include_fci=True)
        assert abs(pt["energies"]["R_CCSD"] - pt["energies"]["Exact_FCI"]) < 1e-8
        pt["R_angstrom"] = r
        pt["errors_vs_fci_kcal_mol"] = {
            k: (v - pt["energies"]["Exact_FCI"]) * HARTREE_TO_KCAL
            for k, v in pt["energies"].items()
            if v is not None
        }
        h2_curve.append(pt)

    print(f"  -> Section 2 completed in {time.time() - t0:.2f}s", flush=True)
    return {
        "basis": basis,
        "two_electron_ions": ions,
        "He2_dispersion_curve": he2_curve,
        "H2_dissociation_curve": h2_curve,
        "H2_exact_2H_asymptote_ha": float(e_2h_exact),
    }


# =============================================================================
# Section 3: Degenerate Diradical & Intruder-State Test (H4 Circle)
# =============================================================================

def h4_circle_atom_str(theta_deg: float, radius_ang: float = 1.738) -> str:
    half = math.radians(theta_deg / 2.0)
    x = radius_ang * math.cos(half)
    y = radius_ang * math.sin(half)
    return (
        f"H {x:.10f} {y:.10f} 0.0; "
        f"H {-x:.10f} {y:.10f} 0.0; "
        f"H {-x:.10f} {-y:.10f} 0.0; "
        f"H {x:.10f} {-y:.10f} 0.0"
    )


def generate_section_3() -> Dict[str, Any]:
    print("[Section 3] Running H4 Circle Diradical & Intruder Tests (6-31G & cc-pVDZ)...", flush=True)
    t0 = time.time()
    angles = [70.0, 80.0, 85.0, 88.0, 89.5, 90.0, 90.5, 92.0, 95.0, 100.0, 110.0]
    radius = 1.738
    out: Dict[str, Any] = {"radius_angstrom": radius, "angles_deg": angles, "curves": {}}

    for basis in ["6-31g", "cc-pvdz"]:
        curve = []
        by_angle: Dict[float, Dict[str, Any]] = {}
        for theta in angles:
            mol = build_mol(h4_circle_atom_str(theta, radius), basis=basis, symmetry=False)
            pt = run_closed_shell_suite(
                mol, include_fci=True, fci_nroots=1, check_rhf_internal_stability=True
            )
            pt["theta_deg"] = theta
            curve.append(pt)
            by_angle[theta] = pt

        # Quantify non-analytic cusp at theta = 90 deg: E(90) - 0.5*(E(89.5) + E(90.5))
        cusp_mha = {}
        barrier_90_minus_70_kcal = {}
        for m in ["RHF", "R_B3LYP", "R_PBE0", "R_MP2", "R_CCSD", "R_CCSD_T", "Exact_FCI"]:
            e90 = by_angle[90.0]["energies"][m]
            e895 = by_angle[89.5]["energies"][m]
            e905 = by_angle[90.5]["energies"][m]
            e70 = by_angle[70.0]["energies"][m]
            cusp_mha[m] = float((e90 - 0.5 * (e895 + e905)) * 1000.0)
            barrier_90_minus_70_kcal[m] = float((e90 - e70) * HARTREE_TO_KCAL)

        out["curves"][basis] = {
            "points": curve,
            "cusp_metric_mHa_at_90deg": cusp_mha,
            "barrier_90_minus_70_kcal_mol": barrier_90_minus_70_kcal,
        }

    print(f"  -> Section 3 completed in {time.time() - t0:.2f}s", flush=True)
    return out


# =============================================================================
# Section 4: Simultaneous Multi-Bond Breaking (H2O, N2, LiH, HF)
# =============================================================================

def h2o_atom_str(r_oh: float, angle_deg: float = 104.5) -> str:
    half = math.radians(angle_deg / 2.0)
    y = r_oh * math.sin(half)
    z = r_oh * math.cos(half)
    return f"O 0.0 0.0 0.0; H 0.0 {y:.10f} {z:.10f}; H 0.0 {-y:.10f} {z:.10f}"


def generate_section_4() -> Dict[str, Any]:
    print("[Section 4] Running Simultaneous Multi-Bond Breaking Suite...", flush=True)
    t0 = time.time()

    # Exact atomic asymptotes in STO-3G and 6-31G
    mol_h_631g = build_mol("H 0 0 0", basis="6-31g", spin=1)
    e_h_631g = solve_open_shell_fci(mol_h_631g, scf.ROHF(mol_h_631g).run())[0]

    mol_o_631g = build_mol("O 0 0 0", basis="6-31g", spin=2)
    e_o_631g = solve_open_shell_fci(mol_o_631g, scf.ROHF(mol_o_631g).run())[0]

    mol_li_631g = build_mol("Li 0 0 0", basis="6-31g", spin=1)
    e_li_631g = solve_open_shell_fci(mol_li_631g, scf.ROHF(mol_li_631g).run())[0]

    mol_f_631g = build_mol("F 0 0 0", basis="6-31g", spin=1)
    e_f_631g = solve_open_shell_fci(mol_f_631g, scf.ROHF(mol_f_631g).run())[0]

    mol_n_sto3g = build_mol("N 0 0 0", basis="sto-3g", spin=3)
    e_n_sto3g = solve_open_shell_fci(mol_n_sto3g, scf.ROHF(mol_n_sto3g).run())[0]

    mol_n_631g = build_mol("N 0 0 0", basis="6-31g", spin=3)
    e_n_631g = solve_open_shell_fci(mol_n_631g, scf.ROHF(mol_n_631g).run())[0]

    # 4a. H2O Symmetric Double-Bond Stretch (6-31G Exact All-Electron Full FCI)
    print("  [4a] H2O symmetric double-bond stretch (6-31G Full FCI)...", flush=True)
    h2o_r_grid = [0.8, 0.96, 1.2, 1.5, 1.8, 2.2, 2.6, 3.0]
    h2o_curve = []
    for r in h2o_r_grid:
        astr = h2o_atom_str(r, 104.5)
        mol_c1 = build_mol(astr, basis="6-31g", symmetry=False)
        mol_sym = build_mol(astr, basis="6-31g", symmetry="C2v")
        pt = run_closed_shell_suite(
            mol_c1,
            include_fci=True,
            fci_nroots=1,
            mol_sym_for_fci=mol_sym,
            fci_wfnsym="A1",
        )
        pt["R_OH_angstrom"] = r
        h2o_curve.append(pt)

    # 4b. N2 Triple-Bond Dissociation in STO-3G (Exact All-Electron Full FCI)
    print("  [4b] N2 triple-bond stretch (STO-3G Full FCI)...", flush=True)
    n2_r_grid = [0.9, 1.098, 1.3, 1.6, 1.9, 2.2, 2.6, 3.2]
    n2_sto3g_curve = []
    for r in n2_r_grid:
        astr = f"N 0 0 0; N 0 0 {r}"
        mol_c1 = build_mol(astr, basis="sto-3g", symmetry=False)
        mol_sym = build_mol(astr, basis="sto-3g", symmetry="D2h")
        pt = run_closed_shell_suite(
            mol_c1,
            include_fci=True,
            fci_nroots=2,
            mol_sym_for_fci=mol_sym,
            fci_wfnsym="Ag",
        )
        pt["R_angstrom"] = r
        n2_sto3g_curve.append(pt)

    # 4c. N2 Triple-Bond Dissociation in 6-31G (RHF/DFT/MP2/CCSD/CCSD(T) + CASSCF(10,8)/NEVPT2 + Exact 2*N(^4S) FCI limit)
    print("  [4c] N2 triple-bond stretch (6-31G + CASSCF(10,8)/NEVPT2 + Exact 2N FCI limit)...", flush=True)
    irrep_cas_n2 = {"Ag": 2, "B1u": 2, "B2u": 1, "B3u": 1, "B2g": 1, "B3g": 1}
    n2_631g_curve = []
    for r in n2_r_grid:
        astr = f"N 0 0 0; N 0 0 {r}"
        mol_c1 = build_mol(astr, basis="6-31g", symmetry=False)
        pt = run_closed_shell_suite(mol_c1, include_fci=False)

        mol_sym = build_mol(astr, basis="6-31g", symmetry="D2h")
        mf_sym = scf.RHF(mol_sym).run()
        mc = mcscf.CASSCF(mf_sym, 8, 10)
        mc.fcisolver = fci.addons.fix_spin_(
            fci.direct_spin0_symm.FCI(mol_sym), ss=0.0, shift=0.5
        )
        mc.fcisolver.wfnsym = "Ag"
        mc.conv_tol = 1e-9
        mo_init = mcscf.sort_mo_by_irrep(mc, mf_sym.mo_coeff, irrep_cas_n2)
        mc.kernel(mo_init)
        e_nevpt2 = float(mc.e_tot + mrpt.NEVPT(mc).kernel())
        ss_cas, _ = fci.spin_op.spin_square0(mc.ci, 8, (5, 5))

        pt["energies"]["CASSCF_10e8o"] = float(mc.e_tot)
        pt["energies"]["CASSCF_10e8o_NEVPT2"] = e_nevpt2
        pt["casscf_spin_square"] = float(ss_cas)
        pt["R_angstrom"] = r
        n2_631g_curve.append(pt)

    # 4d. LiH Heteronuclear Single-Bond Stretch (6-31G Exact All-Electron Full FCI)
    print("  [4d] LiH heteronuclear stretch (6-31G Full FCI)...", flush=True)
    lih_r_grid = [0.8, 1.2, 1.595, 2.0, 2.5, 3.0, 3.6]
    lih_curve = []
    for r in lih_r_grid:
        astr = f"Li 0 0 0; H 0 0 {r}"
        mol_c1 = build_mol(astr, basis="6-31g", symmetry=False)
        mol_sym = build_mol(astr, basis="6-31g", symmetry="C2v")
        pt = run_closed_shell_suite(
            mol_c1,
            include_fci=True,
            fci_nroots=2,
            mol_sym_for_fci=mol_sym,
            fci_wfnsym="A1",
        )
        pt["R_angstrom"] = r
        lih_curve.append(pt)

    # 4e. HF Heteronuclear Single-Bond Stretch (6-31G Exact All-Electron Full FCI)
    print("  [4e] HF heteronuclear stretch (6-31G Full FCI)...", flush=True)
    hf_r_grid = [0.75, 0.917, 1.2, 1.6, 2.0, 2.6, 3.2, 3.6]
    hf_curve = []
    for r in hf_r_grid:
        astr = f"F 0 0 0; H 0 0 {r}"
        mol_c1 = build_mol(astr, basis="6-31g", symmetry=False)
        mol_sym = build_mol(astr, basis="6-31g", symmetry="C2v")
        pt = run_closed_shell_suite(
            mol_c1,
            include_fci=True,
            fci_nroots=2,
            mol_sym_for_fci=mol_sym,
            fci_wfnsym="A1",
        )
        pt["R_angstrom"] = r
        hf_curve.append(pt)

    print(f"  -> Section 4 completed in {time.time() - t0:.2f}s", flush=True)
    return {
        "H2O_symmetric_stretch_631g": {
            "angle_HOH_deg": 104.5,
            "exact_O3P_plus_2H_asymptote_ha": float(e_o_631g + 2.0 * e_h_631g),
            "points": h2o_curve,
        },
        "N2_triple_bond_sto3g": {
            "exact_2N4S_asymptote_ha": float(2.0 * e_n_sto3g),
            "points": n2_sto3g_curve,
        },
        "N2_triple_bond_631g": {
            "exact_all_electron_2N4S_fci_asymptote_ha": float(2.0 * e_n_631g),
            "points": n2_631g_curve,
        },
        "LiH_stretch_631g": {
            "exact_Li2S_plus_H2S_asymptote_ha": float(e_li_631g + e_h_631g),
            "points": lih_curve,
        },
        "HF_stretch_631g": {
            "exact_F2P_plus_H2S_asymptote_ha": float(e_f_631g + e_h_631g),
            "points": hf_curve,
        },
    }


# =============================================================================
# Section 5: Single-Reference Equilibrium Chemistry & Barrier Heights
# =============================================================================

def nh3_pyramidal_str(r_nh: float = 1.012, hnh_deg: float = 106.7) -> str:
    # Place N at origin; 3 H atoms with bond length r_nh and angle hnh_deg
    # sin(theta_z) = (2/sqrt(3)) * sin(hnh/2), where theta_z is angle from C3 z-axis
    s = (2.0 / math.sqrt(3.0)) * math.sin(math.radians(hnh_deg / 2.0))
    c = math.sqrt(max(0.0, 1.0 - s * s))
    rho = r_nh * s
    z = r_nh * c
    pts = ["N 0.0 0.0 0.0"]
    for k in range(3):
        phi = 2.0 * math.pi * k / 3.0
        pts.append(f"H {rho * math.cos(phi):.10f} {rho * math.sin(phi):.10f} {z:.10f}")
    return "; ".join(pts)


def ch4_tetrahedral_str(r_ch: float = 1.089) -> str:
    d = r_ch / math.sqrt(3.0)
    return (
        f"C 0.0 0.0 0.0; "
        f"H {d:.10f} {d:.10f} {d:.10f}; "
        f"H {d:.10f} {-d:.10f} {-d:.10f}; "
        f"H {-d:.10f} {d:.10f} {-d:.10f}; "
        f"H {-d:.10f} {-d:.10f} {d:.10f}"
    )


def c2h6_rotamer_str(dihedral_offset_deg: float, r_cc: float = 1.526, r_ch: float = 1.091, cch_deg: float = 111.2) -> str:
    z_c = 0.5 * r_cc
    theta = math.radians(180.0 - cch_deg)  # angle from +z axis for top methyl
    rho = r_ch * math.sin(theta)
    dz = r_ch * math.cos(theta)
    atoms = [f"C 0.0 0.0 {z_c:.10f}", f"C 0.0 0.0 {-z_c:.10f}"]
    for k in range(3):
        phi1 = math.radians(120.0 * k)
        atoms.append(f"H {rho * math.cos(phi1):.10f} {rho * math.sin(phi1):.10f} {z_c + dz:.10f}")
        phi2 = math.radians(120.0 * k + dihedral_offset_deg)
        atoms.append(f"H {rho * math.cos(phi2):.10f} {rho * math.sin(phi2):.10f} {-z_c - dz:.10f}")
    return "; ".join(atoms)


def generate_section_5() -> Dict[str, Any]:
    print("[Section 5] Running Equilibrium Thermochemistry & Barrier Suite...", flush=True)
    t0 = time.time()

    eq_geoms = {
        "H2": ("H 0 0 0; H 0 0 0.7414", {"H": 2}),
        "Li2": ("Li 0 0 0; Li 0 0 2.6730", {"Li": 2}),
        "LiH": ("Li 0 0 0; H 0 0 1.5949", {"Li": 1, "H": 1}),
        "BeH2": ("Be 0 0 0; H 0 0 1.3264; H 0 0 -1.3264", {"Be": 1, "H": 2}),
        "CH4": (ch4_tetrahedral_str(1.089), {"C": 1, "H": 4}),
        "NH3": (nh3_pyramidal_str(1.012, 106.7), {"N": 1, "H": 3}),
        "H2O": (h2o_atom_str(0.9578, 104.5), {"O": 1, "H": 2}),
        "HF": ("F 0 0 0; H 0 0 0.9168", {"F": 1, "H": 1}),
        "N2": ("N 0 0 0; N 0 0 1.0977", {"N": 2}),
        "CO": ("C 0 0 0; O 0 0 1.1283", {"C": 1, "O": 1}),
        "F2": ("F 0 0 0; F 0 0 1.4119", {"F": 2}),
        "CO2": ("C 0 0 0; O 0 0 1.1600; O 0 0 -1.1600", {"C": 1, "O": 2}),
    }

    atom_spins = {"H": 1, "Li": 1, "Be": 0, "C": 2, "N": 3, "O": 2, "F": 1}

    # Molecules eligible for fast (<4s) Exact All-Electron Full FCI
    fci_molecules_by_basis = {
        "6-31g": {"H2", "Li2", "LiH", "BeH2", "H2O", "HF"},
        "cc-pvdz": {"H2", "LiH"},
    }
    fci_atoms_by_basis = {
        "6-31g": {"H", "Li", "Be", "C", "N", "O", "F"},
        "cc-pvdz": {"H", "Li", "Be"},
    }

    reactions_def = [
        ("H2 + F2 -> 2 HF", {"HF": 2.0, "H2": -1.0, "F2": -1.0}),
        ("N2 + 3 H2 -> 2 NH3", {"NH3": 2.0, "N2": -1.0, "H2": -3.0}),
        ("CO + H2O -> CO2 + H2", {"CO2": 1.0, "H2": 1.0, "CO": -1.0, "H2O": -1.0}),
        ("CH4 + 2 H2O -> CO2 + 4 H2", {"CO2": 1.0, "H2": 4.0, "CH4": -1.0, "H2O": -2.0}),
        ("Li2 + H2 -> 2 LiH", {"LiH": 2.0, "Li2": -1.0, "H2": -1.0}),
    ]

    thermo_by_basis: Dict[str, Any] = {}
    for basis in ["6-31g", "cc-pvdz"]:
        print(f"  [5-thermo] Computing atoms & molecules in {basis}...", flush=True)
        atoms_out = {}
        for el, sp in atom_spins.items():
            mol_a = build_mol(f"{el} 0 0 0", basis=basis, spin=sp)
            do_fci = el in fci_atoms_by_basis[basis]
            if sp == 0:
                atoms_out[el] = run_closed_shell_suite(mol_a, include_fci=do_fci)
            else:
                atoms_out[el] = run_open_shell_suite(mol_a, include_fci=do_fci)

        mols_out = {}
        for mname, (astr, _) in eq_geoms.items():
            mol_m = build_mol(astr, basis=basis, symmetry=False)
            do_fci = mname in fci_molecules_by_basis[basis]
            if do_fci and mname == "H2O":
                mol_sym = build_mol(astr, basis=basis, symmetry="C2v")
                mols_out[mname] = run_closed_shell_suite(
                    mol_m, include_fci=True, mol_sym_for_fci=mol_sym, fci_wfnsym="A1"
                )
            else:
                mols_out[mname] = run_closed_shell_suite(mol_m, include_fci=do_fci)

        # Reaction energies (kcal/mol)
        rxn_out = {}
        for rlabel, stoich in reactions_def:
            rxn_vals = {}
            for meth in ["RHF", "R_B3LYP", "R_PBE0", "R_MP2", "R_CCSD", "R_CCSD_T", "Exact_FCI"]:
                if all(meth in mols_out[sp]["energies"] and mols_out[sp]["energies"][meth] is not None for sp in stoich):
                    de_ha = sum(coeff * mols_out[sp]["energies"][meth] for sp, coeff in stoich.items())
                    rxn_vals[meth] = {
                        "dE_ha": float(de_ha),
                        "dE_kcal_mol": float(de_ha * HARTREE_TO_KCAL),
                    }
            rxn_out[rlabel] = rxn_vals

        # Atomization energies D_e = sum_atoms E(atom) - E(mol) (kcal/mol)
        atomization_out = {}
        method_pairs = [
            ("HF", "RHF", "UHF"),
            ("B3LYP", "R_B3LYP", "UKS_B3LYP"),
            ("PBE0", "R_PBE0", "UKS_PBE0"),
            ("MP2", "R_MP2", "UMP2"),
            ("CCSD", "R_CCSD", "UCCSD"),
            ("CCSD_T", "R_CCSD_T", "UCCSD_T"),
            ("Exact_FCI", "Exact_FCI", "Exact_FCI"),
        ]
        for mname, (_, comp) in eq_geoms.items():
            ae_vals = {}
            for label, closed_key, open_key in method_pairs:
                if closed_key not in mols_out[mname]["energies"] or mols_out[mname]["energies"][closed_key] is None:
                    continue
                ok = True
                e_atoms = 0.0
                for el, count in comp.items():
                    akey = closed_key if atom_spins[el] == 0 else open_key
                    if akey not in atoms_out[el]["energies"] or atoms_out[el]["energies"][akey] is None:
                        ok = False
                        break
                    e_atoms += count * atoms_out[el]["energies"][akey]
                if ok:
                    de_ha = e_atoms - mols_out[mname]["energies"][closed_key]
                    ae_vals[label] = {
                        "De_ha": float(de_ha),
                        "De_kcal_mol": float(de_ha * HARTREE_TO_KCAL),
                    }
            atomization_out[mname] = ae_vals

        thermo_by_basis[basis] = {
            "atoms": atoms_out,
            "molecules": mols_out,
            "reaction_energies": rxn_out,
            "atomization_energies": atomization_out,
        }

    # 5b. Barrier Heights
    print("  [5-barriers] Computing 5 conformational, inversion, radical & diradical barriers...", flush=True)
    barriers: Dict[str, Any] = {}

    # Barrier 1: H2O linear inversion barrier (C2v -> D_inf_h)
    barriers["H2O_linear_inversion"] = {}
    for basis in ["6-31g", "cc-pvdz"]:
        do_fci = basis == "6-31g"
        eq_res = thermo_by_basis[basis]["molecules"]["H2O"]
        ts_str = "O 0 0 0; H 0 0 0.9330; H 0 0 -0.9330"
        mol_ts = build_mol(ts_str, basis=basis, symmetry=False)
        mol_ts_sym = build_mol(ts_str, basis=basis, symmetry="D2h") if do_fci else None
        ts_res = run_closed_shell_suite(
            mol_ts, include_fci=do_fci, mol_sym_for_fci=mol_ts_sym, fci_wfnsym="Ag"
        )
        bvals = {
            m: float((ts_res["energies"][m] - eq_res["energies"][m]) * HARTREE_TO_KCAL)
            for m in ts_res["energies"]
            if ts_res["energies"][m] is not None and eq_res["energies"].get(m) is not None
        }
        barriers["H2O_linear_inversion"][basis] = {
            "min": eq_res,
            "ts": ts_res,
            "barrier_kcal_mol": bvals,
        }

    # Barrier 2: NH3 planar inversion barrier (C3v -> D3h)
    barriers["NH3_planar_inversion"] = {}
    for basis in ["6-31g", "cc-pvdz"]:
        eq_res = thermo_by_basis[basis]["molecules"]["NH3"]
        ts_str = nh3_pyramidal_str(0.995, 120.0)
        ts_res = run_closed_shell_suite(build_mol(ts_str, basis=basis), include_fci=False)
        bvals = {
            m: float((ts_res["energies"][m] - eq_res["energies"][m]) * HARTREE_TO_KCAL)
            for m in ts_res["energies"]
            if ts_res["energies"][m] is not None
        }
        barriers["NH3_planar_inversion"][basis] = {
            "min": eq_res,
            "ts": ts_res,
            "barrier_kcal_mol": bvals,
        }

    # Barrier 3: C2H6 internal rotation barrier (Staggered D3d -> Eclipsed D3h)
    barriers["C2H6_internal_rotation"] = {}
    for basis in ["6-31g", "cc-pvdz"]:
        stag_res = run_closed_shell_suite(
            build_mol(c2h6_rotamer_str(60.0), basis=basis), include_fci=False
        )
        eclip_res = run_closed_shell_suite(
            build_mol(c2h6_rotamer_str(0.0), basis=basis), include_fci=False
        )
        bvals = {
            m: float((eclip_res["energies"][m] - stag_res["energies"][m]) * HARTREE_TO_KCAL)
            for m in stag_res["energies"]
            if stag_res["energies"][m] is not None
        }
        barriers["C2H6_internal_rotation"][basis] = {
            "staggered": stag_res,
            "eclipsed": eclip_res,
            "barrier_kcal_mol": bvals,
        }

    # Barrier 4: Collinear H + H2 -> [H...H...H]^‡ -> H2 + H (cc-pVDZ Exact Full FCI!)
    mol_h3_ts = build_mol("H 0 0 -0.930; H 0 0 0.0; H 0 0 0.930", basis="cc-pvdz", spin=1)
    h3_ts_res = run_open_shell_suite(mol_h3_ts, include_fci=True)
    h_ccpvdz = thermo_by_basis["cc-pvdz"]["atoms"]["H"]
    h2_ccpvdz = thermo_by_basis["cc-pvdz"]["molecules"]["H2"]
    h3_pairs = [
        ("ROHF", "ROHF", "RHF"),
        ("UHF", "UHF", "RHF"),
        ("B3LYP", "UKS_B3LYP", "R_B3LYP"),
        ("PBE0", "UKS_PBE0", "R_PBE0"),
        ("MP2", "UMP2", "R_MP2"),
        ("CCSD", "UCCSD", "R_CCSD"),
        ("CCSD_T", "UCCSD_T", "R_CCSD_T"),
        ("Exact_FCI", "Exact_FCI", "Exact_FCI"),
    ]
    h3_barriers = {}
    for label, open_k, closed_k in h3_pairs:
        e_react = h_ccpvdz["energies"][open_k] + h2_ccpvdz["energies"][closed_k]
        e_ts = h3_ts_res["energies"][open_k]
        h3_barriers[label] = float((e_ts - e_react) * HARTREE_TO_KCAL)
    barriers["H_plus_H2_collinear_exchange_ccpvdz"] = {
        "H3_ts": h3_ts_res,
        "barrier_kcal_mol": h3_barriers,
    }

    # Barrier 5: BeH2 Purvis-Bartlett C2v Diradical Insertion Trajectory (6-31G Exact Full FCI!)
    beh2_x_grid = [0.0, 1.0, 2.0, 2.5, 2.75, 2.85, 3.0, 4.0]
    beh2_curve = []
    for x_bohr in beh2_x_grid:
        y_bohr = 2.54 - 0.46 * x_bohr
        astr = f"Be 0.0 0.0 0.0; H 0.0 {y_bohr:.10f} {x_bohr:.10f}; H 0.0 {-y_bohr:.10f} {x_bohr:.10f}"
        mol_c1 = build_mol(astr, basis="6-31g", unit="Bohr", symmetry=False)
        mol_sym = build_mol(astr, basis="6-31g", unit="Bohr", symmetry="C2v")
        pt = run_closed_shell_suite(
            mol_c1,
            include_fci=True,
            fci_nroots=2,
            mol_sym_for_fci=mol_sym,
            fci_wfnsym="A1",
        )
        pt["x_bohr"] = x_bohr
        pt["y_bohr"] = y_bohr
        beh2_curve.append(pt)

    # Insertion barrier relative to Be + H2 reactant end (x = 4.0 Bohr)
    e_x4 = beh2_curve[-1]["energies"]
    beh2_peak_barrier_kcal = {}
    for m in ["RHF", "R_B3LYP", "R_PBE0", "R_MP2", "R_CCSD", "R_CCSD_T", "Exact_FCI"]:
        max_e = max(pt["energies"][m] for pt in beh2_curve if pt["energies"][m] is not None)
        beh2_peak_barrier_kcal[m] = float((max_e - e_x4[m]) * HARTREE_TO_KCAL)

    barriers["BeH2_C2v_insertion_631g"] = {
        "trajectory_points": beh2_curve,
        "peak_barrier_vs_x4_bohr_kcal_mol": beh2_peak_barrier_kcal,
    }

    print(f"  -> Section 5 completed in {time.time() - t0:.2f}s", flush=True)
    return {
        "thermochemistry": thermo_by_basis,
        "barriers": barriers,
    }


# =============================================================================
# Section 6: Size-Consistency & Size-Extensivity Torture Tests
# =============================================================================

def generate_section_6() -> Dict[str, Any]:
    print("[Section 6] Running Size-Consistency & Size-Extensivity Torture Tests...", flush=True)
    t0 = time.time()

    # Monomers in cc-pVDZ (with R_CISD included to prove non-extensivity detection!)
    mol_he = build_mol("He 0 0 0", basis="cc-pvdz")
    he_1 = run_closed_shell_suite(mol_he, include_fci=True, include_cisd=True, fci_conv_tol=1e-11)

    mol_h2_str = build_mol("H 0 0 0; H 0 0 3.2", basis="cc-pvdz")
    h2_str_1 = run_closed_shell_suite(mol_h2_str, include_fci=True, include_cisd=True, fci_conv_tol=1e-11)

    # Test A: He ... He at 100 A
    mol_test_a = build_mol("He 0 0 0; He 100.0 0 0", basis="cc-pvdz")
    res_a = run_closed_shell_suite(
        mol_test_a, include_fci=True, include_cisd=True, fci_nroots=2, fci_localize=True, fci_conv_tol=1e-11
    )
    err_a = {
        m: float(res_a["energies"][m] - 2.0 * he_1["energies"][m])
        for m in res_a["energies"]
        if res_a["energies"][m] is not None
    }
    assert abs(err_a["Exact_FCI"]) < 1e-8, f"Test A FCI error: {err_a['Exact_FCI']}"

    # Test B: H2(3.2 A) ... H2(3.2 A) at 100 A (Two strongly correlated stretched singlets!)
    mol_test_b = build_mol(
        "H 0 0 0; H 0 0 3.2; H 100.0 0 0; H 100.0 0 3.2", basis="cc-pvdz"
    )
    res_b = run_closed_shell_suite(
        mol_test_b, include_fci=True, include_cisd=True, fci_nroots=5, fci_localize=True, fci_conv_tol=1e-11
    )
    err_b = {
        m: float(res_b["energies"][m] - 2.0 * h2_str_1["energies"][m])
        for m in res_b["energies"]
        if res_b["energies"][m] is not None
    }
    assert abs(err_b["Exact_FCI"]) < 1e-7, f"Test B FCI error: {err_b['Exact_FCI']}"
    assert abs(err_b["R_CISD"]) > 0.05, "Expected CISD to fail Test B strongly!"

    # Test C: H2(3.2 A) ... He at 100 A (Mixed static + dynamic correlation!)
    mol_test_c = build_mol("H 0 0 0; H 0 0 3.2; He 100.0 0 0", basis="cc-pvdz")
    res_c = run_closed_shell_suite(
        mol_test_c, include_fci=True, include_cisd=True, fci_nroots=4, fci_localize=True, fci_conv_tol=1e-11
    )
    err_c = {
        m: float(res_c["energies"][m] - h2_str_1["energies"][m] - he_1["energies"][m])
        for m in res_c["energies"]
        if res_c["energies"][m] is not None
    }
    assert abs(err_c["Exact_FCI"]) < 1e-7, f"Test C FCI error: {err_c['Exact_FCI']}"

    # Test D1: Non-interacting H_{2k} array (k=1..5, R_intra=1.4 A, R_inter=100.0 A) in STO-3G (Exact FCI k=1..5!)
    print("  [6-D1] Non-interacting H_{2k} chains (k=1..5, STO-3G Exact FCI)...", flush=True)
    d1_chains = []
    e1_d1 = None
    for k in [1, 2, 3, 4, 5]:
        coords = []
        for j in range(k):
            x0 = 100.0 * j
            coords.append(f"H {x0:.4f} 0.0 0.0")
            coords.append(f"H {x0:.4f} 0.0 1.4")
        mol_k = build_mol("; ".join(coords), basis="sto-3g", symmetry=False)
        pt = run_closed_shell_suite(
            mol_k, include_fci=True, include_cisd=True, fci_nroots=2, fci_localize=(k >= 2), fci_conv_tol=1e-11
        )
        if k == 1:
            e1_d1 = pt["energies"]
        pt["k_units"] = k
        pt["energy_per_unit_ha"] = {
            m: float(v / k) for m, v in pt["energies"].items() if v is not None
        }
        pt["extensivity_deviation_per_unit_mHa"] = {
            m: float((v / k - e1_d1[m]) * 1000.0)
            for m, v in pt["energies"].items()
            if v is not None
        }
        assert abs(pt["extensivity_deviation_per_unit_mHa"]["Exact_FCI"]) < 1e-4
        d1_chains.append(pt)

    # Test D2: Interacting linear H_{2k} chains (k=1..5, R_intra=0.74 A, unit spacing a=1.80 A) in 6-31G
    print("  [6-D2] Interacting linear H_{2k} chains (k=1..5, 6-31G; FCI for k=1..4)...", flush=True)
    d2_chains = []
    for k in [1, 2, 3, 4, 5]:
        coords = []
        # Center chain around z=0 so D2h symmetry applies cleanly for FCI up to k=4 (8e, 16o)
        total_len = (k - 1) * 1.80 + 0.74
        z_shift = -0.5 * total_len
        for j in range(k):
            z0 = z_shift + 1.80 * j
            coords.append(f"H 0.0 0.0 {z0:.8f}")
            coords.append(f"H 0.0 0.0 {z0 + 0.74:.8f}")
        astr = "; ".join(coords)
        mol_c1 = build_mol(astr, basis="6-31g", symmetry=False)
        do_fci = k <= 4
        mol_sym = build_mol(astr, basis="6-31g", symmetry="D2h") if do_fci else None
        pt = run_closed_shell_suite(
            mol_c1,
            include_fci=do_fci,
            include_cisd=True,
            mol_sym_for_fci=mol_sym,
            fci_wfnsym="Ag" if do_fci else None,
        )
        pt["k_units"] = k
        pt["energy_per_unit_ha"] = {
            m: float(v / k) for m, v in pt["energies"].items() if v is not None
        }
        d2_chains.append(pt)

    print(f"  -> Section 6 completed in {time.time() - t0:.2f}s", flush=True)
    return {
        "monomers_ccpvdz": {
            "He": he_1,
            "H2_R3p2A": h2_str_1,
        },
        "Test_A_He_He_100A_ccpvdz": {
            "supersystem": res_a,
            "size_consistency_error_ha": err_a,
        },
        "Test_B_stretched_H2_dimer_100A_ccpvdz": {
            "supersystem": res_b,
            "size_consistency_error_ha": err_b,
        },
        "Test_C_stretched_H2_plus_He_100A_ccpvdz": {
            "supersystem": res_c,
            "size_consistency_error_ha": err_c,
        },
        "Test_D1_noninteracting_H2k_sto3g": d1_chains,
        "Test_D2_interacting_H2k_631g": d2_chains,
    }


# =============================================================================
# Reusable Reviewer #2 Automated Audit Entrypoint for Candidate Theories
# =============================================================================

def reconstruct_mol_from_spec(spec: Dict[str, Any]) -> gto.Mole:
    return build_mol(
        atom=spec["atom"],
        basis=spec["basis"],
        charge=spec["charge"],
        spin=spec["spin"],
        unit=spec["unit"],
        symmetry=False,
    )


def audit_proposed_theory(
    energy_fn: Callable[[gto.Mole], float],
    reference_json_path: str = OUTPUT_JSON_PATH,
) -> Dict[str, Any]:
    """
    Runs a candidate theory `energy_fn(mol) -> total_energy_hartree` across key
    Reviewer #2 torture gates using the exact geometries stored in `reference_data.json`.
    """
    with open(reference_json_path, "r", encoding="utf-8") as f:
        ref = json.load(f)

    report: Dict[str, Any] = {}

    # 1. One-electron SIE check (H atom & H2+ at R=4.0 A)
    h_spec = ref["section_1_one_electron_sie"]["H_atom"]["spec"]
    e_h = float(energy_fn(reconstruct_mol_from_spec(h_spec)))
    h2p_pt = [p for p in ref["section_1_one_electron_sie"]["H2_plus_curve"] if p["R_angstrom"] == 4.0][0]
    e_h2p = float(energy_fn(reconstruct_mol_from_spec(h2p_pt["spec"])))
    report["sie_H_atom_error_ha"] = e_h - ref["section_1_one_electron_sie"]["H_atom"]["energies"]["Exact_FCI"]
    report["sie_H2plus_4A_error_kcal"] = (e_h2p - h2p_pt["energies"]["Exact_FCI"]) * HARTREE_TO_KCAL

    # 2. Size-consistency checks A, B, C
    sc = ref["section_6_size_consistency_extensivity"]
    e_he = float(energy_fn(reconstruct_mol_from_spec(sc["monomers_ccpvdz"]["He"]["spec"])))
    e_h2s = float(energy_fn(reconstruct_mol_from_spec(sc["monomers_ccpvdz"]["H2_R3p2A"]["spec"])))
    e_ta = float(energy_fn(reconstruct_mol_from_spec(sc["Test_A_He_He_100A_ccpvdz"]["supersystem"]["spec"])))
    e_tb = float(energy_fn(reconstruct_mol_from_spec(sc["Test_B_stretched_H2_dimer_100A_ccpvdz"]["supersystem"]["spec"])))
    e_tc = float(energy_fn(reconstruct_mol_from_spec(sc["Test_C_stretched_H2_plus_He_100A_ccpvdz"]["supersystem"]["spec"])))
    report["size_consistency_A_ha"] = e_ta - 2.0 * e_he
    report["size_consistency_B_ha"] = e_tb - 2.0 * e_h2s
    report["size_consistency_C_ha"] = e_tc - e_h2s - e_he

    return report


# =============================================================================
# Main Generator Driver
# =============================================================================

def main() -> None:
    t_start = time.time()
    print("=" * 76, flush=True)
    print("QUANTUM CHEMISTRY GAUNTLET OF DEATH - REFERENCE SUITE GENERATOR", flush=True)
    print("=" * 76, flush=True)

    rebuild = "--rebuild" in sys.argv
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    if (not rebuild) and os.path.exists(OUTPUT_JSON_PATH):
        with open(OUTPUT_JSON_PATH, "r", encoding="utf-8") as f:
            dataset: Dict[str, Any] = json.load(f)
    else:
        dataset = {}

    dataset["metadata"] = {
        "generator": "gauntlet_reference.py (Reviewer #2 / referee_two)",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pyscf_version": pyscf.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "hartree_to_kcal_mol": HARTREE_TO_KCAL,
    }

    sections = [
        ("section_1_one_electron_sie", generate_section_1),
        ("section_2_two_electron_correlation", generate_section_2),
        ("section_3_h4_diradical_circle", generate_section_3),
        ("section_4_multibond_breaking", generate_section_4),
        ("section_5_equilibrium_chemistry", generate_section_5),
        ("section_6_size_consistency_extensivity", generate_section_6),
    ]

    for key, builder_fn in sections:
        if (not rebuild) and (key in dataset):
            print(f"[Checkpoint] Skipping already-completed {key}", flush=True)
            continue
        dataset[key] = builder_fn()
        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)

    elapsed = time.time() - t_start
    size_kb = os.path.getsize(OUTPUT_JSON_PATH) / 1024.0
    print("=" * 76, flush=True)
    print(f"SUCCESS: Wrote {OUTPUT_JSON_PATH} ({size_kb:.1f} KB) in {elapsed:.2f}s", flush=True)
    print("=" * 76, flush=True)


if __name__ == "__main__":
    main()
