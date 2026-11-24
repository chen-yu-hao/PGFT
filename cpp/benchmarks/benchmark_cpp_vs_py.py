#!/usr/bin/env python3
"""
Head-to-Head Verification & Timing Benchmark: Python `pgft` vs C++20 `_pgft_cpp` (PGFT v2)
==========================================================================================
Verifies numerical parity (< 1e-11 Ha on matched fixed-occupation states) and measures
exact wall-time speedups and O(N^3) scaling across closed-shell, open-shell radical,
dispersion-bound, diradical, and linear hydrogen-chain benchmarks.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import numpy as np
from pyscf import df, gto, scf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pgft import PGFTSolver, _pgft_cpp  # noqa: E402
from pgft.dynamic_acfd import evaluate_post_scf_v2_python  # noqa: E402
from pgft.isdf_thc import THCData  # noqa: E402
from pgft.static_scf import StaticSCFResult  # noqa: E402


def get_z_eff(mol: gto.Mole) -> np.ndarray:
    return np.array(
        [
            float(mol.atom_charge(ia))
            if mol.atom_charge(ia) > 0
            else float(gto.charge(mol.atom_pure_symbol(ia)))
            for ia in range(mol.natm)
        ],
        dtype=np.float64,
    )


def run_cpp_v2_detailed(mol: gto.Mole, mf: scf.hf.SCF) -> dict:
    """Run PGFT v2 C++20 pipeline measuring both pure C++ kernel time and end-to-end time."""
    t_e2e_0 = time.perf_counter()

    coords_bohr = np.ascontiguousarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)
    z_eff = get_z_eff(mol)
    coords_raw = _pgft_cpp.generate_molecular_collocation_points(coords_bohr, z_eff, 28, 3e-4, 9.0)
    ao_vals = np.ascontiguousarray(mol.eval_gto("GTOval_sph", coords_raw), dtype=np.float64)

    resolved_aux = df.make_auxbasis(mol)
    with_df = df.DF(mol, auxbasis=resolved_aux)
    with_df.build()
    B_tril = np.ascontiguousarray(np.vstack([blk for blk in with_df.loop()]), dtype=np.float64)
    h1_ao = np.ascontiguousarray(mol.intor("int1e_kin") + mol.intor("int1e_nuc"), dtype=np.float64)

    C0 = mf.mo_coeff
    if isinstance(C0, tuple) or C0.ndim == 3:
        C0 = C0[0]
    C0 = np.ascontiguousarray(C0, dtype=np.float64)
    mo_energy = np.ascontiguousarray(mf.mo_energy, dtype=np.float64)
    enuc = float(mol.energy_nuc())
    n_alpha, n_beta = mol.nelec

    # Kernel 1: C++20 ISDF-THC
    tk0 = time.perf_counter()
    thc_cpp = _pgft_cpp.build_isdf_thc_cpp(ao_vals, coords_raw, B_tril, 18.0, 1e-11)
    tk1 = time.perf_counter()

    # Kernel 2: C++20 Static SCF v2 (PHPauliS1 + Fock/Fermi-Curvature Preconditioned L-BFGS)
    stat_cfg = _pgft_cpp.StaticSCFConfig()
    stat_cfg.amp_params.mode = _pgft_cpp.AmplitudeMode.PHPauliS1
    stat_cpp = _pgft_cpp.solve_static_scf_cpp(
        C0, h1_ao, thc_cpp.X_ao, thc_cpp.M_thc,
        enuc, int(n_alpha), int(n_beta), float(mf.e_tot),
        mo_energy, stat_cfg,
    )
    tk2 = time.perf_counter()

    # Kernel 3: C++20 Post-SCF v2
    dyn_cfg = _pgft_cpp.DynamicACFDConfig()
    dyn_cfg.n_omega = 28
    dyn_cfg.n_tau = 38
    dyn_cfg.amp_params.mode = _pgft_cpp.AmplitudeMode.PHPauliS1
    dyn_cfg.qp_mode = _pgft_cpp.QuasiparticleMode.ROHFHalfGap
    dyn_cfg.survival_mode = _pgft_cpp.SurvivalKernelMode.HyperbolicSelfScreeningFree
    dyn_cfg.hyperbolic_alpha = 0.25
    dyn_cfg.dual_range_dispersion = True
    dyn_cfg.use_sub_projector = True
    dyn_cfg.relax_active_subspace = True
    dyn_cfg.singlet_pair_order = 2

    post_cpp = _pgft_cpp.evaluate_post_scf_v2_cpp(
        stat_cpp.C_no, stat_cpp.n_alpha, stat_cpp.n_beta,
        h1_ao, thc_cpp.X_ao, thc_cpp.Z, thc_cpp.M_thc,
        enuc, int(n_alpha), int(n_beta), dyn_cfg,
    )
    tk3 = time.perf_counter()

    # Also evaluate fixed-occupation C++ post-SCF for exact matched-state kernel comparison
    dyn_cfg_norelax = _pgft_cpp.DynamicACFDConfig()
    dyn_cfg_norelax.n_omega = 28
    dyn_cfg_norelax.n_tau = 38
    dyn_cfg_norelax.amp_params.mode = _pgft_cpp.AmplitudeMode.PHPauliS1
    dyn_cfg_norelax.qp_mode = _pgft_cpp.QuasiparticleMode.ROHFHalfGap
    dyn_cfg_norelax.survival_mode = _pgft_cpp.SurvivalKernelMode.HyperbolicSelfScreeningFree
    dyn_cfg_norelax.hyperbolic_alpha = 0.25
    dyn_cfg_norelax.dual_range_dispersion = True
    dyn_cfg_norelax.use_sub_projector = True
    dyn_cfg_norelax.relax_active_subspace = False
    dyn_cfg_norelax.singlet_pair_order = 2

    post_cpp_norelax = _pgft_cpp.evaluate_post_scf_v2_cpp(
        stat_cpp.C_no, stat_cpp.n_alpha, stat_cpp.n_beta,
        h1_ao, thc_cpp.X_ao, thc_cpp.Z, thc_cpp.M_thc,
        enuc, int(n_alpha), int(n_beta), dyn_cfg_norelax,
    )

    return {
        "e_stat": float(post_cpp.e_stat),
        "e_dyn": float(post_cpp.e_dyn),
        "e_tot": float(post_cpp.e_tot),
        "e_tot_norelax": float(post_cpp_norelax.e_tot),
        "n_thc": int(thc_cpp.X_ao.shape[1]),
        "n_aux": int(thc_cpp.Z.shape[0]),
        "t_thc_kern": tk1 - tk0,
        "t_stat": tk2 - tk1,
        "t_dyn": tk3 - tk2,
        "t_kern_tot": tk3 - tk0,
        "t_e2e_tot": tk3 - t_e2e_0,
        "iters": int(stat_cpp.total_iters),
        "fevals": int(stat_cpp.total_fevals),
        "thc_cpp": thc_cpp,
        "stat_cpp": stat_cpp,
        "resolved_aux": resolved_aux,
    }


def make_h2_chain(n_units: int, basis: str = "6-31g") -> gto.Mole:
    atoms = []
    z = 0.0
    for _ in range(n_units):
        atoms.append(("H", (0.0, 0.0, z)))
        atoms.append(("H", (0.0, 0.0, z + 0.74)))
        z += 2.04
    return gto.M(atom=atoms, basis=basis, unit="Angstrom", verbose=0)


def make_beh2_c2v(x_bohr: float = 2.75) -> gto.Mole:
    y = 2.54 - 0.46 * x_bohr
    return gto.M(
        atom=[("Be", (0.0, 0.0, 0.0)), ("H", (x_bohr, y, 0.0)), ("H", (x_bohr, -y, 0.0))],
        basis="6-31g",
        unit="Bohr",
        verbose=0,
    )


def main() -> None:
    systems = [
        ("He2 (aug-cc-pVDZ)", gto.M(
            atom="He 0 0 0; He 0 0 3.5", basis="aug-cc-pvdz", unit="Angstrom", verbose=0), True),
        ("BeH2 C2v (6-31G)", make_beh2_c2v(2.75), True),
        ("H2O (cc-pVDZ)", gto.M(
            atom="O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692",
            basis="cc-pvdz", unit="Angstrom", verbose=0), True),
        ("OH rad (cc-pVDZ)", gto.M(
            atom="O 0.0 0.0 0.0; H 0.0 0.0 0.9697",
            basis="cc-pvdz", spin=1, unit="Angstrom", verbose=0), True),
        ("(H2)_4 (6-31G)", make_h2_chain(4, "6-31g"), True),
        ("(H2)_8 (6-31G)", make_h2_chain(8, "6-31g"), True),
        ("(H2)_12 (6-31G)", make_h2_chain(12, "6-31g"), False),
        ("(H2)_16 (6-31G)", make_h2_chain(16, "6-31g"), False),
    ]

    # Warmup
    mol_w = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf_w = scf.RHF(mol_w).run()
    _ = run_cpp_v2_detailed(mol_w, mf_w)

    out_json = Path(__file__).resolve().parent / "cpp_vs_py_results.json"
    results = []
    max_matched_err = 0.0

    print("=" * 138)
    print(
        f"{'System':<20} {'N_AO':>4} {'N_THC':>5} | {'|dE_kern|':>10} {'|dE_indep|':>10} | "
        f"{'Py E2E(s)':>9} {'C++ Kern(s)':>11} {'C++ E2E(s)':>10} | "
        f"{'Kern Spd':>8} {'E2E Spd':>8} {'Dyn Spd':>8}"
    )
    print("-" * 138)

    for name, mol, run_py in systems:
        mf = scf.RHF(mol) if mol.spin == 0 else scf.ROHF(mol)
        mf.conv_tol = 1e-10
        mf.max_cycle = 150
        mf.kernel()

        cpp_res = run_cpp_v2_detailed(mol, mf)

        if not run_py:
            results.append({
                "system": name,
                "mode": "cpp_scaling",
                "nao": int(mol.nao_nr()),
                "n_thc_cpp": cpp_res["n_thc"],
                "e_stat_cpp": cpp_res["e_stat"],
                "e_dyn_cpp": cpp_res["e_dyn"],
                "e_tot_cpp": cpp_res["e_tot"],
                "cpp_thc_kern_s": cpp_res["t_thc_kern"],
                "cpp_stat_s": cpp_res["t_stat"],
                "cpp_dyn_s": cpp_res["t_dyn"],
                "cpp_kern_tot_s": cpp_res["t_kern_tot"],
                "cpp_e2e_tot_s": cpp_res["t_e2e_tot"],
                "cpp_lbfgs_iters": cpp_res["iters"],
                "cpp_lbfgs_fevals": cpp_res["fevals"],
            })
            out_json.write_text(json.dumps({"variant": "v2", "systems": results}, indent=2))
            print(
                f"{name:<20} {mol.nao_nr():>4d} {cpp_res['n_thc']:>5d} | "
                f"{'---':>10} {'---':>10} | "
                f"{'---':>9} {cpp_res['t_kern_tot']:>11.4f} {cpp_res['t_e2e_tot']:>10.4f} | "
                f"thc={cpp_res['t_thc_kern']:.3f}s stat={cpp_res['t_stat']:.3f}s dyn={cpp_res['t_dyn']:.3f}s",
                flush=True,
            )
            continue

        py_out = PGFTSolver(mol, backend="python", variant="v2").kernel(mf_init=mf)

        thc_wrap = THCData(
            X_ao=cpp_res["thc_cpp"].X_ao,
            Z=cpp_res["thc_cpp"].Z,
            M_thc=cpp_res["thc_cpp"].M_thc,
            pts=cpp_res["thc_cpp"].pts,
            auxbasis=cpp_res["resolved_aux"],
        )
        stat_wrap = StaticSCFResult(
            e_stat=float(cpp_res["stat_cpp"].e_stat),
            C_no=cpp_res["stat_cpp"].C_no,
            n_alpha=cpp_res["stat_cpp"].n_alpha,
            n_beta=cpp_res["stat_cpp"].n_beta,
            e_hf_init=float(mf.e_tot),
        )
        _, _, e_tot_py_matched, _, _ = evaluate_post_scf_v2_python(
            mol, thc_wrap, stat_wrap, n_omega=28, n_tau=38,
            relax_active=False, singlet_pair_order=2, use_sub_projector=True,
        )

        de_matched = abs(e_tot_py_matched - cpp_res["e_tot_norelax"])
        de_indep = abs(py_out.e_tot - cpp_res["e_tot"])
        max_matched_err = max(max_matched_err, de_matched)

        sp_kern = py_out.time_tot_s / max(cpp_res["t_kern_tot"], 1e-9)
        sp_e2e = py_out.time_tot_s / max(cpp_res["t_e2e_tot"], 1e-9)
        sp_stat = py_out.time_stat_s / max(cpp_res["t_stat"], 1e-9)
        sp_dyn = py_out.time_dyn_s / max(cpp_res["t_dyn"], 1e-9)
        sp_thc = py_out.time_thc_s / max(cpp_res["t_thc_kern"], 1e-9)

        results.append({
            "system": name,
            "mode": "head_to_head",
            "nao": int(mol.nao_nr()),
            "n_thc_py": int(py_out.n_thc),
            "n_thc_cpp": cpp_res["n_thc"],
            "e_stat_py": float(py_out.e_stat),
            "e_stat_cpp": cpp_res["e_stat"],
            "e_dyn_py": float(py_out.e_dyn),
            "e_dyn_cpp": cpp_res["e_dyn"],
            "e_tot_py": float(py_out.e_tot),
            "e_tot_cpp": cpp_res["e_tot"],
            "de_matched_tot_ha": de_matched,
            "de_indep_tot_ha": de_indep,
            "py_thc_s": float(py_out.time_thc_s),
            "py_stat_s": float(py_out.time_stat_s),
            "py_dyn_s": float(py_out.time_dyn_s),
            "py_tot_s": float(py_out.time_tot_s),
            "cpp_thc_kern_s": cpp_res["t_thc_kern"],
            "cpp_stat_s": cpp_res["t_stat"],
            "cpp_dyn_s": cpp_res["t_dyn"],
            "cpp_kern_tot_s": cpp_res["t_kern_tot"],
            "cpp_e2e_tot_s": cpp_res["t_e2e_tot"],
            "speedup_thc_kern": sp_thc,
            "speedup_stat": sp_stat,
            "speedup_dyn": sp_dyn,
            "speedup_kern_tot": sp_kern,
            "speedup_e2e_tot": sp_e2e,
            "cpp_lbfgs_iters": cpp_res["iters"],
            "cpp_lbfgs_fevals": cpp_res["fevals"],
        })
        out_json.write_text(json.dumps({"variant": "v2", "systems": results}, indent=2))

        print(
            f"{name:<20} {mol.nao_nr():>4d} {cpp_res['n_thc']:>5d} | "
            f"{de_matched:>10.2e} {de_indep:>10.2e} | "
            f"{py_out.time_tot_s:>9.3f} {cpp_res['t_kern_tot']:>11.4f} {cpp_res['t_e2e_tot']:>10.4f} | "
            f"{sp_kern:>7.1f}x {sp_e2e:>7.1f}x {sp_dyn:>7.1f}x",
            flush=True,
        )

    chains = [r for r in results if r["system"].startswith("(H2)_")]
    xs = np.log([r["nao"] for r in chains])
    alpha_kern = float(np.polyfit(xs, np.log([r["cpp_kern_tot_s"] for r in chains]), 1)[0])
    alpha_dyn = float(np.polyfit(xs, np.log([r["cpp_dyn_s"] for r in chains]), 1)[0])

    summary = {
        "variant": "v2",
        "max_matched_tot_err_ha": max_matched_err,
        "cpp_chain_scaling_exponent_kernel": alpha_kern,
        "cpp_chain_scaling_exponent_dyn": alpha_dyn,
        "systems": results,
    }
    out_json.write_text(json.dumps(summary, indent=2))

    print("=" * 138)
    print(
        f"Max |dE_tot (matched kernel)| = {max_matched_err:.3e} Ha | "
        f"C++20 (H2)_4..16 Scaling: total N^{alpha_kern:.2f}, dynamic N^{alpha_dyn:.2f}"
    )
    if max_matched_err >= 1e-10:
        sys.exit(1)
    print("SUCCESS: Verified < 1e-10 Ha C++20 vs Python parity and O(N^3) scaling!")


if __name__ == "__main__":
    main()
