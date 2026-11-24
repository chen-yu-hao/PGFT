"""
Pauli-Purified Gor'kov Fluctuation Theory (PGFT v2)
===================================================

A first-principles, parameter-free, spin-pure, strictly O(N^3) C++20/PySCF electronic
structure package unifying multireference static and single-reference dynamic correlation
completely outside Kohn-Sham density functional theory.
"""

from .isdf_thc import THCData, build_isdf_thc
from .static_scf import (
    StaticSCFResult,
    compute_singlet_pair_restoration,
    pauli_amplitude_and_deriv,
    ph_pauli_amplitude_and_jac,
    solve_static_scf,
)
from .dynamic_acfd import compute_dynamic_acfd, evaluate_post_scf_v2_python
from .solver import PGFTResult, PGFTSolver, build_isdf_thc_fast, run_pgft

__version__ = "2.0.0"

__all__ = [
    "THCData",
    "build_isdf_thc",
    "build_isdf_thc_fast",
    "StaticSCFResult",
    "solve_static_scf",
    "pauli_amplitude_and_deriv",
    "ph_pauli_amplitude_and_jac",
    "compute_singlet_pair_restoration",
    "compute_dynamic_acfd",
    "evaluate_post_scf_v2_python",
    "PGFTResult",
    "PGFTSolver",
    "run_pgft",
]
