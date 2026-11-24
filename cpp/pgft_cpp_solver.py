"""
Zero-Copy High-Level Python Wrapper Around C++20 `_pgft_cpp` Core Engine
========================================================================
Drop-in `PGFTCppSolver` / `run_pgft_cpp` backed by `libpgft_core`
(C++20 + OpenMP + OpenBLAS/LAPACKE/Eigen3), supporting both `PGFT v2`
(default) and `PGFT v1`.
"""

from __future__ import annotations

from typing import Optional

from pyscf import gto, scf

from pgft.isdf_thc import THCData
from pgft.solver import PGFTResult, PGFTSolver, build_isdf_thc_fast


def build_isdf_thc_cpp(
    mol: gto.Mole,
    auxbasis: Optional[str | dict] = None,
    c_thc: float = 18.0,
    tol: float = 1e-11,
    n_rad: int = 28,
) -> THCData:
    """Build deterministic SO(3)-covariant ISDF-THC tensors via C++20 `_pgft_cpp`."""
    return build_isdf_thc_fast(
        mol,
        auxbasis=auxbasis,
        c_thc=c_thc,
        tol=tol,
        n_rad=n_rad,
        use_cpp=True,
    )


class PGFTCppSolver(PGFTSolver):
    """Explicit C++20-enforced `PGFTSolver` (`backend='cpp'`)."""

    def __init__(self, mol: gto.Mole, **kwargs) -> None:
        kwargs.setdefault("backend", "cpp")
        kwargs.setdefault("variant", "v2")
        super().__init__(mol, **kwargs)


def run_pgft_cpp(mol: gto.Mole, mf_init: Optional[scf.hf.SCF] = None, **kwargs) -> PGFTResult:
    """One-line runner for C++20-accelerated PGFT v2."""
    kwargs.setdefault("backend", "cpp")
    return PGFTSolver(mol, **kwargs).kernel(mf_init=mf_init)
