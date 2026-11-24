"""
High-Level PGFT v2 Hybrid C++20 / Python Solver Interface
=========================================================

Defaults to the zero-copy C++20 + OpenMP + OpenBLAS/LAPACKE/Eigen3 `_pgft_cpp`
backend (`backend="auto"`) with automatic pure-Python fallback (`backend="python"`),
supporting both `variant="v2"` (default) and `variant="v1"`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pyscf import df, gto, scf

from .isdf_thc import THCData, build_isdf_thc
from .static_scf import StaticSCFResult, solve_static_scf
from .dynamic_acfd import compute_dynamic_acfd, evaluate_post_scf_v2_python

try:
    from . import _pgft_cpp as _CPP_MOD
except Exception:
    _CPP_MOD = None


@dataclass
class PGFTResult:
    """Complete output of a Pauli-Purified Gor'kov Fluctuation Theory calculation."""
    e_tot: float
    e_stat: float
    e_dyn: float
    e_hf: float
    C_no: np.ndarray
    n_alpha: np.ndarray
    n_beta: np.ndarray
    n_thc: int
    n_aux: int
    time_thc_s: float
    time_stat_s: float
    time_dyn_s: float
    time_tot_s: float
    e_singlet_pair: float = 0.0
    backend_used: str = "cpp"


def _extract_z_eff(mol: gto.Mole) -> np.ndarray:
    return np.array(
        [
            float(mol.atom_charge(ia))
            if mol.atom_charge(ia) > 0
            else float(gto.charge(mol.atom_pure_symbol(ia)))
            for ia in range(mol.natm)
        ],
        dtype=np.float64,
    )


def build_isdf_thc_fast(
    mol: gto.Mole,
    auxbasis: Optional[str | dict] = None,
    c_thc: float = 18.0,
    tol: float = 1e-11,
    n_rad: int = 28,
    use_cpp: bool = True,
) -> THCData:
    """Build deterministic SO(3)-covariant ISDF-THC tensors via C++20 or pure Python."""
    if (not use_cpp) or (_CPP_MOD is None):
        return build_isdf_thc(mol, auxbasis=auxbasis, c_thc=c_thc, tol=tol, n_rad=n_rad)

    resolved_aux = auxbasis if auxbasis is not None else df.make_auxbasis(mol)
    with_df = df.DF(mol, auxbasis=resolved_aux)
    with_df.build()

    coords_bohr = np.ascontiguousarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)
    z_eff = _extract_z_eff(mol)
    coords_raw = _CPP_MOD.generate_molecular_collocation_points(coords_bohr, z_eff, n_rad, 3e-4, 9.0)
    ao_vals = np.ascontiguousarray(mol.eval_gto("GTOval_sph", coords_raw), dtype=np.float64)
    B_tril = np.ascontiguousarray(np.vstack([blk for blk in with_df.loop()]), dtype=np.float64)

    res = _CPP_MOD.build_isdf_thc_cpp(ao_vals, coords_raw, B_tril, c_thc, tol)
    return THCData(
        X_ao=res.X_ao,
        Z=res.Z,
        M_thc=res.M_thc,
        pts=res.pts,
        auxbasis=resolved_aux,
    )


class PGFTSolver:
    """
    Pauli-Purified Gor'kov Fluctuation Theory (PGFT v2) Solver.
    Zero empirical parameters, zero xc functionals/quadratures, spin-pure, strictly O(N^3).
    """

    def __init__(
        self,
        mol: gto.Mole,
        c_thc: float = 18.0,
        n_rad: int = 28,
        n_omega: int = 28,
        n_tau: int = 38,
        auxbasis: Optional[str | dict] = None,
        thc_data: Optional[THCData] = None,
        backend: str = "auto",
        variant: str = "v2",
        relax_active: bool = True,
        singlet_pair_order: int = 2,
        use_sub_projector: bool = True,
    ) -> None:
        self.mol = mol
        self.c_thc = c_thc
        self.n_rad = n_rad
        self.n_omega = n_omega
        self.n_tau = n_tau
        self.auxbasis = auxbasis
        self._thc = thc_data
        self.backend = backend
        self.variant = variant
        self.relax_active = relax_active
        self.singlet_pair_order = singlet_pair_order
        self.use_sub_projector = use_sub_projector

    def kernel(self, mf_init: Optional[scf.hf.SCF] = None) -> PGFTResult:
        use_cpp = (self.backend in ("auto", "cpp")) and (_CPP_MOD is not None)
        if self.backend == "cpp" and _CPP_MOD is None:
            raise RuntimeError("Requested backend='cpp', but `_pgft_cpp` extension is not compiled.")

        if mf_init is None:
            mf_init = scf.RHF(self.mol) if self.mol.spin == 0 else scf.ROHF(self.mol)
            mf_init.conv_tol = 1e-10
            mf_init.max_cycle = 150
            mf_init.kernel()

        t0 = time.perf_counter()
        if self._thc is None:
            self._thc = build_isdf_thc_fast(
                self.mol,
                auxbasis=self.auxbasis,
                c_thc=self.c_thc,
                n_rad=self.n_rad,
                use_cpp=use_cpp,
            )
        t1 = time.perf_counter()

        is_v2 = (self.variant == "v2")

        if use_cpp:
            C0 = mf_init.mo_coeff
            if isinstance(C0, tuple) or C0.ndim == 3:
                C0 = C0[0]
            C0 = np.ascontiguousarray(C0, dtype=np.float64)
            mo_energy = np.ascontiguousarray(mf_init.mo_energy, dtype=np.float64)
            h1_ao = np.ascontiguousarray(
                self.mol.intor("int1e_kin") + self.mol.intor("int1e_nuc"),
                dtype=np.float64,
            )
            enuc = float(self.mol.energy_nuc())
            n_alpha, n_beta = self.mol.nelec

            scf_cfg = _CPP_MOD.StaticSCFConfig()
            scf_cfg.amp_params.mode = (
                _CPP_MOD.AmplitudeMode.PHPauliS1 if is_v2 else _CPP_MOD.AmplitudeMode.Pade13
            )

            X_ao_c = np.ascontiguousarray(self._thc.X_ao, dtype=np.float64)
            Z_c = np.ascontiguousarray(self._thc.Z, dtype=np.float64)
            M_thc_c = np.ascontiguousarray(self._thc.M_thc, dtype=np.float64)

            stat_cpp = _CPP_MOD.solve_static_scf_cpp(
                C0, h1_ao, X_ao_c, M_thc_c, enuc,
                int(n_alpha), int(n_beta), float(mf_init.e_tot),
                mo_energy, scf_cfg,
            )
            t2 = time.perf_counter()

            dyn_cfg = _CPP_MOD.DynamicACFDConfig()
            dyn_cfg.n_omega = self.n_omega
            dyn_cfg.n_tau = self.n_tau
            dyn_cfg.amp_params.mode = scf_cfg.amp_params.mode

            if is_v2:
                dyn_cfg.qp_mode = _CPP_MOD.QuasiparticleMode.ROHFHalfGap
                dyn_cfg.survival_mode = _CPP_MOD.SurvivalKernelMode.HyperbolicSelfScreeningFree
                dyn_cfg.hyperbolic_alpha = 0.25
                dyn_cfg.dual_range_dispersion = True
                dyn_cfg.use_sub_projector = self.use_sub_projector
                dyn_cfg.relax_active_subspace = self.relax_active
                dyn_cfg.singlet_pair_order = self.singlet_pair_order

                post = _CPP_MOD.evaluate_post_scf_v2_cpp(
                    stat_cpp.C_no,
                    stat_cpp.n_alpha,
                    stat_cpp.n_beta,
                    h1_ao,
                    X_ao_c,
                    Z_c,
                    M_thc_c,
                    enuc,
                    int(n_alpha),
                    int(n_beta),
                    dyn_cfg,
                )
                t3 = time.perf_counter()
                return PGFTResult(
                    e_tot=float(post.e_tot),
                    e_stat=float(post.e_stat),
                    e_dyn=float(post.e_dyn),
                    e_hf=float(stat_cpp.e_hf_init),
                    C_no=stat_cpp.C_no,
                    n_alpha=post.n_alpha,
                    n_beta=post.n_beta,
                    n_thc=int(self._thc.X_ao.shape[1]),
                    n_aux=int(self._thc.Z.shape[0]),
                    time_thc_s=t1 - t0,
                    time_stat_s=t2 - t1,
                    time_dyn_s=t3 - t2,
                    time_tot_s=t3 - t0,
                    e_singlet_pair=float(post.e_singlet_pair),
                    backend_used="cpp",
                )
            else:
                dyn_cfg.qp_mode = _CPP_MOD.QuasiparticleMode.MarkovLocalMu
                dyn_cfg.survival_mode = _CPP_MOD.SurvivalKernelMode.LinearSelfScreeningFree
                dyn_cfg.dual_range_dispersion = False
                e_dyn = float(
                    _CPP_MOD.compute_dynamic_acfd_cpp(
                        stat_cpp.C_no,
                        stat_cpp.n_alpha,
                        stat_cpp.n_beta,
                        h1_ao,
                        X_ao_c,
                        Z_c,
                        M_thc_c,
                        int(n_alpha),
                        int(n_beta),
                        dyn_cfg,
                    )
                )
                t3 = time.perf_counter()
                return PGFTResult(
                    e_tot=float(stat_cpp.e_stat) + e_dyn,
                    e_stat=float(stat_cpp.e_stat),
                    e_dyn=e_dyn,
                    e_hf=float(stat_cpp.e_hf_init),
                    C_no=stat_cpp.C_no,
                    n_alpha=stat_cpp.n_alpha,
                    n_beta=stat_cpp.n_beta,
                    n_thc=int(self._thc.X_ao.shape[1]),
                    n_aux=int(self._thc.Z.shape[0]),
                    time_thc_s=t1 - t0,
                    time_stat_s=t2 - t1,
                    time_dyn_s=t3 - t2,
                    time_tot_s=t3 - t0,
                    e_singlet_pair=0.0,
                    backend_used="cpp",
                )

        # Pure-Python reference backend
        amp_mode = "ph_pauli_s1" if is_v2 else "pade13"
        stat: StaticSCFResult = solve_static_scf(
            self.mol, self._thc, mf_init=mf_init, amp_mode=amp_mode
        )
        t2 = time.perf_counter()

        if is_v2:
            e_stat_full, e_dyn, e_tot, na_opt, nb_opt = evaluate_post_scf_v2_python(
                self.mol,
                self._thc,
                stat,
                n_omega=self.n_omega,
                n_tau=self.n_tau,
                relax_active=self.relax_active,
                singlet_pair_order=self.singlet_pair_order,
                use_sub_projector=self.use_sub_projector,
            )
            t3 = time.perf_counter()
            return PGFTResult(
                e_tot=e_tot,
                e_stat=e_stat_full,
                e_dyn=e_dyn,
                e_hf=stat.e_hf_init,
                C_no=stat.C_no,
                n_alpha=na_opt,
                n_beta=nb_opt,
                n_thc=int(self._thc.X_ao.shape[1]),
                n_aux=int(self._thc.Z.shape[0]),
                time_thc_s=t1 - t0,
                time_stat_s=t2 - t1,
                time_dyn_s=t3 - t2,
                time_tot_s=t3 - t0,
                e_singlet_pair=0.0,
                backend_used="python",
            )
        else:
            e_dyn = compute_dynamic_acfd(
                self.mol,
                self._thc,
                stat,
                n_omega=self.n_omega,
                n_tau=self.n_tau,
                amp_mode="pade13",
                rohf_half_gap=False,
                dual_range_dispersion=False,
                hyperbolic_alpha=0.0,
            )
            t3 = time.perf_counter()
            return PGFTResult(
                e_tot=stat.e_stat + e_dyn,
                e_stat=stat.e_stat,
                e_dyn=e_dyn,
                e_hf=stat.e_hf_init,
                C_no=stat.C_no,
                n_alpha=stat.n_alpha,
                n_beta=stat.n_beta,
                n_thc=int(self._thc.X_ao.shape[1]),
                n_aux=int(self._thc.Z.shape[0]),
                time_thc_s=t1 - t0,
                time_stat_s=t2 - t1,
                time_dyn_s=t3 - t2,
                time_tot_s=t3 - t0,
                e_singlet_pair=0.0,
                backend_used="python",
            )


def run_pgft(mol: gto.Mole, mf_init: Optional[scf.hf.SCF] = None, **kwargs) -> PGFTResult:
    """Convenience one-line runner for PGFT v2 (C++20 accelerated by default)."""
    return PGFTSolver(mol, **kwargs).kernel(mf_init=mf_init)
