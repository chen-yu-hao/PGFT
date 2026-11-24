"""
ROHF-Consistent Gor'kov Quasiparticles & Dual-Range Casimir-Polder Dielectric ACFD
==================================================================================

Evaluates the parameter-free, derivative-free, strictly O(N^3) PGFT v2 dynamic
correlation functional E_dyn[C, n_a, n_b] and active-subspace total-energy relaxation:
  1. Exact ROHF-consistent unshifted Fock diagonal bar_eps_{p,s} = h_pp + J_pp - K_pp[n_s]
     with effective Bogoliubov pairing gap Delta^{eff}_{p,s} = (1 - 0.5 w_{p,s}) K_pp[0.5 w_{-s}]
     and Markov P^16 subsystem-localized chemical potential mu^{loc}_{p,s},
  2. Imaginary-time Gor'kov bubble Pi_tau = 2 sum_s (G^hole o G^part - F^anom o F^anom),
  3. Schur-PSD Cauchy-Schwarz bounded pair-exclusion kernel E^{excl}_{KL} in [0, 1] using
     bounded anomalous density kt_{s,KL} = sum_p n_{p,s} w_{p,s} X_{pK} X_{pL},
  4. Dual-Range Casimir-Polder survival f_r^{v2}(iw) (via smooth Tikhonov-damped Coulomb
     metric W^{coul} = Z_s^T Z_s and pivoted-QR occupied connectivity S_{intra}) paired with
     smooth hyperbolic local self-screening-free coupling g_r^{loc}(iw).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.linalg import qr
from scipy.optimize import minimize
from pyscf import gto

from .isdf_thc import THCData
from .static_scf import (
    StaticSCFResult,
    _map_fermi_occupations,
    build_phi2_laplace_quadrature,
    compute_singlet_pair_restoration,
    pauli_amplitude_and_deriv,
    ph_pauli_amplitude_and_jac,
)


def build_laplace_cos_weights(
    omegas: np.ndarray,
    emin: float,
    emax: float,
    n_tau: int = 38,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic O(1) exponential quadrature of the Cauchy-Lorentz kernel:
        E / (omega_m^2 + E^2) ~= sum_{k=1}^{n_tau} C_{m,k} exp(-E * tau_k)
    fitted on a 300-point Chebyshev logarithmic energy grid in [emin, emax].
    """
    emin = max(float(emin), 1e-3)
    emax = max(float(emax), emin * 10.0)
    taus = np.geomspace(0.05 / emax, 18.0 / emin, n_tau)
    n_samp = 300
    k = np.arange(1, n_samp + 1, dtype=float)
    t_cheb = 0.5 * (1.0 - np.cos((2.0 * k - 1.0) * np.pi / (2.0 * n_samp)))
    E_grid = emin * ((emax / emin) ** t_cheb)

    A = np.exp(-np.clip(E_grid[:, None] * taus[None, :], 0.0, 200.0))
    T = E_grid[:, None] / (omegas[None, :] ** 2 + E_grid[:, None] ** 2)
    C_tau_w, *_ = np.linalg.lstsq(A, T, rcond=1e-12)
    return taus, C_tau_w.T


def _quasiparticle_energies_for_spin(
    eps_s: np.ndarray,
    delta_s: np.ndarray,
    n_s: np.ndarray,
    O_ovlp: np.ndarray,
) -> np.ndarray:
    """
    Compute strictly symmetric Gor'kov quasiparticle energies:
        E^{qp}_{p,s} = sqrt((eps_{p,s} - mu^{loc}_{p,s})^2 + delta_{p,s}^2) > 0
    using Markov P^16 spatial-overlap propagation on O_{pq} in O(N^3) FLOPs.
    """
    norb = len(eps_s)
    s_ent = np.clip(4.0 * n_s * (1.0 - n_s), 0.0, 1.0)

    diag_O = np.maximum(np.diag(O_ovlp), 1e-30)
    C_ovlp = np.clip(O_ovlp / np.sqrt(np.outer(diag_O, diag_O)), 0.0, 1.0)
    O_pos = np.where(C_ovlp > 1e-4, C_ovlp ** 2, 0.0) + 1e-24 * np.eye(norb)
    row_sum = np.sum(O_pos, axis=1, keepdims=True)
    W_sub = O_pos / row_sum
    for _ in range(4):
        W_sub = W_sub @ W_sub

    occ_mask = n_s > 0.5
    vir_mask = ~occ_mask

    mu_loc = np.zeros(norb, dtype=float)
    for p in range(norb):
        wp = W_sub[p, :]
        in_sub = wp > (1e-8 * float(np.max(wp)))
        occ_sub = in_sub & occ_mask
        vir_sub = in_sub & vir_mask
        if np.any(occ_sub) and np.any(vir_sub):
            mu_gap = 0.5 * (float(np.max(eps_s[occ_sub])) + float(np.min(eps_s[vir_sub])))
        elif np.any(occ_mask) and np.any(vir_mask):
            mu_gap = 0.5 * (float(np.max(eps_s[occ_mask])) + float(np.min(eps_s[vir_mask])))
        else:
            mu_gap = float(np.median(eps_s))

        s_sub = float(np.dot(wp, s_ent) / (np.sum(wp) + 1e-30)) * float(np.sum(in_sub))
        w_ent_sum = float(np.dot(wp, s_ent))
        if w_ent_sum > 1e-7:
            mu_ent = float(np.dot(wp * s_ent, eps_s)) / w_ent_sum
            blend = min(1.0, s_sub)
            mu_loc[p] = blend * mu_ent + (1.0 - blend) * mu_gap
        else:
            mu_loc[p] = mu_gap

    return np.sqrt((eps_s - mu_loc) ** 2 + delta_s ** 2)


def _occupied_connected_projector(X_mo: np.ndarray, ntot: np.ndarray) -> np.ndarray:
    """
    Construct smooth, SO(3)- and size-consistent collocation connectivity kernel S_intra(K, L) in [0, 1]
    in strictly O(N^3) FLOPs using pivoted-QR spatially localized occupied/entangled MOs.
    """
    occ_idx = np.where(ntot > 1e-4)[0]
    nk = X_mo.shape[1]
    if len(occ_idx) <= 1 or len(occ_idx) > nk:
        return np.ones((nk, nk), dtype=float)

    X_occ = X_mo[occ_idx, :].copy()
    n_occ = len(occ_idx)
    _, _, piv = qr(X_occ, pivoting=True, mode="economic")
    P_pts = X_occ[:, piv[:n_occ]]
    u_s, _, vt_s = np.linalg.svd(P_pts, full_matrices=False)
    U_loc = u_s @ vt_s
    X_loc = U_loc.T @ X_occ
    n_loc = (U_loc * U_loc).T @ ntot[occ_idx]

    Q_occ = (X_loc * X_loc) * np.maximum(n_loc[:, None], 1e-8)
    O_occ = Q_occ @ Q_occ.T
    d_o = np.maximum(np.diag(O_occ), 1e-30)
    C_occ = np.clip(O_occ / np.sqrt(np.outer(d_o, d_o)), 0.0, 1.0)

    rat = (C_occ / 0.02) ** 4
    O_step = rat / (1.0 + rat) + 1e-24 * np.eye(n_occ)
    W_m = O_step / np.sum(O_step, axis=1, keepdims=True)
    for _ in range(4):
        W_m = W_m @ W_m
    rmax = np.maximum(np.max(W_m, axis=1, keepdims=True), 1e-30)
    Rn = W_m / rmax
    W_conn_occ = 0.5 * ((Rn * Rn) / (Rn * Rn + 1e-6) + (Rn.T * Rn.T) / (Rn.T * Rn.T + 1e-6))

    Om = Q_occ / (np.sum(Q_occ, axis=0, keepdims=True) + 1e-30)
    S_intra = (Om.T @ W_conn_occ) @ Om
    return np.clip(S_intra, 0.0, 1.0)


def _build_smooth_coulomb_metric(X_ao: np.ndarray, Z: np.ndarray, eps_cond: float = 1e-2) -> np.ndarray:
    """Smooth, size-consistent Tikhonov-damped Coulomb weight W_coul = Z_s^T Z_s in O(N^3)."""
    nk = X_ao.shape[1]
    XtX_ao = X_ao.T @ X_ao
    AtA_ao = XtX_ao * XtX_ao
    AtB = AtA_ao @ Z.T
    ridge = eps_cond * np.diag(AtA_ao) + (eps_cond * eps_cond) * np.ones(nk)
    Z_s = np.linalg.solve(AtA_ao + np.diag(ridge), AtB).T
    W_coul = Z_s.T @ Z_s
    return 0.5 * (W_coul + W_coul.T)


def compute_dynamic_acfd(
    mol: gto.Mole,
    thc: THCData,
    stat: StaticSCFResult,
    n_omega: int = 28,
    n_tau: int = 38,
    amp_mode: str = "ph_pauli_s1",
    rohf_half_gap: bool = True,
    dual_range_dispersion: bool = True,
    use_sub_projector: bool = True,
    hyperbolic_alpha: float = 0.25,
) -> float:
    """
    Evaluate strictly O(N^3) derivative-free Pauli-Entanglement Dielectric ACFD energy.
    """
    C = stat.C_no
    na = stat.n_alpha
    nb = stat.n_beta
    norb = C.shape[1]
    n_alpha, n_beta = mol.nelec
    is_singlet = (mol.spin == 0)

    if not ((0 < n_alpha < norb) or (0 < n_beta < norb)):
        return 0.0

    h1_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    X_mo = C.T @ thc.X_ao
    h_diag = np.sum((C.T @ h1_ao) * C.T, axis=1)
    M_thc = thc.M_thc
    ntot = na + nb

    Q_mo = X_mo * X_mo
    J_diag = Q_mo @ (M_thc @ (Q_mo.T @ ntot))

    def k_diag_and_kl(f_vec: np.ndarray, need_diag: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        act = np.where(np.abs(f_vec) > 1e-18)[0]
        nk_loc = X_mo.shape[1]
        if len(act) == 0:
            return np.zeros(norb), np.zeros((nk_loc, nk_loc))
        Xa = X_mo[act, :]
        D_KL = (Xa.T * f_vec[act][None, :]) @ Xa
        if not need_diag:
            return np.zeros(norb), D_KL
        W_KL = M_thc * D_KL
        K_diag = np.sum((X_mo @ W_KL) * X_mo, axis=1)
        return K_diag, D_KL

    if amp_mode == "ph_pauli_s1":
        wa, _ = ph_pauli_amplitude_and_jac(na, Q_mo)
        wb = wa if is_singlet else ph_pauli_amplitude_and_jac(nb, Q_mo)[0]
    else:
        wa, _ = pauli_amplitude_and_deriv(na)
        wb = wa if is_singlet else pauli_amplitude_and_deriv(nb)[0]

    Ka_diag, gamma_a_KL = k_diag_and_kl(na, True)
    Kka_diag, kappa_a_KL = k_diag_and_kl(0.5 * wa, True)
    _, kt_a_KL = k_diag_and_kl(na * wa, False) if dual_range_dispersion else (None, kappa_a_KL)
    if is_singlet:
        Kb_diag, gamma_b_KL = Ka_diag, gamma_a_KL
        Kkb_diag, kappa_b_KL = Kka_diag, kappa_a_KL
        kt_b_KL = kt_a_KL
    else:
        Kb_diag, gamma_b_KL = k_diag_and_kl(nb, True)
        Kkb_diag, kappa_b_KL = k_diag_and_kl(0.5 * wb, True)
        _, kt_b_KL = k_diag_and_kl(nb * wb, False) if dual_range_dispersion else (None, kappa_b_KL)

    if rohf_half_gap:
        eps_a = h_diag + J_diag - Ka_diag
        eps_b = h_diag + J_diag - Kb_diag
        delta_a = (1.0 - 0.5 * wa) * np.maximum(0.0, Kkb_diag)
        delta_b = (1.0 - 0.5 * wb) * np.maximum(0.0, Kka_diag)
    else:
        eps_a = h_diag + J_diag - Ka_diag - wa * Kkb_diag
        eps_b = h_diag + J_diag - Kb_diag - wb * Kka_diag
        delta_a = np.maximum(0.0, Kkb_diag)
        delta_b = np.maximum(0.0, Kka_diag)

    O_ovlp = Q_mo @ Q_mo.T
    Eqp_a = _quasiparticle_energies_for_spin(eps_a, delta_a, na, O_ovlp)
    Eqp_b = Eqp_a if is_singlet else _quasiparticle_energies_for_spin(eps_b, delta_b, nb, O_ovlp)

    t_gl, wt_gl = np.polynomial.legendre.leggauss(n_omega)
    w0 = 0.5
    omegas = w0 * (1.0 + t_gl) / (1.0 - t_gl)
    w_omega = wt_gl * w0 * 2.0 / ((1.0 - t_gl) ** 2)
    taus, cos_mat = build_laplace_cos_weights(omegas, 2e-3, 300.0, n_tau=max(n_tau, 38))

    rho_a = np.maximum(np.diag(gamma_a_KL), 0.0)
    rho_b = rho_a if is_singlet else np.maximum(np.diag(gamma_b_KL), 0.0)
    rho_tot = rho_a + rho_b
    den_tot = np.outer(rho_tot, rho_tot)
    if dual_range_dispersion:
        excl_KL = gamma_a_KL * gamma_a_KL + gamma_b_KL * gamma_b_KL + 2.0 * (kt_a_KL * kt_b_KL)
    else:
        excl_KL = gamma_a_KL * gamma_a_KL + gamma_b_KL * gamma_b_KL + 2.0 * np.maximum(0.0, kappa_a_KL * kappa_b_KL)

    E_excl = np.ones_like(excl_KL)
    valid = den_tot > 1e-280
    E_excl[valid] = np.clip(excl_KL[valid] / den_tot[valid], 0.0, 1.0)
    f_KL = 1.0 - E_excl

    W_pos = (X_mo.T @ X_mo) ** 2
    Wf_pos = W_pos * f_KL

    has_inter_proj = True
    if dual_range_dispersion:
        W_coul = _build_smooth_coulomb_metric(thc.X_ao, thc.Z, eps_cond=1e-2)
        if use_sub_projector:
            S_intra = _occupied_connected_projector(X_mo, ntot)
            S_inter = np.maximum(0.0, 1.0 - S_intra)
            has_inter_proj = bool(np.max(S_inter) > 1e-15)
            Wc_mod = (W_coul * S_inter) if has_inter_proj else None
        else:
            Wc_mod = W_coul * E_excl
    else:
        W_coul = None
        Wc_mod = None

    kap_bogo_a = np.sqrt(np.maximum(0.0, na * (1.0 - na)))
    kap_bogo_b = kap_bogo_a if is_singlet else np.sqrt(np.maximum(0.0, nb * (1.0 - nb)))

    nk = thc.X_ao.shape[1]
    Pi_tau = np.zeros((len(taus), nk, nk), dtype=float)
    for it, tau in enumerate(taus):
        for (n_s, kb_s, Eqp_s, nel_s) in [
            (na, kap_bogo_a, Eqp_a, n_alpha),
            (nb, kap_bogo_b, Eqp_b, n_beta),
        ]:
            if 0 < nel_s < norb:
                ex = np.exp(-np.clip(Eqp_s * tau, 0.0, 150.0))
                Gh = (X_mo.T * (n_s * ex)[None, :]) @ X_mo
                Gp = (X_mo.T * ((1.0 - n_s) * ex)[None, :]) @ X_mo
                Fa = (X_mo.T * (kb_s * ex)[None, :]) @ X_mo
                Pi_tau[it] += 2.0 * (Gh * Gp - Fa * Fa)
            if is_singlet:
                Pi_tau[it] *= 2.0
                break

    Pi_iw = np.tensordot(cos_mat, Pi_tau, axes=(1, 0))
    Z = thc.Z

    ec_dyn = 0.0
    for iw in range(n_omega):
        dw = w_omega[iw] / (2.0 * np.pi)
        ZPi = Z @ Pi_iw[iw]
        Pi_pq = ZPi @ Z.T
        Pi_pq = 0.5 * (Pi_pq + Pi_pq.T)
        eig, U = np.linalg.eigh(Pi_pq)
        eig = np.maximum(eig, 0.0)
        V = ZPi.T @ U

        num_loc = np.sum((Wf_pos @ V) * V, axis=0)
        den_loc = np.sum((W_pos @ V) * V, axis=0) + 1e-30
        f_r_loc = np.where(eig > 1e-9, np.clip(num_loc / den_loc, 0.0, 1.0), 0.0)

        if hyperbolic_alpha > 0.0:
            u_r = 2.0 * f_r_loc - 1.0
            reg_r = hyperbolic_alpha * (f_r_loc * f_r_loc)
            g_r = 0.5 * (u_r + np.sqrt(u_r * u_r + reg_r * reg_r))
        else:
            g_r = np.maximum(0.0, 2.0 * f_r_loc - 1.0)

        gx = g_r * eig
        w_int = np.where(
            gx > 1e-4,
            (gx - np.log1p(gx)) / np.maximum(gx * gx, 1e-30),
            0.5 - gx / 3.0 + (gx ** 2) / 4.0 - (gx ** 3) / 5.0,
        )

        if dual_range_dispersion:
            if use_sub_projector:
                if has_inter_proj:
                    mod_num = np.sum((Wc_mod @ V) * V, axis=0)
                    coul_den = np.sum((W_coul @ V) * V, axis=0) + 1e-30
                    inter_ratio = mod_num / coul_den
                    f_r = np.where(eig > 1e-9, np.clip(f_r_loc + (1.0 - f_r_loc) * inter_ratio, 0.0, 1.0), 0.0)
                else:
                    f_r = f_r_loc
            else:
                mod_num = np.sum((Wc_mod @ V) * V, axis=0)
                coul_den = np.sum((W_coul @ V) * V, axis=0) + 1e-30
                f_r = np.where(eig > 1e-9, np.clip(1.0 - mod_num / coul_den, 0.0, 1.0), 0.0)
        else:
            f_r = f_r_loc

        ec_dyn -= dw * float(np.sum(f_r * (eig ** 2) * w_int))

    return float(ec_dyn)


def evaluate_post_scf_v2_python(
    mol: gto.Mole,
    thc: THCData,
    stat: StaticSCFResult,
    n_omega: int = 28,
    n_tau: int = 38,
    relax_active: bool = True,
    singlet_pair_order: int = 2,
    hyperbolic_alpha: float = 0.25,
    use_sub_projector: bool = True,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Pure-Python reference implementation of Unified PGFT v2 Post-Static Active-Subspace
    Relaxation + Dual-Range Casimir-Polder ACFD + Singlet Pair Restoration.
    Returns (e_stat_with_pair, e_dyn, e_tot, n_alpha_opt, n_beta_opt).
    """
    C = stat.C_no
    na0 = stat.n_alpha.copy()
    nb0 = stat.n_beta.copy()
    norb = C.shape[1]
    is_singlet = (mol.spin == 0)

    h1_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    enuc = float(mol.energy_nuc())
    X_mo = C.T @ thc.X_ao
    Q_mo = X_mo * X_mo
    O_ovlp = Q_mo @ Q_mo.T
    h_diag = np.sum((C.T @ h1_ao) * C.T, axis=1)
    M_thc = thc.M_thc
    lam_q, alpha_q = build_phi2_laplace_quadrature(28)

    def eval_stat_base(na_v: np.ndarray, nb_v: np.ndarray) -> float:
        ntot = na_v + nb_v
        J_diag = Q_mo @ (M_thc @ (Q_mo.T @ ntot))

        def k_diag(f_vec: np.ndarray) -> np.ndarray:
            act = np.where(np.abs(f_vec) > 1e-14)[0]
            if len(act) == 0:
                return np.zeros(norb)
            Xa = X_mo[act, :]
            D_KL = (Xa.T * f_vec[act][None, :]) @ Xa
            return np.sum((X_mo @ (M_thc * D_KL)) * X_mo, axis=1)

        wa, _ = ph_pauli_amplitude_and_jac(na_v, Q_mo)
        wb = wa if is_singlet else ph_pauli_amplitude_and_jac(nb_v, Q_mo)[0]
        Ka_d = k_diag(na_v)
        Kb_d = Ka_d if is_singlet else k_diag(nb_v)

        e_cum = 0.0
        if np.any(wa > 1e-13) and np.any(wb > 1e-13):
            act_a = np.where(wa > 1e-13)[0]
            Xa = X_mo[act_a, :]
            for lam_t, a_t in zip(lam_q, alpha_q):
                ua = (wa[act_a] ** 2) * np.exp(-np.clip(lam_t * (wa[act_a] ** 3), 0.0, 100.0))
                if is_singlet:
                    D_KL = (Xa.T * ua[None, :]) @ Xa
                    Kd_a = np.sum((Xa @ (M_thc * D_KL)) * Xa, axis=1)
                else:
                    act_b = np.where(wb > 1e-13)[0]
                    Xb = X_mo[act_b, :]
                    ub = (wb[act_b] ** 2) * np.exp(-np.clip(lam_t * (wb[act_b] ** 3), 0.0, 100.0))
                    D_b = (Xb.T * ub[None, :]) @ Xb
                    Kd_a = np.sum((Xa @ (M_thc * D_b)) * Xa, axis=1)
                e_cum -= a_t * float(np.dot(ua, Kd_a))

        return (
            enuc
            + float(np.dot(ntot, h_diag))
            + 0.5 * float(np.dot(ntot, J_diag))
            - 0.5 * (float(np.dot(na_v, Ka_d)) + float(np.dot(nb_v, Kb_d)))
            + e_cum
        )

    act_idx = np.where((na0 > 1e-3) & (na0 < 1.0 - 1e-3))[0] if (relax_active and is_singlet) else np.array([], dtype=int)

    na_opt = na0.copy()
    nb_opt = nb0.copy()
    if len(act_idx) >= 2:
        m_tot = len(act_idx)
        visited = [False] * m_tot
        components = []
        for s in range(m_tot):
            if visited[s]:
                continue
            comp = []
            stack = [s]
            visited[s] = True
            while stack:
                u = stack.pop()
                pu = int(act_idx[u])
                comp.append(pu)
                du = max(float(O_ovlp[pu, pu]), 1e-30)
                for v in range(m_tot):
                    if not visited[v]:
                        pv = int(act_idx[v])
                        dv = max(float(O_ovlp[pv, pv]), 1e-30)
                        if float(O_ovlp[pu, pv]) / float(np.sqrt(du * dv)) > 1e-3:
                            visited[v] = True
                            stack.append(v)
            if len(comp) >= 2:
                components.append(sorted(comp))

        for comp in components:
            comp_arr = np.array(comp, dtype=int)
            target_act = float(np.sum(na_opt[comp_arr]))
            nc_act = np.clip(na_opt[comp_arr], 1e-6, 1.0 - 1e-6)
            t_init = np.log((1.0 - nc_act) / nc_act)

            def obj(t_sub: np.ndarray) -> float:
                n_sub, _, _ = _map_fermi_occupations(t_sub, target_act)
                na_try = na_opt.copy()
                na_try[comp_arr] = n_sub
                st_try = StaticSCFResult(e_stat=0.0, C_no=C, n_alpha=na_try, n_beta=na_try, e_hf_init=stat.e_hf_init)
                es = eval_stat_base(na_try, na_try)
                ed = compute_dynamic_acfd(
                    mol, thc, st_try, n_omega=n_omega, n_tau=n_tau,
                    hyperbolic_alpha=hyperbolic_alpha, use_sub_projector=use_sub_projector,
                )
                return es + ed

            best_t = t_init.copy()
            best_val = obj(t_init)
            for scale in [1.5, 2.5, 4.0, 8.0, 16.0]:
                val = obj(t_init * scale)
                if val < best_val:
                    best_val = val
                    best_t = t_init * scale
            res = minimize(obj, best_t, method="Powell", options={"maxfev": 40, "ftol": 1e-5, "xtol": 1e-3})
            if res.fun < best_val:
                best_t = res.x
            n_sub_opt, _, _ = _map_fermi_occupations(best_t, target_act)
            na_opt[comp_arr] = n_sub_opt
            nb_opt = na_opt.copy()

    st_opt = StaticSCFResult(e_stat=0.0, C_no=C, n_alpha=na_opt, n_beta=nb_opt, e_hf_init=stat.e_hf_init)
    es_base = eval_stat_base(na_opt, nb_opt)
    e_st = compute_singlet_pair_restoration(mol, thc, st_opt, harmonic_order=singlet_pair_order)
    e_dyn = compute_dynamic_acfd(
        mol, thc, st_opt, n_omega=n_omega, n_tau=n_tau,
        hyperbolic_alpha=hyperbolic_alpha, use_sub_projector=use_sub_projector,
    )
    e_stat_full = es_base + e_st
    return e_stat_full, e_dyn, e_stat_full + e_dyn, na_opt, nb_opt
