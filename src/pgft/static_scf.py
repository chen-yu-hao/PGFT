"""
Particle-Hole Pauli-Protected Generalized Harmonic Static Cumulant Solver (O(N^3))
==================================================================================

Minimizes the parameter-free, spin-pure static energy functional:
    E_{0+stat}[C, n_a, n_b] = E_nuc + sum_{p,s} n_{p,s} h_{pp}
        + 1/2 sum_{pq,st} n_{p,s} n_{q,t} J_{pq}
        - 1/2 sum_{pq,s} n_{p,s} n_{q,s} K_{pq}
        - sum_{pq} Phi_2(w_{p,a}, w_{q,b}) K_{pq}
where in PGFT v2 (`amp_mode="ph_pauli_s1"`, default):
    s_{p,s} = 4 n_{p,s} (1 - n_{p,s}) in [0, 1],
    w_{p,s}(n_s, Q_mo) = s_{p,s} - (1 - g^{PH}_{p,s}) Delta^+_{p,s} in [0, 1],
enforcing Pauli's pigeonhole N-representability bound Gamma^{ab}_{pp,pp} >= max(0, 2n - 1)
on uncompensated lone pairs (2p^4/2p^5) while providing smooth concave 2nd-order recoupling
across spatially overlapping covalent bonding/antibonding pairs, with exact reverse-mode AD
analytical gradients in strictly O(N^3) FLOPs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.linalg import qr
from scipy.optimize import minimize
from pyscf import gto, scf

from .isdf_thc import THCData


@dataclass
class StaticSCFResult:
    """Converged natural orbitals, occupations, and static energy."""
    e_stat: float
    C_no: np.ndarray      # (nao, norb) semi-canonical natural orbitals
    n_alpha: np.ndarray   # (norb,) alpha natural occupations in [0, 1]
    n_beta: np.ndarray    # (norb,) beta natural occupations in [0, 1]
    e_hf_init: float      # Initial RHF/ROHF reference energy


def build_phi2_laplace_quadrature(n_quad: int = 28, L_exp: float = 13.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Legendre log-Laplace quadrature for the m=2 generalized harmonic mean:
        Phi_2(a, b) = 0.5 * a^2 b^2 / (a^3 + b^3) ~= sum_{t=1}^{N_s} alpha_t u_t(a) u_t(b),
        u_t(a) = a^2 * exp(-lam_t * a^3).
    """
    u, wu = np.polynomial.legendre.leggauss(n_quad)
    lam = np.exp(L_exp * u)
    alpha = 0.5 * L_exp * wu * lam
    val_11 = float(np.sum(alpha * np.exp(-2.0 * lam)))
    alpha *= 0.25 / val_11
    return lam, alpha


def pauli_amplitude_and_deriv(n_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameter-free (1,3)-Pade univariate Pauli-bounded entanglement amplitude (PGFT v1):
        s = 4 n (1 - n) in [0, 1],
        w(n) = s / (2 - s^3) in [0, 1].
    """
    nc = np.clip(n_vec, 0.0, 1.0)
    s = 4.0 * nc * (1.0 - nc)
    ds_dn = 4.0 * (1.0 - 2.0 * nc)
    den = 2.0 - (s ** 3)
    w = s / den
    dw_dn = ds_dn * (2.0 + 2.0 * (s ** 3)) / (den * den)
    return w, dw_dn


def ph_pauli_amplitude_and_jac(
    n_vec: np.ndarray,
    Q_mo: np.ndarray,
    eta: float = 1e-3,
) -> Tuple[np.ndarray, Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    """
    Parameter-free C^inf Particle-Hole Overlap-Gated Pauli-Pigeonhole Amplitude (PGFT v2):
        w_p(n, Q_mo) = s_p - (1 - g^{PH}_p) Delta^+_p,
    evaluated in O(N_MO * N_K) = O(N^2) FLOPs, returning (w, backprop_fn).
    """
    nc = np.clip(n_vec, 0.0, 1.0)
    B = 4.0 * nc * (1.0 - nc)
    dB = 4.0 * (1.0 - 2.0 * nc)

    u = 2.0 * nc - 1.0
    ru = np.sqrt(u * u + eta * eta)
    s1c = float(np.sqrt(1.0 + eta * eta))
    den_m = s1c - eta
    m_min = 0.5 * (s1c - ru) / den_m
    dm_min = -(u / ru) / den_m

    p_bnd = 4.0 * (1.0 - nc) * m_min
    dp_bnd = -4.0 * m_min + 4.0 * (1.0 - nc) * dm_min
    diff = B - p_bnd
    ddiff = dB - dp_bnd
    r_diff = np.sqrt(diff * diff + 1e-8)
    exc_plus = np.maximum(0.0, 0.5 * (diff + r_diff - 1e-4))
    dexc_plus = np.where(exc_plus > 0.0, 0.5 * (1.0 + diff / r_diff) * ddiff, 0.0)

    p_bnd_m = 4.0 * nc * m_min
    dp_bnd_m = 4.0 * m_min + 4.0 * nc * dm_min
    diff_m = B - p_bnd_m
    ddiff_m = dB - dp_bnd_m
    r_diff_m = np.sqrt(diff_m * diff_m + 1e-8)
    exc_minus = np.maximum(0.0, 0.5 * (diff_m + r_diff_m - 1e-4))
    dexc_minus = np.where(exc_minus > 0.0, 0.5 * (1.0 + diff_m / r_diff_m) * ddiff_m, 0.0)

    rho_p = Q_mo.T @ exc_plus
    rho_m = Q_mo.T @ exc_minus
    Op = Q_mo @ rho_p
    Om = Q_mo @ rho_m

    a_p = Op * Op
    b_p = 4.0 * Om * Om
    den_g = a_p + b_p + 1e-24
    gate = b_p / den_g
    dgate_dOp = -2.0 * Op * b_p / (den_g * den_g)
    dgate_dOm = 8.0 * Om * a_p / (den_g * den_g)

    w = B - (1.0 - gate) * exc_plus

    def backprop(dE_dw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dE_dn = dE_dw * (dB - (1.0 - gate) * dexc_plus)
        dE_dgate = dE_dw * exc_plus
        v_Op = dE_dgate * dgate_dOp
        v_Om = dE_dgate * dgate_dOm
        q_vOp = Q_mo.T @ v_Op
        q_vOm = Q_mo.T @ v_Om
        dE_dn = dE_dn + (Q_mo @ q_vOp) * dexc_plus + (Q_mo @ q_vOm) * dexc_minus
        dE_dQ = (
            v_Op[:, None] * rho_p[None, :]
            + exc_plus[:, None] * q_vOp[None, :]
            + v_Om[:, None] * rho_m[None, :]
            + exc_minus[:, None] * q_vOm[None, :]
        )
        return dE_dn, dE_dQ

    return w, backprop


def compute_singlet_pair_restoration(
    mol: gto.Mole,
    thc: THCData,
    stat: StaticSCFResult,
    harmonic_order: int = 2,
    amp_mode: str = "ph_pauli_s1",
) -> float:
    """
    Evaluate parameter-free, strictly O(N^3), O(2)- & SO(3)-invariant, zero-monopole,
    Clebsch-Gordan monogamy-gated Singlet Pair-Exchange Restoration E_{S-T}[C_no, n].
    """
    if mol.spin != 0 or harmonic_order <= 0:
        return 0.0
    na = stat.n_alpha
    X_mo = stat.C_no.T @ thc.X_ao
    Q_mo = X_mo * X_mo

    if amp_mode == "ph_pauli_s1":
        wa, _ = ph_pauli_amplitude_and_jac(na, Q_mo)
    else:
        wa, _ = pauli_amplitude_and_deriv(na)

    eta = 1e-3
    s1c = float(np.sqrt(1.0 + eta * eta))
    nc = np.clip(na, 0.0, 1.0)
    u = 2.0 * nc - 1.0
    nu_vec = np.clip((s1c - np.sqrt(u * u + eta * eta)) / (s1c - eta), 0.0, 1.0)

    act = np.where(wa > 1e-6)[0]
    if len(act) < 2:
        return 0.0

    X_act = X_mo[act, :].copy()
    wa_act = wa[act].copy()
    nu_act = nu_vec[act].copy()
    n_act = len(act)

    order = np.argsort(-wa_act, kind="mergesort")
    X_act = X_act[order]
    wa_act = wa_act[order]
    nu_act = nu_act[order]

    i0 = 0
    while i0 < n_act:
        i1 = i0 + 1
        while i1 < n_act and abs(float(wa_act[i1] - wa_act[i0])) < 0.15:
            i1 += 1
        m = i1 - i0
        if m > 1 and m <= X_act.shape[1]:
            X_sub = X_act[i0:i1, :]
            _, _, piv = qr(X_sub, pivoting=True, mode="economic")
            P_pts = X_sub[:, piv[:m]]
            u_s, _, vt_s = np.linalg.svd(P_pts, full_matrices=False)
            U_loc = u_s @ vt_s
            X_act[i0:i1, :] = U_loc.T @ X_sub
            U2 = U_loc * U_loc
            wa_act[i0:i1] = U2.T @ wa_act[i0:i1]
            nu_act[i0:i1] = U2.T @ nu_act[i0:i1]
        i0 = i1

    Q_act = X_act * X_act
    O_act = Q_act @ Q_act.T
    d_o = np.maximum(np.diag(O_act), 1e-30)
    C_o = np.clip(O_act / np.sqrt(np.outer(d_o, d_o)), 0.0, 1.0)
    O_p = np.where(C_o > 1e-4, C_o * C_o, 0.0) + 1e-24 * np.eye(n_act)
    W_s = O_p / np.sum(O_p, axis=1, keepdims=True)
    for _ in range(4):
        W_s = W_s @ W_s
    rmax = np.maximum(np.max(W_s, axis=1, keepdims=True), 1e-30)
    Rn = W_s / rmax
    W_conn = 0.5 * ((Rn * Rn) / (Rn * Rn + 1e-8) + (Rn.T * Rn.T) / (Rn.T * Rn.T + 1e-8))

    M_thc = thc.M_thc
    MQ_act = Q_act @ M_thc

    e_st = 0.0
    inv_rt2 = float(1.0 / np.sqrt(2.0))
    rt2 = float(np.sqrt(2.0))
    for i in range(n_act):
        for j in range(i + 1, n_act):
            w_pq = float(wa_act[i] * wa_act[j] * W_conn[i, j])
            if w_pq < 1e-10:
                continue
            mono = 1.0
            for k in range(n_act):
                if k != i and k != j:
                    mono *= max(0.0, 1.0 - 0.5 * float(W_conn[i, k] + W_conn[j, k]) * float(nu_act[k]))
            weight = w_pq * mono
            if weight < 1e-10:
                continue

            d1 = (Q_act[i] - Q_act[j]) * inv_rt2
            d1_Mq = (MQ_act[i] - MQ_act[j]) * inv_rt2
            d2 = rt2 * (X_act[i] * X_act[j])
            Md2 = M_thc @ d2

            G11 = max(0.0, float(np.dot(d1_Mq, d1)))
            G12 = float(np.dot(d1_Mq, d2))
            G22 = max(0.0, float(np.dot(d2, Md2)))
            tr = G11 + G22
            det = max(0.0, G11 * G22 - G12 * G12)
            if tr > 1e-18:
                gap = (det / tr) if harmonic_order == 1 else (8.0 * (det * det) / (tr * tr * tr))
                e_st += weight * gap
    return float(e_st)


def _map_fermi_occupations(t_vec: np.ndarray, target_n: float) -> Tuple[np.ndarray, np.ndarray, float]:
    norb = len(t_vec)
    if target_n <= 1e-12:
        return np.zeros(norb), np.zeros(norb), 1.0
    if target_n >= norb - 1e-12:
        return np.ones(norb), np.zeros(norb), 1.0
    lo = float(np.min(t_vec)) - 50.0
    hi = float(np.max(t_vec)) + 50.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        n = 1.0 / (1.0 + np.exp(np.clip(t_vec - mid, -50.0, 50.0)))
        if float(np.sum(n)) < target_n:
            lo = mid
        else:
            hi = mid
    mu = 0.5 * (lo + hi)
    n = 1.0 / (1.0 + np.exp(np.clip(t_vec - mu, -50.0, 50.0)))
    wt = n * (1.0 - n)
    return n, wt, float(np.sum(wt)) + 1e-30


def _skew_expm_eig(K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w, V = np.linalg.eigh(1j * K)
    ew = np.exp(-1j * w)
    U = np.real((V * ew[None, :]) @ V.conj().T)
    return U, w, V


def _skew_frechet_triu(w: np.ndarray, V: np.ndarray, M: np.ndarray, idx_triu: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    dw = w[:, None] - w[None, :]
    ew = np.exp(1j * w)
    mask = np.abs(dw) > 1e-7
    D = np.where(
        mask,
        (ew[:, None] - ew[None, :]) / np.where(mask, dw, 1.0),
        1j * ew[:, None] * (1.0 - 0.5j * dw),
    )
    VMV = V.conj().T @ M @ V
    L = np.real(V @ ((-1j * D) * VMV) @ V.conj().T)
    return (L - L.T)[idx_triu]


def _thc_exchange_matrix(X_mo: np.ndarray, M_thc: np.ndarray, f_vec: np.ndarray) -> np.ndarray:
    """Strictly O(N^3) generalized exchange matrix K_{pq}[f] = sum_{KL} X_{pK} (M_{KL} D_{KL}[f]) X_{qL}."""
    act = np.where(np.abs(f_vec) > 1e-14)[0]
    if len(act) == 0:
        norb = X_mo.shape[0]
        return np.zeros((norb, norb), dtype=float)
    X_a = X_mo[act, :]
    f_a = f_vec[act]
    D_KL = (X_a.T * f_a[None, :]) @ X_a
    W_KL = M_thc * D_KL
    return (X_mo @ W_KL) @ X_mo.T


def _deterministic_localized_seed_rotation(
    X0: np.ndarray,
    h0: np.ndarray,
    M_thc: np.ndarray,
    n_alpha: int,
    n_beta: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a deterministic O(N^3) localized frontier rotation + spin-separated occupation guess.
    """
    norb = X0.shape[0]
    is_singlet = (n_alpha == n_beta)
    k_w = min(3, n_beta, norb - n_alpha)
    k_lo = (n_beta - k_w) if is_singlet else max(0, n_beta - 3)
    k_hi = (n_alpha + k_w) if is_singlet else min(norb, n_alpha + 3)
    sub = slice(k_lo, k_hi)
    m_sub = k_hi - k_lo

    U_seed = np.eye(norb, dtype=float)
    if m_sub > 1:
        X_sub = X0[sub, :]
        _, _, piv = qr(X_sub, pivoting=True, mode="economic")
        P_pts = X_sub[:, piv[:m_sub]]
        sv_u, _, sv_vt = np.linalg.svd(P_pts, full_matrices=False)
        U_loc = sv_u @ sv_vt
        h_loc_sub = np.diag(U_loc.T @ h0[sub, sub] @ U_loc)
        loc_ord = np.argsort(h_loc_sub, kind="mergesort")
        U_loc = U_loc[:, loc_ord].copy()
        if np.linalg.det(U_loc) < 0.0:
            U_loc[:, -1] *= -1.0
        U_seed[sub, sub] = U_loc

    _ = M_thc
    h_loc = np.diag(U_seed.T @ h0 @ U_seed)

    if is_singlet:
        t0 = np.ones(norb) * 5.0
        t0[:n_alpha] = -5.0
        if m_sub == 1:
            t0[k_lo] = 0.0
        elif m_sub > 1:
            h_seg = h_loc[k_lo:k_hi]
            h_span = max(float(np.max(h_seg) - np.min(h_seg)), 0.20)
            t0[k_lo:k_hi] = 0.16 * (h_seg - float(np.mean(h_seg))) / h_span
        return U_seed, t0, t0
    else:
        ta0 = np.ones(norb) * 5.0
        tb0 = np.ones(norb) * 5.0
        ta0[:k_lo] = -5.0
        tb0[:k_lo] = -5.0
        H_sub = (U_seed.T @ h0 @ U_seed)[sub, sub].copy()
        np.fill_diagonal(H_sub, 0.0)
        hop_norm = np.sqrt(np.sum(H_sub * H_sub, axis=1))
        n_unpaired = n_alpha - n_beta
        rad_score = hop_norm + 0.1 * (h_loc[sub] - np.min(h_loc[sub]))
        rad_order = np.argsort(rad_score)
        rad_idx = k_lo + rad_order[:n_unpaired]
        rem_idx = [k_lo + i for i in range(m_sub) if (k_lo + i) not in set(rad_idx)]

        ta0[rad_idx] = -5.0
        tb0[rad_idx] = +5.0
        n_pairs_rem = n_beta - k_lo
        rem_sorted = sorted(rem_idx, key=lambda idx: h_loc[idx])
        for rank, idx in enumerate(rem_sorted):
            if rank < n_pairs_rem - 1:
                ta0[idx] = -3.0
                tb0[idx] = -3.0
            elif rank <= n_pairs_rem:
                ta0[idx] = -0.15 + 0.30 * (rank - (n_pairs_rem - 1))
                tb0[idx] = -0.15 + 0.30 * (rank - (n_pairs_rem - 1))
            else:
                ta0[idx] = +4.0
                tb0[idx] = +4.0
        return U_seed, ta0, tb0


def solve_static_scf(
    mol: gto.Mole,
    thc: THCData,
    mf_init: Optional[scf.hf.SCF] = None,
    n_quad_phi2: int = 28,
    amp_mode: str = "ph_pauli_s1",
) -> StaticSCFResult:
    """
    Variationally optimize natural orbitals C and spin-resolved occupations (n_alpha, n_beta)
    under E_{0+stat} in strictly O(N^3) FLOPs, followed by degenerate-subspace semi-canonicalization.
    """
    if mf_init is None:
        mf_init = scf.RHF(mol) if mol.spin == 0 else scf.ROHF(mol)
        mf_init.conv_tol = 1e-10
        mf_init.max_cycle = 150
        mf_init.kernel()

    C0 = mf_init.mo_coeff
    if isinstance(C0, tuple) or C0.ndim == 3:
        C0 = C0[0]
    C0 = np.asarray(C0, dtype=float).copy()

    h1_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    enuc = float(mol.energy_nuc())
    norb = C0.shape[1]
    n_alpha, n_beta = mol.nelec
    is_singlet = (mol.spin == 0)
    use_ph_pauli = (amp_mode == "ph_pauli_s1")

    h0 = C0.T @ h1_ao @ C0
    X0 = C0.T @ thc.X_ao
    M_thc = thc.M_thc

    lam_q, alpha_q = build_phi2_laplace_quadrature(n_quad=n_quad_phi2)
    idx_triu = np.triu_indices(norb, 1)
    n_rot = len(idx_triu[0])

    def make_val_and_grad(C_base: np.ndarray):
        h_base = C_base.T @ h1_ao @ C_base
        X_base = C_base.T @ thc.X_ao

        def val_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            Krot = np.zeros((norb, norb), dtype=float)
            Krot[idx_triu] = x[:n_rot]
            Krot = Krot - Krot.T
            U, w_eig, V_eig = _skew_expm_eig(Krot)
            h_mo = U.T @ h_base @ U
            X_mo = U.T @ X_base
            Q_mo = X_mo * X_mo

            if is_singlet:
                na, wa_t, wsa = _map_fermi_occupations(x[n_rot:], float(n_alpha))
                nb = na
                if use_ph_pauli:
                    wa, bp_a = ph_pauli_amplitude_and_jac(na, Q_mo)
                    wb = wa
                else:
                    wa, dwa = pauli_amplitude_and_deriv(na)
                    wb, dwb = wa, dwa
            else:
                na, wa_t, wsa = _map_fermi_occupations(x[n_rot:n_rot + norb], float(n_alpha))
                nb, wb_t, wsb = _map_fermi_occupations(x[n_rot + norb:], float(n_beta))
                if use_ph_pauli:
                    wa, bp_a = ph_pauli_amplitude_and_jac(na, Q_mo)
                    wb, bp_b = ph_pauli_amplitude_and_jac(nb, Q_mo)
                else:
                    wa, dwa = pauli_amplitude_and_deriv(na)
                    wb, dwb = pauli_amplitude_and_deriv(nb)

            ntot = na + nb
            d_K = Q_mo.T @ ntot
            v_K = M_thc @ d_K
            J_mo = (X_mo * v_K[None, :]) @ X_mo.T

            Ka = _thc_exchange_matrix(X_mo, M_thc, na)
            Kb = Ka if is_singlet else _thc_exchange_matrix(X_mo, M_thc, nb)

            e_cum = 0.0
            dE_dwa = np.zeros(norb, dtype=float)
            dE_dwb = np.zeros(norb, dtype=float)
            dE_dna_cum = np.zeros(norb, dtype=float)
            dE_dnb_cum = np.zeros(norb, dtype=float)
            F_cum = np.zeros((norb, norb), dtype=float)

            has_wa = bool(np.any(wa > 1e-13))
            has_wb = bool(np.any(wb > 1e-13))
            if has_wa and has_wb:
                act_a = np.where(wa > 1e-13)[0]
                Xa = X_mo[act_a, :]
                for lam_t, a_t in zip(lam_q, alpha_q):
                    ea = np.exp(-np.clip(lam_t * (wa[act_a] ** 3), 0.0, 100.0))
                    ua = (wa[act_a] * wa[act_a]) * ea
                    dua = (2.0 * wa[act_a] - 3.0 * lam_t * (wa[act_a] ** 4)) * ea
                    if is_singlet:
                        D_KL = (Xa.T * ua[None, :]) @ Xa
                        Kub_act = (X_mo @ (M_thc * D_KL)) @ Xa.T
                        Kd_act = np.diag(Kub_act[act_a, :])
                        e_cum -= a_t * float(np.dot(ua, Kd_act))
                        dE_dwa[act_a] -= a_t * dua * Kd_act
                        F_cum[:, act_a] -= (2.0 * a_t) * (Kub_act * ua[None, :])
                    else:
                        act_b = np.where(wb > 1e-13)[0]
                        Xb = X_mo[act_b, :]
                        eb = np.exp(-np.clip(lam_t * (wb[act_b] ** 3), 0.0, 100.0))
                        ub = (wb[act_b] * wb[act_b]) * eb
                        dub = (2.0 * wb[act_b] - 3.0 * lam_t * (wb[act_b] ** 4)) * eb
                        D_b = (Xb.T * ub[None, :]) @ Xb
                        D_a = (Xa.T * ua[None, :]) @ Xa
                        Kub_act = (X_mo @ (M_thc * D_b)) @ Xa.T
                        Kua_act = (X_mo @ (M_thc * D_a)) @ Xb.T
                        Kd_a = np.diag(Kub_act[act_a, :])
                        Kd_b = np.diag(Kua_act[act_b, :])
                        e_cum -= a_t * float(np.dot(ua, Kd_a))
                        dE_dwa[act_a] -= a_t * dua * Kd_a
                        dE_dwb[act_b] -= a_t * dub * Kd_b
                        F_cum[:, act_a] -= a_t * (Kub_act * ua[None, :])
                        F_cum[:, act_b] -= a_t * (Kua_act * ub[None, :])

                if use_ph_pauli:
                    if is_singlet:
                        dE_dna_cum, dE_dQ = bp_a(2.0 * dE_dwa)
                        F_cum = F_cum + X_mo @ ((X_mo * dE_dQ).T)
                    else:
                        dE_dna_cum, dE_dQa = bp_a(dE_dwa)
                        dE_dnb_cum, dE_dQb = bp_b(dE_dwb)
                        F_cum = F_cum + X_mo @ ((X_mo * (dE_dQa + dE_dQb)).T)
                else:
                    if is_singlet:
                        dE_dna_cum = 2.0 * dwa * dE_dwa
                    else:
                        dE_dna_cum = dwa * dE_dwa
                        dE_dnb_cum = dwb * dE_dwb

            e1 = float(np.dot(ntot, np.diag(h_mo)))
            eJ = 0.5 * float(np.dot(ntot, np.diag(J_mo)))
            eK_mf = -0.5 * (float(np.dot(na, np.diag(Ka))) + float(np.dot(nb, np.diag(Kb))))
            E_tot = enuc + e1 + eJ + eK_mf + e_cum

            A = (
                (h_mo + J_mo) * ntot[None, :]
                - Ka * na[None, :]
                - Kb * nb[None, :]
                + F_cum
            )
            grad_k = _skew_frechet_triu(w_eig, V_eig, 2.0 * (U @ A), idx_triu)

            if is_singlet:
                dE_dn = 2.0 * (np.diag(h_mo) + np.diag(J_mo) - np.diag(Ka)) + dE_dna_cum
                grad_t = -wa_t * (dE_dn - float(np.dot(wa_t, dE_dn)) / wsa)
                return E_tot, np.concatenate([grad_k, grad_t])
            else:
                dE_dna = np.diag(h_mo) + np.diag(J_mo) - np.diag(Ka) + dE_dna_cum
                dE_dnb = np.diag(h_mo) + np.diag(J_mo) - np.diag(Kb) + dE_dnb_cum
                ga = -wa_t * (dE_dna - float(np.dot(wa_t, dE_dna)) / wsa) if n_alpha > 0 else np.zeros(norb)
                gb = -wb_t * (dE_dnb - float(np.dot(wb_t, dE_dnb)) / wsb) if n_beta > 0 else np.zeros(norb)
                return E_tot, np.concatenate([grad_k, ga, gb])

        return val_and_grad

    dim_t = norb if is_singlet else 2 * norb
    seeds = []

    x_s0 = np.zeros(n_rot + dim_t, dtype=float)
    if is_singlet:
        t0 = np.ones(norb) * 38.0
        t0[:n_alpha] = -38.0
        x_s0[n_rot:] = t0
    else:
        ta0 = np.ones(norb) * 38.0
        ta0[:n_alpha] = -38.0
        tb0 = np.ones(norb) * 38.0
        tb0[:n_beta] = -38.0
        x_s0[n_rot:n_rot + norb] = ta0
        x_s0[n_rot + norb:] = tb0
    seeds.append((C0, x_s0, 120))

    x_s1 = np.zeros(n_rot + dim_t, dtype=float)
    if n_rot > 0:
        K_init = np.zeros((norb, norb), dtype=float)
        for p in range(norb):
            for q in range(p + 1, norb):
                if abs(p - n_alpha) <= 3 or abs(q - n_alpha) <= 3:
                    K_init[p, q] = 0.14 * np.sin(1.7 * (p + 1) + 2.3 * (q + 1))
        x_s1[:n_rot] = K_init[idx_triu]
    if is_singlet:
        k_op = min(3, n_alpha, norb - n_alpha)
        t1 = np.ones(norb) * 4.5
        t1[:n_alpha - k_op] = -4.5
        if k_op > 0:
            t1[n_alpha - k_op:n_alpha + k_op] = 0.08 * np.linspace(-1.0, 1.0, 2 * k_op)
        x_s1[n_rot:] = t1
    else:
        ehomo_a = mf_init.mo_energy[max(0, n_alpha - 1)]
        ta1 = (mf_init.mo_energy - ehomo_a) / 0.14
        tb1 = np.ones(norb) * 4.5
        if n_beta > 0:
            ehomo_b = mf_init.mo_energy[max(0, n_beta - 1)]
            tb1 = (mf_init.mo_energy - ehomo_b) / 0.14
        x_s1[n_rot:n_rot + norb] = ta1
        x_s1[n_rot + norb:] = tb1
    seeds.append((C0, x_s1, 280))

    if n_beta > 0 and norb > 1:
        U_loc, ta2, tb2 = _deterministic_localized_seed_rotation(X0, h0, M_thc, n_alpha, n_beta)
        x_s2 = np.zeros(n_rot + dim_t, dtype=float)
        if is_singlet:
            x_s2[n_rot:] = ta2
        else:
            x_s2[n_rot:n_rot + norb] = ta2
            x_s2[n_rot + norb:] = tb2
        seeds.append((C0 @ U_loc, x_s2, 280))

    if is_singlet and n_alpha >= 2 and (norb - n_alpha) >= 2:
        _, U_hc = np.linalg.eigh(0.5 * (h0 + h0.T))
        if np.linalg.det(U_hc) < 0.0:
            U_hc[:, -1] *= -1.0
        x_s3 = np.zeros(n_rot + dim_t, dtype=float)
        k_op3 = min(2, n_alpha, norb - n_alpha)
        t3 = np.ones(norb) * 5.0
        t3[:n_alpha - k_op3] = -5.0
        t3[n_alpha - k_op3:n_alpha + k_op3] = 0.05 * np.linspace(-1.0, 1.0, 2 * k_op3)
        x_s3[n_rot:] = t3
        seeds.append((C0 @ U_hc, x_s3, 280))

    best_E = 1e18
    best_C = C0
    best_na = np.zeros(norb)
    best_nb = np.zeros(norb)

    for C_base, x_init, max_it in seeds:
        vg = make_val_and_grad(C_base)
        res = minimize(
            vg,
            x_init,
            jac=True,
            method="L-BFGS-B",
            options={"gtol": 1e-7, "ftol": 1e-13, "maxiter": max_it},
        )
        if res.fun < best_E:
            best_E = float(res.fun)
            Krot = np.zeros((norb, norb), dtype=float)
            Krot[idx_triu] = res.x[:n_rot]
            Krot = Krot - Krot.T
            U_opt, _, _ = _skew_expm_eig(Krot)
            best_C = C_base @ U_opt
            if is_singlet:
                na_opt, _, _ = _map_fermi_occupations(res.x[n_rot:], float(n_alpha))
                best_na = na_opt
                best_nb = na_opt.copy()
            else:
                na_opt, _, _ = _map_fermi_occupations(res.x[n_rot:n_rot + norb], float(n_alpha))
                nb_opt, _, _ = _map_fermi_occupations(res.x[n_rot + norb:], float(n_beta))
                best_na = na_opt
                best_nb = nb_opt

    ntot = best_na + best_nb
    order = np.argsort(-(ntot + 0.01 * (best_na - best_nb)), kind="mergesort")
    C_opt = best_C[:, order].copy()
    na = best_na[order].copy()
    nb = best_nb[order].copy()
    ntot = na + nb

    X_mo = C_opt.T @ thc.X_ao
    h_mo = C_opt.T @ h1_ao @ C_opt
    d_K = (X_mo * X_mo).T @ ntot
    v_K = M_thc @ d_K
    J_mo = (X_mo * v_K[None, :]) @ X_mo.T
    Ka = _thc_exchange_matrix(X_mo, M_thc, na)
    Kb = Ka if is_singlet else _thc_exchange_matrix(X_mo, M_thc, nb)
    F_semi = h_mo + J_mo - 0.5 * (Ka + Kb)

    p_start = 0
    while p_start < norb:
        p_end = p_start + 1
        while (
            p_end < norb
            and abs(float(na[p_end] - na[p_start])) < 1e-4
            and abs(float(nb[p_end] - nb[p_start])) < 1e-4
        ):
            p_end += 1
        is_idemp = (4.0 * float(na[p_start]) * (1.0 - float(na[p_start])) < 1e-6) and (
            4.0 * float(nb[p_start]) * (1.0 - float(nb[p_start])) < 1e-6
        )
        if (p_end - p_start > 1) and is_idemp:
            sub = slice(p_start, p_end)
            _, V_sub = np.linalg.eigh(0.5 * (F_semi[sub, sub] + F_semi[sub, sub].T))
            C_opt[:, sub] = C_opt[:, sub] @ V_sub
        p_start = p_end

    return StaticSCFResult(
        e_stat=best_E,
        C_no=C_opt,
        n_alpha=na,
        n_beta=nb,
        e_hf_init=float(mf_init.e_tot),
    )
