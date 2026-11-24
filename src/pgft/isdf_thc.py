"""
Deterministic, SO(3)-Covariant Interpolative Separable Density Fitting / THC
============================================================================

Strictly O(N^3), xc-grid-free factorization of 3-center RI/DF Coulomb integrals:
    B^P_{uv} ~= sum_{K=1}^{N_K} X_{uK} X_{vK} Z_{PK},    M^{THC} = Z^T Z
with ZERO imports of `pyscf.dft`, `libxc`, or `xcfun`, and ZERO 4-index ERIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from pyscf import df, gto


@dataclass
class THCData:
    """Container for strictly O(N^3) ISDF-THC factorization tensors."""
    X_ao: np.ndarray      # (nao, nk) AO values at selected collocation pivots
    Z: np.ndarray         # (naux, nk) fitted auxiliary-to-THC projection matrix
    M_thc: np.ndarray     # (nk, nk) symmetric effective Coulomb kernel Z^T Z
    pts: np.ndarray       # (nk, 3) selected 3D collocation coordinates (Bohr)
    auxbasis: str | dict  # Auxiliary RI basis specification


def generate_oh_unit_sphere() -> np.ndarray:
    """
    Construct a deterministic 170-point O_h-symmetric (48-fold signed-permutation
    invariant) unit-sphere angular shell in pure NumPy.

    Combines:
      - 6 axial points:       (+-1, 0, 0) permutations
      - 12 edge midpoints:    (+-1, +-1, 0) / sqrt(2)
      - 8 octant corners:     (+-1, +-1, +-1) / sqrt(3)
      - 4 x 24-point orbits and 1 x 48-point orbit evenly tiling the unit octant.

    Any axis permutation or sign flip in the full octahedral group O_h maps this
    point set strictly onto itself.
    """
    # Octant generators via spherical equal-angle simplex (m = 6) + diagonal arc midpoint
    generators = []
    m = 6
    for i in range(m, -1, -1):
        for j in range(m - i, -1, -1):
            k = m - i - j
            if i >= j >= k >= 0:
                v = np.sin(0.5 * np.pi * np.array([i, j, k], dtype=float) / float(m))
                v /= np.linalg.norm(v)
                generators.append(v)

    # Midpoint on the diagonal symmetry arc between (1,1,0)/sqrt(2) and (1,1,1)/sqrt(3)
    v_arc = np.array([5.0, 5.0, 2.0], dtype=float)
    v_arc /= np.linalg.norm(v_arc)
    generators.append(v_arc)

    perms = [
        (0, 1, 2), (0, 2, 1), (1, 0, 2),
        (1, 2, 0), (2, 0, 1), (2, 1, 0),
    ]
    signs = np.array([
        [sx, sy, sz]
        for sx in (1.0, -1.0)
        for sy in (1.0, -1.0)
        for sz in (1.0, -1.0)
    ], dtype=float)

    all_pts = []
    for g in generators:
        for p in perms:
            gp = g[list(p)]
            for s in signs:
                all_pts.append(gp * s)

    pts_arr = np.array(all_pts, dtype=float)
    # Deterministic deduplication at 1e-12 precision
    rounded = np.round(pts_arr, decimals=11)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    sphere = pts_arr[np.sort(unique_idx)]
    sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
    return sphere  # shape (170, 3)


def compute_local_so3_frames(mol: gto.Mole) -> np.ndarray:
    """
    Compute SO(3)-covariant, size-consistent local orthogonal frames Q_A in SO(3)
    for each atom A from smooth short-range neighbor geometry:
        I_A = sum_{B != A} w_{AB} (R_B - R_A)(R_B - R_A)^T,
        w_{AB} = Z_B * exp(-(R_{AB} / 4.0)^2)   (R_{AB} in Bohr).

    Properties:
      1. Rigid 3D rotation R_rot in SO(3) transforms I_A -> R_rot I_A R_rot^T,
         so Q_A transforms covariantly as R_rot Q_A (up to O_h symmetries under
         which the 170-point O_h angular shell is 100% invariant).
      2. Distant fragments at R > 18 A (> 34 Bohr) have w_{AB} = 0 to machine
         precision, preserving exact isolated monomer point sets in supersystems.
    """
    natm = mol.natm
    coords = mol.atom_coords(unit="Bohr")  # (natm, 3)
    z_eff = np.array([
        float(mol.atom_charge(ia)) if mol.atom_charge(ia) > 0
        else float(gto.charge(mol.atom_pure_symbol(ia)))
        for ia in range(natm)
    ], dtype=float)

    frames = np.zeros((natm, 3, 3), dtype=float)
    for ia in range(natm):
        d = coords - coords[ia][None, :]           # (natm, 3)
        r2 = np.sum(d * d, axis=1)                 # (natm,)
        w = z_eff * np.exp(-r2 / 16.0)
        w[ia] = 0.0
        w = np.where(w < 1e-30, 0.0, w)

        I_loc = (d.T * w[None, :]) @ d             # (3, 3)
        trace_I = float(np.trace(I_loc))
        if trace_I < 1e-24:
            frames[ia] = np.eye(3)
            continue

        evals, Q = np.linalg.eigh(I_loc)

        # SO(3)-covariant vector for deterministic eigenvector sign canonicalization
        idx_asym = 1.0 + 0.0137 * (np.arange(natm, dtype=float) + 1.0)
        u_vec = np.sum(d * (w * idx_asym)[:, None], axis=0)
        projs = Q.T @ u_vec
        for m in range(3):
            if abs(projs[m]) > 1e-10 and projs[m] < 0.0:
                Q[:, m] *= -1.0

        # Enforce right-handed proper rotation det(Q) = +1 on least-projected axis
        if np.linalg.det(Q) < 0.0:
            m_flip = int(np.argmin(np.abs(Q.T @ u_vec)))
            Q[:, m_flip] *= -1.0

        frames[ia] = Q
    return frames


def generate_molecular_collocation_points(
    mol: gto.Mole,
    n_rad: int = 28,
    r_min: float = 3e-4,
    r_max: float = 9.0,
) -> np.ndarray:
    """
    Generate deterministic, SO(3)-covariant, quadrature-weight-free candidate
    collocation points around all atomic centers using Chebyshev radial shells
    on [r_min, r_max] Bohr and the 170-point O_h unit sphere rotated by Q_A.
    """
    sphere = generate_oh_unit_sphere()  # (170, 3)
    frames = compute_local_so3_frames(mol)
    coords = mol.atom_coords(unit="Bohr")

    k = np.arange(1, n_rad + 1, dtype=float)
    t_cheb = 0.5 * (1.0 - np.cos((2.0 * k - 1.0) * np.pi / (2.0 * n_rad)))
    radii = r_min * ((r_max / r_min) ** t_cheb)

    pts = []
    for ia in range(mol.natm):
        rot_sphere = sphere @ frames[ia].T  # (170, 3)
        # Vectorized outer sum over radial shells: (n_rad, 170, 3) -> (n_rad * 170, 3)
        atom_pts = coords[ia][None, None, :] + radii[:, None, None] * rot_sphere[None, :, :]
        pts.append(atom_pts.reshape(-1, 3))
    return np.vstack(pts)


def build_isdf_thc(
    mol: gto.Mole,
    auxbasis: Optional[str | dict] = None,
    c_thc: float = 18.0,
    tol: float = 1e-11,
    n_rad: int = 28,
) -> THCData:
    r"""
    Strictly O(N^3) Interpolative Separable Density Fitting (ISDF-THC) decomposition
    of the 3-center RI/DF Coulomb tensor:
        B^P_{uv} ~= sum_{K=1}^{N_K} X_{uK} X_{vK} Z_{PK},   M^{THC}_{KL} = (Z^T Z)_{KL}.

    Complexity Audit (all operations <= 3 O(N) indices):
      1. AO evaluation on N_g = O(N) candidate points:            O(N_AO * N_g) = O(N^2)
      2. Lazy Pivoted Cholesky on S^{(2)}_{gg'} = (Phi^T Phi)^2:  O(N_g * N_K^2) = O(N^3)
      3. Exact all-pair Gram matrix (X^T X)^{\odot 2}:            O(N_AO * N_K^2) = O(N^3)
      4. Schwarz-screened RHS A_sig^T B_sig^T (N_sig = O(N)):     O(N_sig * N_K * N_aux) = O(N^3)
      5. Linear solve & M_thc = Z^T Z:                            O(N_K^3 + N_aux * N_K^2) = O(N^3)
    """
    nao = mol.nao_nr()
    resolved_aux = auxbasis if auxbasis is not None else df.make_auxbasis(mol)
    with_df = df.DF(mol, auxbasis=resolved_aux)
    with_df.build()

    # 1. Candidate points & unit-norm preconditioned AO evaluation: O(N^2)
    coords = generate_molecular_collocation_points(mol, n_rad=n_rad)
    Phi_raw = mol.eval_gto("GTOval_sph", coords).T  # (nao, ng)
    col_norm = np.sqrt(np.sum(Phi_raw * Phi_raw, axis=0))
    keep = col_norm > 1e-10
    Phi_raw = np.ascontiguousarray(Phi_raw[:, keep])
    coords = coords[keep]
    col_norm = col_norm[keep]
    Phi = np.ascontiguousarray(Phi_raw / col_norm[None, :])
    ng = Phi.shape[1]

    naux = with_df.get_naoaux()
    max_rank = min(ng, max(int(round(c_thc * nao)), 3 * naux))

    # 2. Exact Lazy Pivoted Cholesky on S^{(2)}_{g,g'} = (Phi^T Phi)_{g,g'}^2: O(N^3)
    # Add SO(3)-invariant 1e-15 tie-breaker at k=0 from Unsöld-invariant col_norm
    # and compact dead columns (diag < tol) periodically with zero change in pivots.
    tb = 1e-15 * (col_norm / (np.max(col_norm) + 1e-30))
    diag = np.ones(ng, dtype=float)
    active_idx = np.arange(ng, dtype=int)
    Phi_act = Phi
    tb_act = tb
    blk_rows = min(max_rank, 128)
    L_act = np.zeros((blk_rows, ng), dtype=float)
    pivots = []

    for k in range(max_rank):
        piv_local = int(np.argmax(diag + tb_act)) if k == 0 else int(np.argmax(diag))
        d_val = diag[piv_local]
        if d_val < tol:
            break
        if k >= L_act.shape[0]:
            grow = min(max_rank - L_act.shape[0], max(128, L_act.shape[0]))
            L_act = np.vstack([L_act, np.zeros((grow, L_act.shape[1]), dtype=float)])
        piv_global = int(active_idx[piv_local])
        pivots.append(piv_global)

        phi_p = Phi[:, piv_global]
        ovlp_col = Phi_act.T @ phi_p
        col = ovlp_col * ovlp_col
        if k > 0:
            lp = L_act[:k, piv_local].copy()
            col = col - lp @ L_act[:k, :]

        l_kk = np.sqrt(max(d_val, 1e-30))
        row_k = col / l_kk
        L_act[k, :] = row_k
        diag = np.maximum(0.0, diag - row_k * row_k)
        diag[piv_local] = 0.0

        # Exact pruning of dead columns (diag < tol => can never be future pivots)
        if (k + 1) % 48 == 0 and len(diag) > 1024:
            alive = diag >= tol
            if not np.any(alive):
                break
            if int(np.sum(alive)) < int(0.8 * len(diag)):
                diag = diag[alive]
                active_idx = active_idx[alive]
                Phi_act = np.ascontiguousarray(Phi_act[:, alive])
                tb_act = tb_act[alive]
                L_act = np.ascontiguousarray(L_act[:k + 1, alive])

    pivots_arr = np.array(pivots, dtype=int)
    X_ao = np.ascontiguousarray(Phi_raw[:, pivots_arr])  # (nao, nk)
    nk = X_ao.shape[1]

    # 3. Exact all-pair Khatri-Rao Gram matrix (X^T X)^{\odot 2} in O(nao * nk^2) = O(N^3)
    XtX = X_ao.T @ X_ao
    AtA = XtX * XtX

    # 4. Schwarz-significant AO-pair contraction with 3-center RI tensor B^P_{uv} in O(N^3)
    # Stream over PySCF DF chunks in packed lower-triangular form (zero unpacked (naux, nao, nao) needed!)
    iu, iv = np.tril_indices(nao)
    mult_tril = np.where(iu == iv, 1.0, 2.0)

    # First pass over chunks (in-memory in PySCF): accumulate pair norms sum_P (B^P_{uv})^2
    chunks = [blk for blk in with_df.loop()]
    B_tril = np.vstack(chunks)  # (naux, nao*(nao+1)//2), O(N_aux * N_AO^2) = O(N^3)
    pair_norms = np.sum(B_tril * B_tril, axis=0)
    sig = pair_norms > 1e-18
    iu_s = iu[sig]
    iv_s = iv[sig]
    mult_s = mult_tril[sig]

    A_sig = X_ao[iu_s, :] * X_ao[iv_s, :]          # (N_sig, nk)
    B_sig = B_tril[:, sig] * mult_s[None, :]       # (naux, N_sig)
    AtB = A_sig.T @ B_sig.T                        # (nk, naux): O(N_sig * nk * naux) = O(N^3)

    # 5. Preconditioned symmetric solve AtA @ Z^T = AtB: O(nk^3 + naux * nk^2) = O(N^3)
    d_scale = np.sqrt(np.maximum(np.diag(AtA), 1e-30))
    AtA_norm = AtA / np.outer(d_scale, d_scale) + 1e-13 * np.eye(nk)
    Z_norm = np.linalg.solve(AtA_norm, AtB / d_scale[:, None])
    Z = np.ascontiguousarray((Z_norm / d_scale[:, None]).T)  # (naux, nk)

    M_thc = Z.T @ Z  # (nk, nk)
    M_thc = 0.5 * (M_thc + M_thc.T)

    return THCData(
        X_ao=X_ao,
        Z=Z,
        M_thc=M_thc,
        pts=coords[pivots_arr],
        auxbasis=resolved_aux,
    )
