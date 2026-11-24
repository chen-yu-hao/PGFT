#pragma once

#include "pgft/linalg.hpp"
#include <vector>

namespace pgft {

struct THCResult {
    RowMatrixXd X_ao;         // (nao, nk) AO values at selected collocation pivots
    RowMatrixXd Z;            // (naux, nk) fitted auxiliary-to-THC projection matrix
    RowMatrixXd M_thc;        // (nk, nk) symmetric effective Coulomb kernel Z^T Z
    RowMatrixXd pts;          // (nk, 3) selected 3D collocation coordinates (Bohr)
    std::vector<int> pivots;  // (nk,) selected indices into filtered candidate points
};

/**
 * Deterministic 170-point O_h-symmetric unit-sphere angular shell (170, 3).
 */
RowMatrixXd generate_oh_unit_sphere();

/**
 * SO(3)-covariant, size-consistent local orthogonal frames Q_A in SO(3) for each atom.
 */
std::vector<Eigen::Matrix3d> compute_local_so3_frames(
    const Eigen::Ref<const RowMatrixXd>& coords_bohr,
    const Eigen::Ref<const Eigen::VectorXd>& z_eff
);

/**
 * Deterministic, SO(3)-covariant candidate collocation points around all atomic centers (natm * n_rad * 170, 3).
 */
RowMatrixXd generate_molecular_collocation_points(
    const Eigen::Ref<const RowMatrixXd>& coords_bohr,
    const Eigen::Ref<const Eigen::VectorXd>& z_eff,
    int n_rad = 28,
    double r_min = 3e-4,
    double r_max = 9.0
);

/**
 * Fast O(N^3) Lazy Pivoted Cholesky on S^{(2)}_{gg'} = (Phi^T Phi)_{gg'}^2
 * with column-major cache locality and dead-column compaction.
 *
 * @param Phi_unit   (nao, ng) column-unit-normalized AO values on candidate points
 * @param col_norm   (ng,) original Unsöld column norms (for SO(3)-invariant 1e-15 tie-breaker at k=0)
 * @param max_rank   maximum number of pivots
 * @param tol        Cholesky diagonal stopping threshold (default 1e-11)
 * @return selected pivot indices into [0, ng)
 */
std::vector<int> pivoted_cholesky_isdf(
    const Eigen::Ref<const RowMatrixXd>& Phi_unit,
    const Eigen::Ref<const Eigen::VectorXd>& col_norm,
    int max_rank,
    double tol = 1e-11
);

/**
 * Fit auxiliary-to-THC matrix Z (naux, nk) and symmetric kernel M_thc = Z^T Z (nk, nk)
 * from X_ao (nao, nk) and packed lower-triangular 3-center RI integrals B_tril (naux, nao*(nao+1)/2)
 * in strictly O(N^3) FLOPs using dsyrk, tiled Schwarz-screened dgemm, and Cholesky solve.
 */
std::pair<RowMatrixXd, RowMatrixXd> fit_thc_z(
    const Eigen::Ref<const RowMatrixXd>& X_ao,
    const Eigen::Ref<const RowMatrixXd>& B_tril,
    double schwarz_tol = 1e-18,
    double reg = 1e-13
);

/**
 * End-to-end O(N^3) ISDF-THC builder from PySCF-evaluated AO collocation matrix ao_vals (ng_raw, nao),
 * candidate coordinates coords_raw (ng_raw, 3), and packed 3-center DF integrals B_tril (naux, nao*(nao+1)/2).
 * Note: ao_vals is accepted in PySCF's native (ng_raw, nao) C-contiguous layout to avoid transposing!
 */
THCResult build_isdf_thc_cpp(
    const Eigen::Ref<const RowMatrixXd>& ao_vals_ng_nao,
    const Eigen::Ref<const RowMatrixXd>& coords_raw,
    const Eigen::Ref<const RowMatrixXd>& B_tril,
    double c_thc = 18.0,
    double tol = 1e-11
);

} // namespace pgft
