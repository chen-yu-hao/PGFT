#pragma once

#include "pgft/linalg.hpp"
#include "pgft/static_scf.hpp"
#include <utility>

namespace pgft {

enum class QuasiparticleMode : int {
    MarkovLocalMu = 0,     // PGFT v1: shifted eps_s and full pairing gap delta_s
    GlobalGapMu = 1,       // Global HOMO-LUMO midpoint chemical potential
    ScaledPairingGap = 2,  // Markov local mu with configurable pairing_gap_scale
    ROHFHalfGap = 3        // PGFT v2 Default: unshifted bar_eps_s and ROHF-consistent (1 - 0.5*w)*delta_s
};

enum class SurvivalKernelMode : int {
    LinearSelfScreeningFree = 0,    // PGFT v1: g_r = max(0, 2 * f_r - 1)
    SmoothRational = 1,             // g_r = f_r^2 / (1 + f_r - f_r^2)
    PowerSelfScreeningFree = 2,     // g_r = max(0, 2 * f_r - 1)^p
    HyperbolicSelfScreeningFree = 3 // PGFT v2 Default: 0.5 * ((2f-1) + sqrt((2f-1)^2 + (alpha*f^2)^2))
};

struct DynamicACFDConfig {
    int n_omega = 28;
    int n_tau = 38;
    AmplitudeParams amp_params{};
    QuasiparticleMode qp_mode = QuasiparticleMode::ROHFHalfGap;
    SurvivalKernelMode survival_mode = SurvivalKernelMode::HyperbolicSelfScreeningFree;
    double pairing_gap_scale = 1.0;
    double survival_exponent = 1.0;
    double hyperbolic_alpha = 0.25;
    bool dual_range_dispersion = true;
    bool use_sub_projector = true;
    double coulomb_reg = 1e-2;
    bool relax_active_subspace = true;
    int singlet_pair_order = 2;
};

struct PostSCFResult {
    double e_stat = 0.0;       // Base static + singlet pair restoration energy
    double e_stat_base = 0.0;  // Base E_{0+stat} energy on relaxed occupations
    double e_singlet_pair = 0.0;
    double e_dyn = 0.0;
    double e_tot = 0.0;
    Eigen::VectorXd n_alpha;
    Eigen::VectorXd n_beta;
};

/**
 * Deterministic O(1) exponential quadrature of the Cauchy-Lorentz kernel:
 *   E / (omega_m^2 + E^2) ~= sum_{k=1}^{n_tau} C_{m,k} exp(-E * tau_k)
 * fitted on a 300-point Chebyshev logarithmic energy grid in [emin, emax].
 */
std::pair<Eigen::VectorXd, RowMatrixXd> build_laplace_cos_weights(
    const Eigen::Ref<const Eigen::VectorXd>& omegas,
    double emin,
    double emax,
    int n_tau = 38
);

/**
 * Evaluate strictly O(N^3) derivative-free Pauli-Entanglement Dielectric ACFD energy
 * (supports both PGFT v1 and PGFT v2 Dual-Range Casimir-Polder + ROHF-HalfGap modes).
 */
double compute_dynamic_acfd_cpp(
    const Eigen::Ref<const RowMatrixXd>& C_no,
    const Eigen::Ref<const Eigen::VectorXd>& n_alpha_vec,
    const Eigen::Ref<const Eigen::VectorXd>& n_beta_vec,
    const Eigen::Ref<const RowMatrixXd>& h1_ao,
    const Eigen::Ref<const RowMatrixXd>& X_ao,
    const Eigen::Ref<const RowMatrixXd>& Z,
    const Eigen::Ref<const RowMatrixXd>& M_thc,
    int n_alpha_elec,
    int n_beta_elec,
    const DynamicACFDConfig& cfg = {}
);

/**
 * Unified PGFT v2 O(N^3) Post-Static Active-Subspace Variational Relaxation +
 * Dual-Range Casimir-Polder ACFD + Monogamy-Gated O(2) Singlet Pair Restoration.
 */
PostSCFResult evaluate_post_scf_v2_cpp(
    const Eigen::Ref<const RowMatrixXd>& C_no,
    const Eigen::Ref<const Eigen::VectorXd>& n_alpha_init,
    const Eigen::Ref<const Eigen::VectorXd>& n_beta_init,
    const Eigen::Ref<const RowMatrixXd>& h1_ao,
    const Eigen::Ref<const RowMatrixXd>& X_ao,
    const Eigen::Ref<const RowMatrixXd>& Z,
    const Eigen::Ref<const RowMatrixXd>& M_thc,
    double enuc,
    int n_alpha_elec,
    int n_beta_elec,
    const DynamicACFDConfig& cfg = {}
);

} // namespace pgft
