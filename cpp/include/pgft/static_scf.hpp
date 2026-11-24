#pragma once

#include "pgft/linalg.hpp"
#include <tuple>
#include <utility>

namespace pgft {

/**
 * Parameter-free Pauli-bounded entanglement amplitude generator w(n) in [0, 1]
 * where s = 4 n (1 - n) in [0, 1]:
 *   - Pade13:       w(s) = s / (2 - s^3)                          [Exact PGFT v1]
 *   - LinearS:      w(s) = s                                      [Simple Cumulant limit]
 *   - RationalPade: w(s) = (a * s) / (1 + (a - 1) * s^p)
 *   - PowerPade:    w(s) = s^p / (2 - s^(3*p))
 *   - PHPauliS1:    w_p(n, Q_mo) = s_p - (1 - g^{PH}_p) Delta^+_p [Exact PGFT v2 Default]
 */
enum class AmplitudeMode : int {
    Pade13 = 0,
    LinearS = 1,
    RationalPade = 2,
    PowerPade = 3,
    PHPauliS1 = 4
};

struct AmplitudeParams {
    AmplitudeMode mode = AmplitudeMode::PHPauliS1;
    double alpha = 0.5;
    double power = 3.0;
    double ph_eta = 1e-3;
};

struct StaticSCFConfig {
    int n_quad_phi2 = 28;
    double L_exp_phi2 = 13.5;
    int lbfgs_m = 12;
    double gtol = 2e-8;
    double ftol = 1e-13;
    bool use_preconditioner = true;
    AmplitudeParams amp_params{};
};

struct StaticSCFOutput {
    double e_stat = 0.0;
    RowMatrixXd C_no;          // (nao, norb) semi-canonical natural orbitals
    Eigen::VectorXd n_alpha;   // (norb,) alpha natural occupations in [0, 1]
    Eigen::VectorXd n_beta;    // (norb,) beta natural occupations in [0, 1]
    double e_hf_init = 0.0;
    int total_iters = 0;
    int total_fevals = 0;
};

/**
 * Intermediate forward-pass state for the O(N^2) spatial particle-hole overlap-gated
 * Pauli-pigeonhole entanglement amplitude w_p(n, Q_mo) = s_p - (1 - g^{PH}_p) Delta^+_p.
 */
struct PHPauliCache {
    Eigen::VectorXd w;
    Eigen::VectorXd dB;
    Eigen::VectorXd gate;
    Eigen::VectorXd exc_plus;
    Eigen::VectorXd dexc_plus;
    Eigen::VectorXd exc_minus;
    Eigen::VectorXd dexc_minus;
    Eigen::VectorXd dgate_dOp;
    Eigen::VectorXd dgate_dOm;
    Eigen::VectorXd rho_p;
    Eigen::VectorXd rho_m;
};

/**
 * Gauss-Legendre log-Laplace quadrature for m=2 generalized harmonic mean:
 *   Phi_2(a, b) = 0.5 * a^2 b^2 / (a^3 + b^3) ~= sum_{t=1}^{N_s} alpha_t u_t(a) u_t(b)
 */
std::pair<Eigen::VectorXd, Eigen::VectorXd> build_phi2_laplace_quadrature(
    int n_quad = 28,
    double L_exp = 13.5
);

/**
 * Evaluate univariate Pauli entanglement amplitude w(n) and derivative dw/dn.
 */
std::pair<Eigen::VectorXd, Eigen::VectorXd> pauli_amplitude_and_deriv(
    const Eigen::Ref<const Eigen::VectorXd>& n_vec,
    const AmplitudeParams& params = {}
);

/**
 * Forward pass of the O(N^2) spatial particle-hole Pauli-pigeonhole entanglement amplitude.
 */
PHPauliCache ph_pauli_amplitude_forward(
    const Eigen::Ref<const Eigen::VectorXd>& n_vec,
    const Eigen::Ref<const RowMatrixXd>& Q_mo,
    double eta = 1e-3
);

/**
 * Reverse-mode AD backpropagation through ph_pauli_amplitude_forward:
 *   - Writes occupation derivative dE/dn into dE_dn_out (norb)
 *   - Accumulates X_mo(p, K) * (dE / dQ_mo(p, K)) into G_cum_T_accum (norb, nk)
 *     so F_cum = X_mo * G_cum_T^T includes the exact spatial-gate orbital gradient in zero extra GEMMs!
 */
void ph_pauli_amplitude_backprop(
    const PHPauliCache& cache,
    const Eigen::Ref<const Eigen::VectorXd>& dE_dw,
    const Eigen::Ref<const RowMatrixXd>& X_mo,
    const Eigen::Ref<const RowMatrixXd>& Q_mo,
    Eigen::VectorXd& dE_dn_out,
    RowMatrixXd& G_cum_T_accum
);

/**
 * Compute Fermi-Dirac occupations n_p = 1 / (1 + exp(t_p - mu)) with exact bisection for sum_p n_p = target_n.
 * Returns (n, wt = n*(1-n), sum(wt) + 1e-30).
 */
std::tuple<Eigen::VectorXd, Eigen::VectorXd, double> map_fermi_occupations(
    const Eigen::Ref<const Eigen::VectorXd>& t_vec,
    double target_n
);

/**
 * Strictly O(N^3) generalized THC exchange matrix K_{pq}[f] = sum_{KL} X_{pK} (M_{KL} D_{KL}[f]) X_{qL}.
 */
RowMatrixXd thc_exchange_matrix(
    const Eigen::Ref<const RowMatrixXd>& X_mo,
    const Eigen::Ref<const RowMatrixXd>& M_thc,
    const Eigen::Ref<const Eigen::VectorXd>& f_vec
);

/**
 * Exact orthogonal exponential U = exp(K) of skew-symmetric K = -K^T and upper-triangular
 * skew-Hermitian Fréchet derivative d/dK_{pq} Tr(M^T exp(K)) for p < q.
 */
std::pair<RowMatrixXd, Eigen::VectorXd> skew_expm_and_frechet(
    const Eigen::Ref<const RowMatrixXd>& K,
    const Eigen::Ref<const RowMatrixXd>& M
);

/**
 * Evaluate parameter-free, strictly O(N^3), O(2)- & SO(3)-invariant, zero-monopole,
 * monogamy-gated Singlet Pair-Exchange Restoration E_{S-T}[C_no, n_alpha].
 */
double compute_singlet_pair_restoration_cpp(
    const Eigen::Ref<const RowMatrixXd>& C_no,
    const Eigen::Ref<const Eigen::VectorXd>& n_alpha_vec,
    const Eigen::Ref<const RowMatrixXd>& X_ao,
    const Eigen::Ref<const RowMatrixXd>& M_thc,
    bool is_singlet,
    int harmonic_order = 2,
    const AmplitudeParams& amp_params = {}
);

/**
 * High-performance O(N^3) C++20 + OpenMP + Fock-Preconditioned L-BFGS Static Cumulant SCF solver.
 */
StaticSCFOutput solve_static_scf_cpp(
    const Eigen::Ref<const RowMatrixXd>& C0,
    const Eigen::Ref<const RowMatrixXd>& h1_ao,
    const Eigen::Ref<const RowMatrixXd>& X_ao,
    const Eigen::Ref<const RowMatrixXd>& M_thc,
    double enuc,
    int n_alpha,
    int n_beta,
    double e_hf_init,
    const Eigen::Ref<const Eigen::VectorXd>& mo_energy = Eigen::VectorXd(),
    const StaticSCFConfig& cfg = {}
);

} // namespace pgft
