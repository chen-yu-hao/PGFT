#include "pgft/dynamic_acfd.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace pgft {

std::pair<Eigen::VectorXd, RowMatrixXd> build_laplace_cos_weights(
    const Eigen::Ref<const Eigen::VectorXd>& omegas,
    double emin,
    double emax,
    int n_tau
) {
    emin = std::max(emin, 1e-3);
    emax = std::max(emax, emin * 10.0);
    const int n_omega = static_cast<int>(omegas.size());

    const double tau_min = 0.05 / emax;
    const double tau_max = 18.0 / emin;
    Eigen::VectorXd taus(n_tau);
    if (n_tau == 1) {
        taus[0] = tau_min;
    } else {
        const double l10_min = std::log10(tau_min);
        const double l10_max = std::log10(tau_max);
        for (int i = 0; i < n_tau; ++i) {
            const double exp10 = l10_min + (l10_max - l10_min) * static_cast<double>(i) / static_cast<double>(n_tau - 1);
            taus[i] = std::pow(10.0, exp10);
        }
    }

    constexpr int n_samp = 300;
    Eigen::VectorXd E_grid(n_samp);
    const double e_ratio = emax / emin;
    for (int k = 1; k <= n_samp; ++k) {
        const double t_cheb = 0.5 * (1.0 - std::cos((2.0 * k - 1.0) * std::numbers::pi / (2.0 * n_samp)));
        E_grid[k - 1] = emin * std::pow(e_ratio, t_cheb);
    }

    ColMatrixXd A_col(n_samp, n_tau);
    for (int j = 0; j < n_tau; ++j) {
        for (int i = 0; i < n_samp; ++i) {
            const double prod = std::clamp(E_grid[i] * taus[j], 0.0, 200.0);
            A_col(i, j) = std::exp(-prod);
        }
    }

    const int ldb = std::max(n_samp, n_tau);
    ColMatrixXd B_col = ColMatrixXd::Zero(ldb, n_omega);
    for (int m = 0; m < n_omega; ++m) {
        for (int i = 0; i < n_samp; ++i) {
            const double E = E_grid[i];
            B_col(i, m) = E / (omegas[m] * omegas[m] + E * E);
        }
    }

    Eigen::VectorXd s_vals(std::min(n_samp, n_tau));
    int rank = 0;
    oblas_dgelsd(n_samp, n_tau, n_omega, A_col.data(), n_samp, B_col.data(), ldb, s_vals.data(), 1e-12, &rank);

    RowMatrixXd cos_mat(n_omega, n_tau);
    for (int m = 0; m < n_omega; ++m) {
        for (int j = 0; j < n_tau; ++j) {
            cos_mat(m, j) = B_col(j, m);
        }
    }
    return {taus, cos_mat};
}

namespace {

Eigen::VectorXd quasiparticle_energies_for_spin(
    const Eigen::VectorXd& eps_s,
    const Eigen::VectorXd& delta_s,
    const Eigen::VectorXd& n_s,
    const RowMatrixXd& O_ovlp,
    const DynamicACFDConfig& cfg
) {
    const int norb = static_cast<int>(eps_s.size());
    const double gap_scale = (cfg.qp_mode == QuasiparticleMode::ScaledPairingGap) ? cfg.pairing_gap_scale : 1.0;

    auto median_of = [](Eigen::VectorXd v) -> double {
        const int n = static_cast<int>(v.size());
        if (n == 0) return 0.0;
        std::sort(v.data(), v.data() + n);
        if (n % 2 == 1) return v[n / 2];
        return 0.5 * (v[n / 2 - 1] + v[n / 2]);
    };

    bool any_occ = false, any_vir = false;
    double max_occ_glob = -std::numeric_limits<double>::infinity();
    double min_vir_glob =  std::numeric_limits<double>::infinity();
    for (int p = 0; p < norb; ++p) {
        if (n_s[p] > 0.5) {
            any_occ = true;
            max_occ_glob = std::max(max_occ_glob, eps_s[p]);
        } else {
            any_vir = true;
            min_vir_glob = std::min(min_vir_glob, eps_s[p]);
        }
    }
    const double global_mu_gap = (any_occ && any_vir) ? (0.5 * (max_occ_glob + min_vir_glob)) : median_of(eps_s);

    if (cfg.qp_mode == QuasiparticleMode::GlobalGapMu) {
        Eigen::VectorXd eqp(norb);
        for (int p = 0; p < norb; ++p) {
            const double de = eps_s[p] - global_mu_gap;
            const double dl = gap_scale * delta_s[p];
            eqp[p] = std::sqrt(de * de + dl * dl);
        }
        return eqp;
    }

    Eigen::VectorXd s_ent(norb);
    Eigen::VectorXd inv_sqrt_diag(norb);
    for (int p = 0; p < norb; ++p) {
        s_ent[p] = std::clamp(4.0 * n_s[p] * (1.0 - n_s[p]), 0.0, 1.0);
        inv_sqrt_diag[p] = 1.0 / std::sqrt(std::max(O_ovlp(p, p), 1e-30));
    }

    RowMatrixXd P_markov(norb, norb);
    for (int p = 0; p < norb; ++p) {
        double row_sum = 0.0;
        for (int q = 0; q < norb; ++q) {
            const double c_pq = std::clamp(O_ovlp(p, q) * inv_sqrt_diag[p] * inv_sqrt_diag[q], 0.0, 1.0);
            const double val = ((c_pq > 1e-4) ? (c_pq * c_pq) : 0.0) + ((p == q) ? 1e-24 : 0.0);
            P_markov(p, q) = val;
            row_sum += val;
        }
        P_markov.row(p) /= row_sum;
    }

    RowMatrixXd W_sub = P_markov;
    RowMatrixXd W_next(norb, norb);
    for (int step = 0; step < 4; ++step) {
        blas_dgemm_rm(false, false, norb, norb, norb, 1.0, W_sub.data(), norb, W_sub.data(), norb, 0.0, W_next.data(), norb);
        W_sub.swap(W_next);
    }

    Eigen::VectorXd Eqp(norb);
    for (int p = 0; p < norb; ++p) {
        const auto wp = W_sub.row(p);
        const double max_wp = wp.maxCoeff();
        const double thresh = 1e-8 * max_wp;

        bool has_occ_sub = false, has_vir_sub = false;
        double max_occ_sub = -std::numeric_limits<double>::infinity();
        double min_vir_sub =  std::numeric_limits<double>::infinity();
        int count_in_sub = 0;
        double sum_wp = 0.0;
        double w_ent_sum = 0.0;
        double w_ent_eps = 0.0;

        for (int q = 0; q < norb; ++q) {
            const double wq = wp[q];
            sum_wp += wq;
            const double w_se = wq * s_ent[q];
            w_ent_sum += w_se;
            w_ent_eps += w_se * eps_s[q];
            if (wq > thresh) {
                ++count_in_sub;
                if (n_s[q] > 0.5) {
                    has_occ_sub = true;
                    max_occ_sub = std::max(max_occ_sub, eps_s[q]);
                } else {
                    has_vir_sub = true;
                    min_vir_sub = std::min(min_vir_sub, eps_s[q]);
                }
            }
        }

        const double mu_gap = (has_occ_sub && has_vir_sub) ? (0.5 * (max_occ_sub + min_vir_sub)) : global_mu_gap;
        const double s_sub = (w_ent_sum / (sum_wp + 1e-30)) * static_cast<double>(count_in_sub);

        double mu_p = mu_gap;
        if (w_ent_sum > 1e-7) {
            const double mu_ent = w_ent_eps / w_ent_sum;
            const double blend = std::min(1.0, s_sub);
            mu_p = blend * mu_ent + (1.0 - blend) * mu_gap;
        }

        const double de = eps_s[p] - mu_p;
        const double dl = gap_scale * delta_s[p];
        Eqp[p] = std::sqrt(de * de + dl * dl);
    }
    return Eqp;
}

void k_diag_and_kl(
    const RowMatrixXd& X_mo,
    const RowMatrixXd& M_thc,
    const Eigen::VectorXd& f_vec,
    Eigen::VectorXd& K_diag_out,
    RowMatrixXd& D_KL_out,
    bool need_k_diag = true
) {
    const int norb = static_cast<int>(X_mo.rows());
    const int nk = static_cast<int>(X_mo.cols());

    std::vector<int> act;
    act.reserve(norb);
    bool all_nonneg = true;
    for (int p = 0; p < norb; ++p) {
        if (std::abs(f_vec[p]) > 1e-18) {
            act.push_back(p);
            if (f_vec[p] < 0.0) all_nonneg = false;
        }
    }

    D_KL_out.resize(nk, nk);
    if (act.empty()) {
        D_KL_out.setZero();
        if (need_k_diag) K_diag_out = Eigen::VectorXd::Zero(norb);
        return;
    }

    const int n_act = static_cast<int>(act.size());
    if (all_nonneg) {
        RowMatrixXd X_scaled(n_act, nk);
        for (int i = 0; i < n_act; ++i) {
            const int p = act[i];
            X_scaled.row(i) = std::sqrt(f_vec[p]) * X_mo.row(p);
        }
        blas_dsyrk_at_a_rm(nk, n_act, 1.0, X_scaled.data(), nk, 0.0, D_KL_out.data(), nk);
    } else {
        RowMatrixXd X_a(n_act, nk);
        RowMatrixXd X_af(n_act, nk);
        for (int i = 0; i < n_act; ++i) {
            const int p = act[i];
            X_a.row(i) = X_mo.row(p);
            X_af.row(i) = f_vec[p] * X_mo.row(p);
        }
        blas_dgemm_rm(true, false, nk, nk, n_act, 1.0, X_af.data(), nk, X_a.data(), nk, 0.0, D_KL_out.data(), nk);
    }

    if (!need_k_diag) return;

    RowMatrixXd W_KL = M_thc.cwiseProduct(D_KL_out);
    RowMatrixXd Y(norb, nk);
    blas_dgemm_rm(false, false, norb, nk, nk, 1.0, X_mo.data(), nk, W_KL.data(), nk, 0.0, Y.data(), nk);

    K_diag_out.resize(norb);
    for (int p = 0; p < norb; ++p) {
        K_diag_out[p] = Y.row(p).dot(X_mo.row(p));
    }
}

/**
 * Smooth, size-consistent Tikhonov-damped Coulomb weight W_coul = Z_s^T Z_s in O(N^3).
 */
RowMatrixXd build_smooth_coulomb_metric(
    const Eigen::Ref<const RowMatrixXd>& X_ao,
    const Eigen::Ref<const RowMatrixXd>& Z,
    double eps_cond
) {
    const int nao = static_cast<int>(X_ao.rows());
    const int nk = static_cast<int>(X_ao.cols());
    const int naux = static_cast<int>(Z.rows());

    RowMatrixXd XtX_ao(nk, nk);
    blas_dsyrk_at_a_rm(nk, nao, 1.0, X_ao.data(), nk, 0.0, XtX_ao.data(), nk);
    RowMatrixXd AtA_rm = XtX_ao.cwiseProduct(XtX_ao);

    // AtB_rm (nk x naux) = AtA_rm (nk x nk) * Z^T (nk x naux)
    RowMatrixXd AtB_rm(nk, naux);
    blas_dgemm_rm(false, true, nk, naux, nk, 1.0, AtA_rm.data(), nk, Z.data(), nk, 0.0, AtB_rm.data(), naux);

    ColMatrixXd A_col = AtA_rm;
    for (int K = 0; K < nk; ++K) {
        A_col(K, K) += eps_cond * AtA_rm(K, K) + eps_cond * eps_cond;
    }
    ColMatrixXd B_col = AtB_rm; // (nk x naux) ColMajor
    std::vector<int> ipiv(nk, 0);
    int info = oblas_dgesv(nk, naux, A_col.data(), nk, ipiv.data(), B_col.data(), nk);
    if (info != 0) {
        throw std::runtime_error("oblas_dgesv failed in build_smooth_coulomb_metric");
    }

    // Z_s has shape (naux, nk) RowMajor == transpose of B_col (nk, naux)
    RowMatrixXd Z_s(naux, nk);
    for (int P = 0; P < naux; ++P) {
        for (int K = 0; K < nk; ++K) {
            Z_s(P, K) = B_col(K, P);
        }
    }

    RowMatrixXd W_coul(nk, nk);
    blas_dsyrk_at_a_rm(nk, naux, 1.0, Z_s.data(), nk, 0.0, W_coul.data(), nk);
    return W_coul;
}

/**
 * Strictly O(N^3), SO(3)- and size-consistent collocation connectivity kernel S_intra(K, L) in [0, 1]
 * using pivoted-QR spatially localized occupied/entangled MOs.
 */
RowMatrixXd occupied_connected_projector(
    const RowMatrixXd& X_mo,
    const Eigen::VectorXd& ntot
) {
    const int norb = static_cast<int>(X_mo.rows());
    const int nk = static_cast<int>(X_mo.cols());

    std::vector<int> occ_idx;
    occ_idx.reserve(norb);
    for (int p = 0; p < norb; ++p) {
        if (ntot[p] > 1e-4) occ_idx.push_back(p);
    }
    const int n_occ = static_cast<int>(occ_idx.size());
    if (n_occ <= 1 || n_occ > nk) {
        return RowMatrixXd::Ones(nk, nk);
    }

    RowMatrixXd X_occ(n_occ, nk);
    Eigen::VectorXd n_sub(n_occ);
    for (int i = 0; i < n_occ; ++i) {
        X_occ.row(i) = X_mo.row(occ_idx[i]);
        n_sub[i] = ntot[occ_idx[i]];
    }

    ColMatrixXd X_occ_col = X_occ;
    std::vector<int> jpvt(nk, 0);
    Eigen::VectorXd tau_qr(std::min(n_occ, nk));
    oblas_dgeqp3(n_occ, nk, X_occ_col.data(), n_occ, jpvt.data(), tau_qr.data());

    Eigen::MatrixXd P_pts(n_occ, n_occ);
    for (int j = 0; j < n_occ; ++j) {
        const int col_idx = std::clamp(jpvt[j] - 1, 0, nk - 1);
        for (int i = 0; i < n_occ; ++i) {
            P_pts(i, j) = X_occ(i, col_idx);
        }
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(P_pts, Eigen::ComputeFullU | Eigen::ComputeFullV);
    RowMatrixXd U_loc = svd.matrixU() * svd.matrixV().transpose();

    RowMatrixXd X_loc(n_occ, nk);
    blas_dgemm_rm(true, false, n_occ, nk, n_occ, 1.0, U_loc.data(), n_occ, X_occ.data(), nk, 0.0, X_loc.data(), nk);
    Eigen::VectorXd n_loc = (U_loc.cwiseProduct(U_loc)).transpose() * n_sub;

    RowMatrixXd Q_occ(n_occ, nk);
    for (int i = 0; i < n_occ; ++i) {
        const double nl = std::max(n_loc[i], 1e-8);
        Q_occ.row(i) = nl * X_loc.row(i).cwiseProduct(X_loc.row(i));
    }

    RowMatrixXd O_occ(n_occ, n_occ);
    blas_dsyrk_a_at_rm(n_occ, nk, 1.0, Q_occ.data(), nk, 0.0, O_occ.data(), n_occ);

    Eigen::VectorXd inv_rt_d(n_occ);
    for (int i = 0; i < n_occ; ++i) {
        inv_rt_d[i] = 1.0 / std::sqrt(std::max(O_occ(i, i), 1e-30));
    }

    RowMatrixXd W_m(n_occ, n_occ);
    for (int i = 0; i < n_occ; ++i) {
        double rsum = 0.0;
        for (int j = 0; j < n_occ; ++j) {
            const double co = std::clamp(O_occ(i, j) * inv_rt_d[i] * inv_rt_d[j], 0.0, 1.0);
            const double r = co / 0.02;
            const double r2 = r * r;
            const double r4 = r2 * r2;
            const double val = (r4 / (1.0 + r4)) + ((i == j) ? 1e-24 : 0.0);
            W_m(i, j) = val;
            rsum += val;
        }
        W_m.row(i) /= rsum;
    }

    for (int st = 0; st < 4; ++st) {
        RowMatrixXd W_next = W_m * W_m;
        W_m.swap(W_next);
    }

    RowMatrixXd Rn(n_occ, n_occ);
    for (int i = 0; i < n_occ; ++i) {
        const double rmax = std::max(W_m.row(i).maxCoeff(), 1e-30);
        Rn.row(i) = W_m.row(i) / rmax;
    }

    RowMatrixXd W_conn_occ(n_occ, n_occ);
    for (int i = 0; i < n_occ; ++i) {
        for (int j = 0; j < n_occ; ++j) {
            const double a = Rn(i, j) * Rn(i, j);
            const double b = Rn(j, i) * Rn(j, i);
            W_conn_occ(i, j) = 0.5 * (a / (a + 1e-6) + b / (b + 1e-6));
        }
    }

    RowMatrixXd Om(n_occ, nk);
    for (int K = 0; K < nk; ++K) {
        double csum = 0.0;
        for (int i = 0; i < n_occ; ++i) csum += Q_occ(i, K);
        const double inv_csum = 1.0 / (csum + 1e-30);
        for (int i = 0; i < n_occ; ++i) {
            Om(i, K) = Q_occ(i, K) * inv_csum;
        }
    }

    RowMatrixXd W_Om(n_occ, nk);
    blas_dgemm_rm(false, false, n_occ, nk, n_occ, 1.0, W_conn_occ.data(), n_occ, Om.data(), nk, 0.0, W_Om.data(), nk);
    RowMatrixXd S_intra(nk, nk);
    blas_dgemm_rm(true, false, nk, nk, n_occ, 1.0, Om.data(), nk, W_Om.data(), nk, 0.0, S_intra.data(), nk);

    for (int K = 0; K < nk; ++K) {
        for (int L = 0; L < nk; ++L) {
            S_intra(K, L) = std::clamp(S_intra(K, L), 0.0, 1.0);
        }
    }
    return S_intra;
}

struct PostSCFContext {
    int nao = 0;
    int norb = 0;
    int nk = 0;
    int naux = 0;
    int n_alpha_elec = 0;
    int n_beta_elec = 0;
    bool is_singlet = true;
    double enuc = 0.0;
    DynamicACFDConfig cfg;

    RowMatrixXd X_mo;
    RowMatrixXd Q_mo;
    RowMatrixXd O_ovlp;
    RowMatrixXd W_pos;
    RowMatrixXd W_coul;
    Eigen::VectorXd h_diag;
    Eigen::VectorXd lam_q;
    Eigen::VectorXd alpha_q;
    Eigen::VectorXd omegas;
    Eigen::VectorXd w_omega;
    Eigen::VectorXd taus;
    RowMatrixXd cos_mat;

    PostSCFContext(
        const Eigen::Ref<const RowMatrixXd>& C_no,
        const Eigen::Ref<const RowMatrixXd>& h1_ao,
        const Eigen::Ref<const RowMatrixXd>& X_ao,
        const Eigen::Ref<const RowMatrixXd>& Z,
        double enuc_in,
        int na_el,
        int nb_el,
        const DynamicACFDConfig& cfg_in
    )
        : nao(static_cast<int>(C_no.rows())),
          norb(static_cast<int>(C_no.cols())),
          nk(static_cast<int>(X_ao.cols())),
          naux(static_cast<int>(Z.rows())),
          n_alpha_elec(na_el),
          n_beta_elec(nb_el),
          is_singlet(na_el == nb_el),
          enuc(enuc_in),
          cfg(cfg_in) {
        X_mo.resize(norb, nk);
        blas_dgemm_rm(true, false, norb, nk, nao, 1.0, C_no.data(), norb, X_ao.data(), nk, 0.0, X_mo.data(), nk);
        Q_mo = X_mo.cwiseProduct(X_mo);

        RowMatrixXd hC(nao, norb);
        blas_dgemm_rm(false, false, nao, norb, nao, 1.0, h1_ao.data(), nao, C_no.data(), norb, 0.0, hC.data(), norb);
        h_diag.resize(norb);
        for (int p = 0; p < norb; ++p) {
            h_diag[p] = C_no.col(p).dot(hC.col(p));
        }

        O_ovlp.resize(norb, norb);
        blas_dsyrk_a_at_rm(norb, nk, 1.0, Q_mo.data(), nk, 0.0, O_ovlp.data(), norb);

        RowMatrixXd XtX(nk, nk);
        blas_dsyrk_at_a_rm(nk, norb, 1.0, X_mo.data(), nk, 0.0, XtX.data(), nk);
        W_pos = XtX.cwiseProduct(XtX);

        if (cfg.dual_range_dispersion && ((0 < n_alpha_elec && n_alpha_elec < norb) || (0 < n_beta_elec && n_beta_elec < norb))) {
            W_coul = build_smooth_coulomb_metric(X_ao, Z, cfg.coulomb_reg);
        }

        std::tie(lam_q, alpha_q) = build_phi2_laplace_quadrature(28, 13.5);

        const int n_omega = cfg.n_omega;
        const int n_tau_eff = std::max(cfg.n_tau, 38);
        auto [t_gl, wt_gl] = leggauss(n_omega);
        constexpr double w0 = 0.5;
        omegas.resize(n_omega);
        w_omega.resize(n_omega);
        for (int i = 0; i < n_omega; ++i) {
            const double denom = 1.0 - t_gl[i];
            omegas[i] = w0 * (1.0 + t_gl[i]) / denom;
            w_omega[i] = wt_gl[i] * w0 * 2.0 / (denom * denom);
        }
        std::tie(taus, cos_mat) = build_laplace_cos_weights(omegas, 2e-3, 300.0, n_tau_eff);
    }

    std::pair<double, double> eval_stat_and_dyn(
        const Eigen::Ref<const Eigen::VectorXd>& na_v,
        const Eigen::Ref<const Eigen::VectorXd>& nb_v,
        const Eigen::Ref<const RowMatrixXd>& Z,
        const Eigen::Ref<const RowMatrixXd>& M_thc,
        bool need_stat_base
    ) const {
        const Eigen::VectorXd ntot = na_v + nb_v;
        const Eigen::VectorXd d_K = Q_mo.transpose() * ntot;
        const Eigen::VectorXd v_K = M_thc * d_K;
        const Eigen::VectorXd J_diag = Q_mo * v_K;

        Eigen::VectorXd wa, wb;
        if (cfg.amp_params.mode == AmplitudeMode::PHPauliS1) {
            wa = ph_pauli_amplitude_forward(na_v, Q_mo, cfg.amp_params.ph_eta).w;
            wb = is_singlet ? wa : ph_pauli_amplitude_forward(nb_v, Q_mo, cfg.amp_params.ph_eta).w;
        } else {
            wa = pauli_amplitude_and_deriv(na_v, cfg.amp_params).first;
            wb = is_singlet ? wa : pauli_amplitude_and_deriv(nb_v, cfg.amp_params).first;
        }

        const Eigen::VectorXd kapa_half = 0.5 * wa;
        const Eigen::VectorXd kapb_half = 0.5 * wb;

        Eigen::VectorXd Ka_diag, Kka_diag, Kb_diag, Kkb_diag, dummy_diag;
        RowMatrixXd gamma_a_KL, kappa_a_KL, gamma_b_KL, kappa_b_KL;
        RowMatrixXd kt_a_KL, kt_b_KL;

        k_diag_and_kl(X_mo, M_thc, na_v, Ka_diag, gamma_a_KL, true);
        k_diag_and_kl(X_mo, M_thc, kapa_half, Kka_diag, kappa_a_KL, true);
        if (cfg.dual_range_dispersion) {
            const Eigen::VectorXd nwa = na_v.cwiseProduct(wa);
            k_diag_and_kl(X_mo, M_thc, nwa, dummy_diag, kt_a_KL, false);
        }
        if (is_singlet) {
            Kb_diag = Ka_diag;
            gamma_b_KL = gamma_a_KL;
            Kkb_diag = Kka_diag;
            kappa_b_KL = kappa_a_KL;
            if (cfg.dual_range_dispersion) kt_b_KL = kt_a_KL;
        } else {
            k_diag_and_kl(X_mo, M_thc, nb_v, Kb_diag, gamma_b_KL, true);
            k_diag_and_kl(X_mo, M_thc, kapb_half, Kkb_diag, kappa_b_KL, true);
            if (cfg.dual_range_dispersion) {
                const Eigen::VectorXd nwb = nb_v.cwiseProduct(wb);
                k_diag_and_kl(X_mo, M_thc, nwb, dummy_diag, kt_b_KL, false);
            }
        }

        double e_stat_base = 0.0;
        if (need_stat_base) {
            double e_cum = 0.0;
            if (wa.maxCoeff() > 1e-7 && wb.maxCoeff() > 1e-7) {
                std::vector<int> act_a;
                for (int p = 0; p < norb; ++p) {
                    if (wa[p] > 1e-13) act_a.push_back(p);
                }
                const int ma = static_cast<int>(act_a.size());
                RowMatrixXd Xa(ma, nk);
                for (int i = 0; i < ma; ++i) Xa.row(i) = X_mo.row(act_a[i]);

                std::vector<int> act_b;
                RowMatrixXd Xb;
                int mb = ma;
                if (!is_singlet) {
                    for (int p = 0; p < norb; ++p) {
                        if (wb[p] > 1e-13) act_b.push_back(p);
                    }
                    mb = static_cast<int>(act_b.size());
                    Xb.resize(mb, nk);
                    for (int i = 0; i < mb; ++i) Xb.row(i) = X_mo.row(act_b[i]);
                }

                if (ma > 0 && mb > 0) {
                    RowMatrixXd X_sc(std::max(ma, mb), nk);
                    RowMatrixXd W_KL(nk, nk);
                    RowMatrixXd Y_a(ma, nk);
                    for (int t = 0; t < static_cast<int>(lam_q.size()); ++t) {
                        const double lam_t = lam_q[t];
                        const double a_t = alpha_q[t];
                        Eigen::VectorXd ua(ma);
                        for (int i = 0; i < ma; ++i) {
                            const double w = wa[act_a[i]];
                            ua[i] = (w * w) * std::exp(-std::clamp(lam_t * w * w * w, 0.0, 100.0));
                        }
                        if (is_singlet) {
                            for (int i = 0; i < ma; ++i) X_sc.row(i) = std::sqrt(ua[i]) * Xa.row(i);
                            cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, nk, ma, 1.0, X_sc.data(), nk, 0.0, W_KL.data(), nk);
                        } else {
                            for (int i = 0; i < mb; ++i) {
                                const double w = wb[act_b[i]];
                                const double ub = (w * w) * std::exp(-std::clamp(lam_t * w * w * w, 0.0, 100.0));
                                X_sc.row(i) = std::sqrt(ub) * Xb.row(i);
                            }
                            cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, nk, mb, 1.0, X_sc.data(), nk, 0.0, W_KL.data(), nk);
                        }
                        for (int K = 0; K < nk; ++K) {
                            W_KL(K, K) *= M_thc(K, K);
                            for (int L = K + 1; L < nk; ++L) {
                                const double w_kl = W_KL(K, L) * M_thc(K, L);
                                W_KL(K, L) = w_kl;
                                W_KL(L, K) = w_kl;
                            }
                        }
                        blas_dgemm_rm(false, false, ma, nk, nk, 1.0, Xa.data(), nk, W_KL.data(), nk, 0.0, Y_a.data(), nk);
                        double dot_sum = 0.0;
                        for (int i = 0; i < ma; ++i) {
                            dot_sum += ua[i] * Xa.row(i).dot(Y_a.row(i));
                        }
                        e_cum -= a_t * dot_sum;
                    }
                }
            }
            e_stat_base = enuc + ntot.dot(h_diag) + 0.5 * ntot.dot(J_diag)
                        - 0.5 * (na_v.dot(Ka_diag) + nb_v.dot(Kb_diag)) + e_cum;
        }

        if (!((0 < n_alpha_elec && n_alpha_elec < norb) || (0 < n_beta_elec && n_beta_elec < norb))) {
            return {e_stat_base, 0.0};
        }

        Eigen::VectorXd eps_a, eps_b, delta_a, delta_b;
        if (cfg.qp_mode == QuasiparticleMode::ROHFHalfGap) {
            eps_a = h_diag + J_diag - Ka_diag;
            eps_b = h_diag + J_diag - Kb_diag;
            delta_a = (1.0 - 0.5 * wa.array()).matrix().cwiseProduct(Kkb_diag.cwiseMax(0.0));
            delta_b = (1.0 - 0.5 * wb.array()).matrix().cwiseProduct(Kka_diag.cwiseMax(0.0));
        } else {
            eps_a = h_diag + J_diag - Ka_diag - wa.cwiseProduct(Kkb_diag);
            eps_b = h_diag + J_diag - Kb_diag - wb.cwiseProduct(Kka_diag);
            delta_a = Kkb_diag.cwiseMax(0.0);
            delta_b = Kka_diag.cwiseMax(0.0);
        }

        const Eigen::VectorXd Eqp_a = quasiparticle_energies_for_spin(eps_a, delta_a, na_v, O_ovlp, cfg);
        const Eigen::VectorXd Eqp_b = is_singlet
            ? Eqp_a
            : quasiparticle_energies_for_spin(eps_b, delta_b, nb_v, O_ovlp, cfg);

        Eigen::VectorXd rho_tot(nk);
        for (int K = 0; K < nk; ++K) {
            const double ra = std::max(gamma_a_KL(K, K), 0.0);
            const double rb = is_singlet ? ra : std::max(gamma_b_KL(K, K), 0.0);
            rho_tot[K] = ra + rb;
        }

        RowMatrixXd E_excl = RowMatrixXd::Zero(nk, nk);
        RowMatrixXd Wf_pos(nk, nk);
        const bool use_dr = cfg.dual_range_dispersion;

        #pragma omp parallel for schedule(static) if(nk >= 128)
        for (int K = 0; K < nk; ++K) {
            const double rk = rho_tot[K];
            for (int L = 0; L < nk; ++L) {
                const double den = rk * rho_tot[L];
                double e_kl = 1.0;
                if (den > 1e-280) {
                    const double ga = gamma_a_KL(K, L);
                    const double gb = gamma_b_KL(K, L);
                    const double kap_prod = use_dr
                        ? (kt_a_KL(K, L) * kt_b_KL(K, L))
                        : std::max(0.0, kappa_a_KL(K, L) * kappa_b_KL(K, L));
                    const double excl = ga * ga + gb * gb + 2.0 * kap_prod;
                    e_kl = std::clamp(excl / den, 0.0, 1.0);
                }
                E_excl(K, L) = e_kl;
                Wf_pos(K, L) = W_pos(K, L) * (1.0 - e_kl);
            }
        }

        RowMatrixXd Wc_mod;
        bool has_inter_proj = true;
        if (use_dr) {
            if (cfg.use_sub_projector) {
                RowMatrixXd S_intra = occupied_connected_projector(X_mo, ntot);
                RowMatrixXd S_inter = (1.0 - S_intra.array()).max(0.0).matrix();
                has_inter_proj = (S_inter.maxCoeff() > 1e-15);
                if (has_inter_proj) {
                    Wc_mod = W_coul.cwiseProduct(S_inter);
                }
            } else {
                Wc_mod = W_coul.cwiseProduct(E_excl);
            }
        }

        Eigen::VectorXd kap_bogo_a(norb);
        Eigen::VectorXd kap_bogo_b(norb);
        for (int p = 0; p < norb; ++p) {
            kap_bogo_a[p] = std::sqrt(std::max(0.0, na_v[p] * (1.0 - na_v[p])));
            kap_bogo_b[p] = is_singlet
                ? kap_bogo_a[p]
                : std::sqrt(std::max(0.0, nb_v[p] * (1.0 - nb_v[p])));
        }

        const int n_tau = static_cast<int>(taus.size());
        const int n_omega = static_cast<int>(omegas.size());
        const std::size_t nk2 = static_cast<std::size_t>(nk) * static_cast<std::size_t>(nk);
        RowMatrixXd Pi_tau_flat = RowMatrixXd::Zero(n_tau, static_cast<Eigen::Index>(nk2));

        {
            ScopedSingleThreadBLAS blas_guard;
            #pragma omp parallel if(nk >= 256)
            {
                RowMatrixXd X_w(norb, nk);
                RowMatrixXd Gh(nk, nk);
                RowMatrixXd Gp(nk, nk);
                RowMatrixXd Fa(nk, nk);

                auto accum_spin = [&](double tau,
                                      const Eigen::Ref<const Eigen::VectorXd>& n_s,
                                      const Eigen::VectorXd& kb_s,
                                      const Eigen::VectorXd& Eqp_s,
                                      int nel_s,
                                      double spin_factor,
                                      double* pi_out) {
                    if (!(0 < nel_s && nel_s < norb)) return;

                    int nh = 0;
                    for (int p = 0; p < norb; ++p) {
                        const double ex = std::exp(-std::clamp(Eqp_s[p] * tau, 0.0, 150.0));
                        const double wh = n_s[p] * ex;
                        if (wh > 1e-18) {
                            X_w.row(nh++) = std::sqrt(wh) * X_mo.row(p);
                        }
                    }
                    if (nh == 0) return;
                    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, nk, nh, 1.0, X_w.data(), nk, 0.0, Gh.data(), nk);

                    int np_cnt = 0;
                    for (int p = 0; p < norb; ++p) {
                        const double ex = std::exp(-std::clamp(Eqp_s[p] * tau, 0.0, 150.0));
                        const double wp = (1.0 - n_s[p]) * ex;
                        if (wp > 1e-18) {
                            X_w.row(np_cnt++) = std::sqrt(wp) * X_mo.row(p);
                        }
                    }
                    if (np_cnt == 0) return;
                    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, nk, np_cnt, 1.0, X_w.data(), nk, 0.0, Gp.data(), nk);

                    int nf = 0;
                    for (int p = 0; p < norb; ++p) {
                        if (kb_s[p] <= 1e-18) continue;
                        const double ex = std::exp(-std::clamp(Eqp_s[p] * tau, 0.0, 150.0));
                        const double wf = kb_s[p] * ex;
                        if (wf > 1e-18) {
                            X_w.row(nf++) = std::sqrt(wf) * X_mo.row(p);
                        }
                    }
                    if (nf > 0) {
                        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, nk, nf, 1.0, X_w.data(), nk, 0.0, Fa.data(), nk);
                    }

                    const double pref = 2.0 * spin_factor;
                    for (int K = 0; K < nk; ++K) {
                        const std::size_t row_off = static_cast<std::size_t>(K) * nk;
                        for (int L = K; L < nk; ++L) {
                            const std::size_t idx_kl = row_off + L;
                            const double fa_val = (nf > 0) ? Fa.data()[idx_kl] : 0.0;
                            const double val = pref * (Gh.data()[idx_kl] * Gp.data()[idx_kl] - fa_val * fa_val);
                            pi_out[idx_kl] += val;
                            if (L > K) {
                                pi_out[static_cast<std::size_t>(L) * nk + K] += val;
                            }
                        }
                    }
                };

                #pragma omp for schedule(static)
                for (int it = 0; it < n_tau; ++it) {
                    const double tau = taus[it];
                    double* pi_row = Pi_tau_flat.data() + static_cast<std::size_t>(it) * nk2;
                    if (is_singlet) {
                        accum_spin(tau, na_v, kap_bogo_a, Eqp_a, n_alpha_elec, 2.0, pi_row);
                    } else {
                        accum_spin(tau, na_v, kap_bogo_a, Eqp_a, n_alpha_elec, 1.0, pi_row);
                        accum_spin(tau, nb_v, kap_bogo_b, Eqp_b, n_beta_elec,  1.0, pi_row);
                    }
                }
            }
        }

        RowMatrixXd Pi_iw_flat(n_omega, static_cast<Eigen::Index>(nk2));
        blas_dgemm_rm(
            false, false,
            n_omega, static_cast<int>(nk2), n_tau,
            1.0, cos_mat.data(), n_tau,
            Pi_tau_flat.data(), static_cast<int>(nk2),
            0.0, Pi_iw_flat.data(), static_cast<int>(nk2)
        );

        double ec_dyn = 0.0;
        {
            ScopedSingleThreadBLAS blas_guard;
            #pragma omp parallel reduction(+:ec_dyn) if(nk >= 256)
            {
                RowMatrixXd ZPi(naux, nk);
                RowMatrixXd Pi_pq_rm(naux, naux);
                ColMatrixXd U_eig(naux, naux);
                Eigen::VectorXd eig(naux);
                RowMatrixXd U_act(naux, naux);
                RowMatrixXd V_act(nk, naux);
                RowMatrixXd WV(nk, naux);
                std::vector<int> act_modes;
                act_modes.reserve(naux);

                auto quad_diag = [&](const RowMatrixXd& W_mat, int nr, double eps_add, Eigen::VectorXd& out_v) {
                    blas_dgemm_rm(false, false, nk, nr, nk, 1.0, W_mat.data(), nk, V_act.data(), nr, 0.0, WV.data(), nr);
                    out_v = Eigen::VectorXd::Constant(nr, eps_add);
                    for (int K = 0; K < nk; ++K) {
                        const double* wv_row = WV.data() + static_cast<std::size_t>(K) * nr;
                        const double* v_row = V_act.data() + static_cast<std::size_t>(K) * nr;
                        for (int j = 0; j < nr; ++j) {
                            out_v[j] += wv_row[j] * v_row[j];
                        }
                    }
                };

                #pragma omp for schedule(static)
                for (int iw = 0; iw < n_omega; ++iw) {
                    const double dw = w_omega[iw] / (2.0 * std::numbers::pi);
                    const double* pi_iw = Pi_iw_flat.data() + static_cast<std::size_t>(iw) * nk2;

                    blas_dgemm_rm(false, false, naux, nk, nk, 1.0, Z.data(), nk, pi_iw, nk, 0.0, ZPi.data(), nk);
                    blas_dgemm_rm(false, true, naux, naux, nk, 1.0, ZPi.data(), nk, Z.data(), nk, 0.0, Pi_pq_rm.data(), naux);

                    for (int p = 0; p < naux; ++p) {
                        for (int q = 0; q < naux; ++q) {
                            U_eig(p, q) = 0.5 * (Pi_pq_rm(p, q) + Pi_pq_rm(q, p));
                        }
                    }

                    int info = oblas_dsyevd('V', 'U', naux, U_eig.data(), naux, eig.data());
                    if (info != 0) continue;

                    act_modes.clear();
                    for (int r = 0; r < naux; ++r) {
                        if (eig[r] > 1e-9) act_modes.push_back(r);
                    }
                    const int nr = static_cast<int>(act_modes.size());
                    if (nr == 0) continue;

                    for (int p = 0; p < naux; ++p) {
                        double* u_row = U_act.data() + static_cast<std::size_t>(p) * nr;
                        for (int j = 0; j < nr; ++j) {
                            u_row[j] = U_eig(p, act_modes[j]);
                        }
                    }

                    blas_dgemm_rm(true, false, nk, nr, naux, 1.0, ZPi.data(), nk, U_act.data(), nr, 0.0, V_act.data(), nr);

                    Eigen::VectorXd num_loc, den_loc, mod_num, coul_den;
                    quad_diag(Wf_pos, nr, 0.0, num_loc);
                    quad_diag(W_pos, nr, 1e-30, den_loc);
                    if (use_dr && (!cfg.use_sub_projector || has_inter_proj)) {
                        quad_diag(Wc_mod, nr, 0.0, mod_num);
                        quad_diag(W_coul, nr, 1e-30, coul_den);
                    }

                    double iw_sum = 0.0;
                    for (int j = 0; j < nr; ++j) {
                        const double lam_r = eig[act_modes[j]];
                        const double f_r_loc = std::clamp(num_loc[j] / den_loc[j], 0.0, 1.0);
                        double g_r = 0.0;
                        switch (cfg.survival_mode) {
                            case SurvivalKernelMode::LinearSelfScreeningFree:
                                g_r = std::max(0.0, 2.0 * f_r_loc - 1.0);
                                break;
                            case SurvivalKernelMode::SmoothRational:
                                g_r = (f_r_loc * f_r_loc) / (1.0 + f_r_loc - f_r_loc * f_r_loc);
                                break;
                            case SurvivalKernelMode::PowerSelfScreeningFree: {
                                const double base = std::max(0.0, 2.0 * f_r_loc - 1.0);
                                g_r = (base > 0.0) ? std::pow(base, cfg.survival_exponent) : 0.0;
                                break;
                            }
                            case SurvivalKernelMode::HyperbolicSelfScreeningFree: {
                                const double u_r = 2.0 * f_r_loc - 1.0;
                                const double reg_r = cfg.hyperbolic_alpha * (f_r_loc * f_r_loc);
                                g_r = 0.5 * (u_r + std::sqrt(u_r * u_r + reg_r * reg_r));
                                break;
                            }
                        }

                        const double gx = g_r * lam_r;
                        double w_int = 0.0;
                        if (gx > 1e-4) {
                            w_int = (gx - std::log1p(gx)) / std::max(gx * gx, 1e-30);
                        } else {
                            w_int = 0.5 - gx / 3.0 + (gx * gx) / 4.0 - (gx * gx * gx) / 5.0;
                        }

                        double f_r = f_r_loc;
                        if (use_dr) {
                            if (cfg.use_sub_projector) {
                                if (has_inter_proj) {
                                    const double inter_ratio = mod_num[j] / coul_den[j];
                                    f_r = std::clamp(f_r_loc + (1.0 - f_r_loc) * inter_ratio, 0.0, 1.0);
                                }
                            } else {
                                const double ratio = mod_num[j] / coul_den[j];
                                f_r = std::clamp(1.0 - ratio, 0.0, 1.0);
                            }
                        }
                        iw_sum += f_r * (lam_r * lam_r) * w_int;
                    }
                    ec_dyn -= dw * iw_sum;
                }
            }
        }

        return {e_stat_base, ec_dyn};
    }
};

/**
 * Deterministic derivative-free Brent 1D line minimization + Powell direction-set search
 * for low-dimensional active-occupation relaxation (dim <= 8, maxfev <= 45).
 */
template <typename ObjFn>
Eigen::VectorXd powell_minimize_small(
    ObjFn&& fn,
    const Eigen::VectorXd& x0,
    int maxfev = 40,
    double ftol = 1e-5,
    double xtol = 1e-3
) {
    const int n = static_cast<int>(x0.size());
    int fev = 0;
    auto eval = [&](const Eigen::VectorXd& x) -> double {
        ++fev;
        return fn(x);
    };

    Eigen::VectorXd x = x0;
    double fx = eval(x);
    if (n == 0) return x;

    Eigen::MatrixXd dirs = Eigen::MatrixXd::Identity(n, n);

    auto brent_1d = [&](const Eigen::VectorXd& base, const Eigen::VectorXd& dir, double f_base) -> std::pair<double, double> {
        // Bracket around alpha = 0 using step 0.35
        double ax = 0.0, bx = 0.35;
        double fa = f_base;
        if (fev >= maxfev) return {0.0, fa};
        double fb = eval(base + bx * dir);
        if (fb > fa) {
            std::swap(ax, bx);
            std::swap(fa, fb);
        }
        double cx = bx + 1.618034 * (bx - ax);
        double fc = (fev < maxfev) ? eval(base + cx * dir) : fb;
        while (fb > fc && fev < maxfev && std::abs(cx - ax) < 12.0) {
            const double u = cx + 1.618034 * (cx - bx);
            ax = bx; fa = fb;
            bx = cx; fb = fc;
            cx = u;  fc = eval(base + cx * dir);
        }

        double lo = std::min(ax, cx);
        double hi = std::max(ax, cx);
        double x_b = bx, w_b = bx, v_b = bx;
        double fx_b = fb, fw_b = fb, fv_b = fb;
        if (fa < fx_b) { x_b = ax; fx_b = fa; }
        if (fc < fx_b) { x_b = cx; fx_b = fc; }

        double d = 0.0, e = 0.0;
        constexpr double cg = 0.3819660112501051;
        for (int it = 0; it < 14 && fev < maxfev; ++it) {
            const double xm = 0.5 * (lo + hi);
            const double tol1 = xtol * std::abs(x_b) + 1e-5;
            const double tol2 = 2.0 * tol1;
            if (std::abs(x_b - xm) <= (tol2 - 0.5 * (hi - lo))) break;

            bool para_ok = false;
            double u = 0.0;
            if (std::abs(e) > tol1) {
                const double r = (x_b - w_b) * (fx_b - fv_b);
                double q = (x_b - v_b) * (fx_b - fw_b);
                double p = (x_b - v_b) * q - (x_b - w_b) * r;
                q = 2.0 * (q - r);
                if (q > 0.0) p = -p;
                q = std::abs(q);
                const double etemp = e;
                e = d;
                if (std::abs(p) < std::abs(0.5 * q * etemp) && p > q * (lo - x_b) && p < q * (hi - x_b)) {
                    d = p / q;
                    u = x_b + d;
                    if (u - lo < tol2 || hi - u < tol2) {
                        d = (xm >= x_b) ? tol1 : -tol1;
                    }
                    para_ok = true;
                }
            }
            if (!para_ok) {
                e = (x_b >= xm) ? (lo - x_b) : (hi - x_b);
                d = cg * e;
            }
            u = (std::abs(d) >= tol1) ? (x_b + d) : (x_b + ((d >= 0.0) ? tol1 : -tol1));
            const double fu = eval(base + u * dir);
            if (fu <= fx_b) {
                if (u >= x_b) lo = x_b; else hi = x_b;
                v_b = w_b; fv_b = fw_b;
                w_b = x_b; fw_b = fx_b;
                x_b = u;   fx_b = fu;
            } else {
                if (u < x_b) lo = u; else hi = u;
                if (fu <= fw_b || w_b == x_b) {
                    v_b = w_b; fv_b = fw_b;
                    w_b = u;   fw_b = fu;
                } else if (fu <= fv_b || v_b == x_b || v_b == w_b) {
                    v_b = u;   fv_b = fu;
                }
            }
        }
        return {x_b, fx_b};
    };

    for (int outer = 0; outer < 6 && fev < maxfev; ++outer) {
        const Eigen::VectorXd x_start = x;
        const double f_start = fx;
        int ibig = 0;
        double del_big = 0.0;

        for (int i = 0; i < n && fev < maxfev; ++i) {
            const Eigen::VectorXd dir = dirs.col(i);
            const double f_prev = fx;
            auto [a_opt, f_opt] = brent_1d(x, dir, fx);
            if (f_opt < fx) {
                x += a_opt * dir;
                fx = f_opt;
                if (f_prev - fx > del_big) {
                    del_big = f_prev - fx;
                    ibig = i;
                }
            }
        }

        if (2.0 * std::abs(f_start - fx) <= ftol * (std::abs(f_start) + std::abs(fx) + 1e-20)) {
            break;
        }
        if ((x - x_start).lpNorm<Eigen::Infinity>() <= xtol) {
            break;
        }
        if (fev < maxfev && n > 1) {
            const Eigen::VectorXd x_ext = 2.0 * x - x_start;
            const double f_ext = eval(x_ext);
            if (f_ext < f_start) {
                const double t = 2.0 * (f_start - 2.0 * fx + f_ext) * (f_start - fx - del_big) * (f_start - fx - del_big)
                               - del_big * (f_start - f_ext) * (f_start - f_ext);
                if (t < 0.0) {
                    const Eigen::VectorXd new_dir = (x - x_start).normalized();
                    auto [a_opt, f_opt] = brent_1d(x, new_dir, fx);
                    if (f_opt < fx) {
                        x += a_opt * new_dir;
                        fx = f_opt;
                    }
                    dirs.col(ibig) = dirs.col(n - 1);
                    dirs.col(n - 1) = new_dir;
                }
            }
        }
    }
    return x;
}

} // namespace

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
    const DynamicACFDConfig& cfg
) {
    const int norb = static_cast<int>(C_no.cols());
    if (!((0 < n_alpha_elec && n_alpha_elec < norb) || (0 < n_beta_elec && n_beta_elec < norb))) {
        return 0.0;
    }
    PostSCFContext ctx(C_no, h1_ao, X_ao, Z, 0.0, n_alpha_elec, n_beta_elec, cfg);
    return ctx.eval_stat_and_dyn(n_alpha_vec, n_beta_vec, Z, M_thc, false).second;
}

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
    const DynamicACFDConfig& cfg
) {
    const int norb = static_cast<int>(C_no.cols());
    const bool is_singlet = (n_alpha_elec == n_beta_elec);

    PostSCFContext ctx(C_no, h1_ao, X_ao, Z, enuc, n_alpha_elec, n_beta_elec, cfg);

    std::vector<int> act_idx;
    if (cfg.relax_active_subspace && is_singlet) {
        for (int p = 0; p < norb; ++p) {
            if (n_alpha_init[p] > 1e-3 && n_alpha_init[p] < 1.0 - 1e-3) {
                act_idx.push_back(p);
            }
        }
    }

    Eigen::VectorXd na_opt = n_alpha_init;
    Eigen::VectorXd nb_opt = n_beta_init;
    double es_base = 0.0;
    double ed_opt = 0.0;

    if (act_idx.size() >= 2) {
        // Partition act_idx into spatially connected components via O_ovlp so separated dimers relax independently!
        const int m_tot = static_cast<int>(act_idx.size());
        std::vector<bool> visited(m_tot, false);
        std::vector<std::vector<int>> components;
        for (int s = 0; s < m_tot; ++s) {
            if (visited[s]) continue;
            std::vector<int> comp;
            std::vector<int> stack = {s};
            visited[s] = true;
            while (!stack.empty()) {
                const int u = stack.back();
                stack.pop_back();
                comp.push_back(act_idx[u]);
                const int pu = act_idx[u];
                const double du = std::max(ctx.O_ovlp(pu, pu), 1e-30);
                for (int v = 0; v < m_tot; ++v) {
                    if (!visited[v]) {
                        const int pv = act_idx[v];
                        const double dv = std::max(ctx.O_ovlp(pv, pv), 1e-30);
                        const double c_uv = ctx.O_ovlp(pu, pv) / std::sqrt(du * dv);
                        if (c_uv > 1e-3) {
                            visited[v] = true;
                            stack.push_back(v);
                        }
                    }
                }
            }
            if (comp.size() >= 2) {
                std::sort(comp.begin(), comp.end());
                components.push_back(std::move(comp));
            }
        }

        for (const auto& comp : components) {
            const int n_act = static_cast<int>(comp.size());
            double target_act = 0.0;
            Eigen::VectorXd t_init(n_act);
            for (int i = 0; i < n_act; ++i) {
                const double nv = na_opt[comp[i]];
                target_act += nv;
                const double nc = std::clamp(nv, 1e-6, 1.0 - 1e-6);
                t_init[i] = std::log((1.0 - nc) / nc);
            }

            auto obj = [&](const Eigen::VectorXd& t_sub) -> double {
                auto [n_sub, wt_unused, ws_unused] = map_fermi_occupations(t_sub, target_act);
                (void)wt_unused; (void)ws_unused;
                Eigen::VectorXd na_try = na_opt;
                for (int i = 0; i < n_act; ++i) {
                    na_try[comp[i]] = n_sub[i];
                }
                auto [es, ed] = ctx.eval_stat_and_dyn(na_try, na_try, Z, M_thc, true);
                return es + ed;
            };

            Eigen::VectorXd best_t = t_init;
            double best_val = obj(t_init);
            for (double scale : {1.5, 2.5, 4.0, 8.0, 16.0}) {
                const Eigen::VectorXd t_sc = scale * t_init;
                const double val = obj(t_sc);
                if (val < best_val) {
                    best_val = val;
                    best_t = t_sc;
                }
            }

            const Eigen::VectorXd t_opt = powell_minimize_small(obj, best_t, 40, 1e-5, 1e-3);
            const double val_opt = obj(t_opt);
            const Eigen::VectorXd t_final = (val_opt < best_val) ? t_opt : best_t;
            auto [n_sub_opt, wt_u, ws_u] = map_fermi_occupations(t_final, target_act);
            (void)wt_u; (void)ws_u;
            for (int i = 0; i < n_act; ++i) {
                na_opt[comp[i]] = n_sub_opt[i];
            }
            nb_opt = na_opt;
        }
    }

    std::tie(es_base, ed_opt) = ctx.eval_stat_and_dyn(na_opt, nb_opt, Z, M_thc, true);
    const double e_st = compute_singlet_pair_restoration_cpp(
        C_no, na_opt, X_ao, M_thc, is_singlet, cfg.singlet_pair_order, cfg.amp_params
    );

    PostSCFResult out;
    out.e_stat_base = es_base;
    out.e_singlet_pair = e_st;
    out.e_stat = es_base + e_st;
    out.e_dyn = ed_opt;
    out.e_tot = es_base + e_st + ed_opt;
    out.n_alpha = std::move(na_opt);
    out.n_beta = std::move(nb_opt);
    return out;
}

} // namespace pgft
