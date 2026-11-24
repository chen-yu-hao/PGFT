#include "pgft/static_scf.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <deque>
#include <limits>
#include <numeric>
#include <set>
#include <vector>

namespace pgft {

std::pair<Eigen::VectorXd, Eigen::VectorXd> build_phi2_laplace_quadrature(int n_quad, double L_exp) {
    auto [u, wu] = leggauss(n_quad);
    Eigen::VectorXd lam(n_quad);
    Eigen::VectorXd alpha(n_quad);
    double val_11 = 0.0;
    for (int i = 0; i < n_quad; ++i) {
        lam[i] = std::exp(L_exp * u[i]);
        alpha[i] = 0.5 * L_exp * wu[i] * lam[i];
        val_11 += alpha[i] * std::exp(-2.0 * lam[i]);
    }
    alpha *= (0.25 / val_11);
    return {lam, alpha};
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> pauli_amplitude_and_deriv(
    const Eigen::Ref<const Eigen::VectorXd>& n_vec,
    const AmplitudeParams& params
) {
    const int n = static_cast<int>(n_vec.size());
    Eigen::VectorXd w(n);
    Eigen::VectorXd dw_dn(n);

    for (int i = 0; i < n; ++i) {
        const double nc = std::clamp(n_vec[i], 0.0, 1.0);
        const double s = 4.0 * nc * (1.0 - nc);
        const double ds_dn = 4.0 * (1.0 - 2.0 * nc);

        switch (params.mode) {
            case AmplitudeMode::Pade13: {
                const double s3 = s * s * s;
                const double den = 2.0 - s3;
                w[i] = s / den;
                dw_dn[i] = ds_dn * (2.0 + 2.0 * s3) / (den * den);
                break;
            }
            case AmplitudeMode::LinearS: {
                w[i] = s;
                dw_dn[i] = ds_dn;
                break;
            }
            case AmplitudeMode::RationalPade: {
                const double a = params.alpha;
                const double p = params.power;
                const double sp = (s > 0.0) ? std::pow(s, p) : 0.0;
                const double sp_m1 = (s > 0.0) ? std::pow(s, p - 1.0) : 0.0;
                const double den = 1.0 + (a - 1.0) * sp;
                const double dw_ds = a * (1.0 - (p - 1.0) * (a - 1.0) * sp) / (den * den);
                (void)sp_m1;
                w[i] = (a * s) / den;
                dw_dn[i] = ds_dn * dw_ds;
                break;
            }
            case AmplitudeMode::PowerPade: {
                const double p = params.power;
                if (s <= 1e-30) {
                    w[i] = 0.0;
                    dw_dn[i] = (std::abs(p - 1.0) < 1e-12) ? (0.5 * ds_dn) : 0.0;
                } else {
                    const double sp = std::pow(s, p);
                    const double s3p = sp * sp * sp;
                    const double den = 2.0 - s3p;
                    const double dw_ds = p * std::pow(s, p - 1.0) * (2.0 + 2.0 * s3p) / (den * den);
                    w[i] = sp / den;
                    dw_dn[i] = ds_dn * dw_ds;
                }
                break;
            }
            case AmplitudeMode::PHPauliS1: {
                // Univariate single-orbital Pauli-pigeonhole fallback if Q_mo is not supplied
                const double eta = params.ph_eta;
                const double u = 2.0 * nc - 1.0;
                const double ru = std::sqrt(u * u + eta * eta);
                const double s1c = std::sqrt(1.0 + eta * eta);
                const double m_min = 0.5 * (s1c - ru) / (s1c - eta);
                const double dm_min = -(u / ru) / (s1c - eta);
                w[i] = 4.0 * (1.0 - nc) * m_min;
                dw_dn[i] = -4.0 * m_min + 4.0 * (1.0 - nc) * dm_min;
                break;
            }
        }
    }
    return {w, dw_dn};
}

PHPauliCache ph_pauli_amplitude_forward(
    const Eigen::Ref<const Eigen::VectorXd>& n_vec,
    const Eigen::Ref<const RowMatrixXd>& Q_mo,
    double eta
) {
    const int norb = static_cast<int>(n_vec.size());
    const int nk = static_cast<int>(Q_mo.cols());

    PHPauliCache c;
    c.w.resize(norb);
    c.dB.resize(norb);
    c.gate.resize(norb);
    c.exc_plus.resize(norb);
    c.dexc_plus.resize(norb);
    c.exc_minus.resize(norb);
    c.dexc_minus.resize(norb);
    c.dgate_dOp.resize(norb);
    c.dgate_dOm.resize(norb);
    c.rho_p.resize(nk);
    c.rho_m.resize(nk);

    const double s1c = std::sqrt(1.0 + eta * eta);
    const double den_m = s1c - eta;

    Eigen::VectorXd B_vec(norb);
    for (int p = 0; p < norb; ++p) {
        const double nc = std::clamp(n_vec[p], 0.0, 1.0);
        const double s = 4.0 * nc * (1.0 - nc);
        const double dB = 4.0 * (1.0 - 2.0 * nc);
        B_vec[p] = s;
        c.dB[p] = dB;

        const double u = 2.0 * nc - 1.0;
        const double ru = std::sqrt(u * u + eta * eta);
        const double m_min = 0.5 * (s1c - ru) / den_m;
        const double dm_min = -(u / ru) / den_m;

        const double p_bnd = 4.0 * (1.0 - nc) * m_min;
        const double dp_bnd = -4.0 * m_min + 4.0 * (1.0 - nc) * dm_min;
        const double diff = s - p_bnd;
        const double ddiff = dB - dp_bnd;
        const double r_diff = std::sqrt(diff * diff + 1e-8);
        const double ep = std::max(0.0, 0.5 * (diff + r_diff - 1e-4));
        c.exc_plus[p] = ep;
        c.dexc_plus[p] = (ep > 0.0) ? (0.5 * (1.0 + diff / r_diff) * ddiff) : 0.0;

        const double p_bnd_m = 4.0 * nc * m_min;
        const double dp_bnd_m = 4.0 * m_min + 4.0 * nc * dm_min;
        const double diff_m = s - p_bnd_m;
        const double ddiff_m = dB - dp_bnd_m;
        const double r_diff_m = std::sqrt(diff_m * diff_m + 1e-8);
        const double em = std::max(0.0, 0.5 * (diff_m + r_diff_m - 1e-4));
        c.exc_minus[p] = em;
        c.dexc_minus[p] = (em > 0.0) ? (0.5 * (1.0 + diff_m / r_diff_m) * ddiff_m) : 0.0;
    }

    if (norb > 0 && nk > 0) {
        cblas_dgemv(CblasRowMajor, CblasTrans, norb, nk, 1.0, Q_mo.data(), nk, c.exc_plus.data(), 1, 0.0, c.rho_p.data(), 1);
        cblas_dgemv(CblasRowMajor, CblasTrans, norb, nk, 1.0, Q_mo.data(), nk, c.exc_minus.data(), 1, 0.0, c.rho_m.data(), 1);
    } else {
        c.rho_p.setZero();
        c.rho_m.setZero();
    }

    Eigen::VectorXd Op = Eigen::VectorXd::Zero(norb);
    Eigen::VectorXd Om = Eigen::VectorXd::Zero(norb);
    if (norb > 0 && nk > 0) {
        cblas_dgemv(CblasRowMajor, CblasNoTrans, norb, nk, 1.0, Q_mo.data(), nk, c.rho_p.data(), 1, 0.0, Op.data(), 1);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, norb, nk, 1.0, Q_mo.data(), nk, c.rho_m.data(), 1, 0.0, Om.data(), 1);
    }

    for (int p = 0; p < norb; ++p) {
        const double a_p = Op[p] * Op[p];
        const double b_p = 4.0 * Om[p] * Om[p];
        const double den_g = a_p + b_p + 1e-24;
        const double g_p = b_p / den_g;
        c.gate[p] = g_p;
        c.dgate_dOp[p] = -2.0 * Op[p] * b_p / (den_g * den_g);
        c.dgate_dOm[p] =  8.0 * Om[p] * a_p / (den_g * den_g);
        c.w[p] = B_vec[p] - (1.0 - g_p) * c.exc_plus[p];
    }
    return c;
}

void ph_pauli_amplitude_backprop(
    const PHPauliCache& cache,
    const Eigen::Ref<const Eigen::VectorXd>& dE_dw,
    const Eigen::Ref<const RowMatrixXd>& X_mo,
    const Eigen::Ref<const RowMatrixXd>& Q_mo,
    Eigen::VectorXd& dE_dn_out,
    RowMatrixXd& G_cum_T_accum
) {
    const int norb = static_cast<int>(dE_dw.size());
    const int nk = static_cast<int>(Q_mo.cols());
    dE_dn_out.resize(norb);

    Eigen::VectorXd v_Op(norb);
    Eigen::VectorXd v_Om(norb);
    for (int p = 0; p < norb; ++p) {
        dE_dn_out[p] = dE_dw[p] * (cache.dB[p] - (1.0 - cache.gate[p]) * cache.dexc_plus[p]);
        const double dE_dg = dE_dw[p] * cache.exc_plus[p];
        v_Op[p] = dE_dg * cache.dgate_dOp[p];
        v_Om[p] = dE_dg * cache.dgate_dOm[p];
    }

    if (norb == 0 || nk == 0) return;

    Eigen::VectorXd q_vOp(nk);
    Eigen::VectorXd q_vOm(nk);
    cblas_dgemv(CblasRowMajor, CblasTrans, norb, nk, 1.0, Q_mo.data(), nk, v_Op.data(), 1, 0.0, q_vOp.data(), 1);
    cblas_dgemv(CblasRowMajor, CblasTrans, norb, nk, 1.0, Q_mo.data(), nk, v_Om.data(), 1, 0.0, q_vOm.data(), 1);

    Eigen::VectorXd Qv_p(norb);
    Eigen::VectorXd Qv_m(norb);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, norb, nk, 1.0, Q_mo.data(), nk, q_vOp.data(), 1, 0.0, Qv_p.data(), 1);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, norb, nk, 1.0, Q_mo.data(), nk, q_vOm.data(), 1, 0.0, Qv_m.data(), 1);

    for (int p = 0; p < norb; ++p) {
        dE_dn_out[p] += Qv_p[p] * cache.dexc_plus[p] + Qv_m[p] * cache.dexc_minus[p];
        const double vp = v_Op[p];
        const double ep = cache.exc_plus[p];
        const double vm = v_Om[p];
        const double em = cache.exc_minus[p];
        const double* x_row = X_mo.data() + static_cast<std::size_t>(p) * nk;
        double* g_row = G_cum_T_accum.data() + static_cast<std::size_t>(p) * nk;
        for (int K = 0; K < nk; ++K) {
            const double dE_dQ = vp * cache.rho_p[K] + ep * q_vOp[K] + vm * cache.rho_m[K] + em * q_vOm[K];
            g_row[K] += x_row[K] * dE_dQ;
        }
    }
}

double compute_singlet_pair_restoration_cpp(
    const Eigen::Ref<const RowMatrixXd>& C_no,
    const Eigen::Ref<const Eigen::VectorXd>& n_alpha_vec,
    const Eigen::Ref<const RowMatrixXd>& X_ao,
    const Eigen::Ref<const RowMatrixXd>& M_thc,
    bool is_singlet,
    int harmonic_order,
    const AmplitudeParams& amp_params
) {
    if (!is_singlet || harmonic_order <= 0) {
        return 0.0;
    }
    const int nao = static_cast<int>(C_no.rows());
    const int norb = static_cast<int>(C_no.cols());
    const int nk = static_cast<int>(X_ao.cols());
    if (norb < 2 || nk == 0) return 0.0;

    RowMatrixXd X_mo(norb, nk);
    blas_dgemm_rm(true, false, norb, nk, nao, 1.0, C_no.data(), norb, X_ao.data(), nk, 0.0, X_mo.data(), nk);
    RowMatrixXd Q_mo = X_mo.cwiseProduct(X_mo);

    Eigen::VectorXd wa;
    if (amp_params.mode == AmplitudeMode::PHPauliS1) {
        wa = ph_pauli_amplitude_forward(n_alpha_vec, Q_mo, amp_params.ph_eta).w;
    } else {
        wa = pauli_amplitude_and_deriv(n_alpha_vec, amp_params).first;
    }

    const double eta = amp_params.ph_eta;
    const double s1c = std::sqrt(1.0 + eta * eta);
    const double den_m = s1c - eta;
    Eigen::VectorXd nu_vec(norb);
    for (int p = 0; p < norb; ++p) {
        const double nc = std::clamp(n_alpha_vec[p], 0.0, 1.0);
        const double u = 2.0 * nc - 1.0;
        const double m_min = 0.5 * (s1c - std::sqrt(u * u + eta * eta)) / den_m;
        nu_vec[p] = std::clamp(2.0 * m_min, 0.0, 1.0);
    }

    std::vector<int> act;
    act.reserve(norb);
    for (int p = 0; p < norb; ++p) {
        if (wa[p] > 1e-6) act.push_back(p);
    }
    const int n_act = static_cast<int>(act.size());
    if (n_act < 2) return 0.0;

    std::stable_sort(act.begin(), act.end(), [&](int a, int b) {
        return wa[a] > wa[b];
    });

    RowMatrixXd X_act(n_act, nk);
    Eigen::VectorXd wa_act(n_act);
    Eigen::VectorXd nu_act(n_act);
    for (int i = 0; i < n_act; ++i) {
        X_act.row(i) = X_mo.row(act[i]);
        wa_act[i] = wa[act[i]];
        nu_act[i] = nu_vec[act[i]];
    }

    int i0 = 0;
    while (i0 < n_act) {
        int i1 = i0 + 1;
        while (i1 < n_act && std::abs(wa_act[i1] - wa_act[i0]) < 0.15) {
            ++i1;
        }
        const int m = i1 - i0;
        if (m > 1 && m <= nk) {
            ColMatrixXd X_sub_col = X_act.middleRows(i0, m);
            std::vector<int> jpvt(nk, 0);
            Eigen::VectorXd tau_qr(std::min(m, nk));
            oblas_dgeqp3(m, nk, X_sub_col.data(), m, jpvt.data(), tau_qr.data());

            Eigen::MatrixXd P_pts(m, m);
            for (int j = 0; j < m; ++j) {
                const int col_idx = std::clamp(jpvt[j] - 1, 0, nk - 1);
                for (int r = 0; r < m; ++r) {
                    P_pts(r, j) = X_act(i0 + r, col_idx);
                }
            }
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(P_pts, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::MatrixXd U_loc = svd.matrixU() * svd.matrixV().transpose();
            RowMatrixXd X_rot = U_loc.transpose() * X_act.middleRows(i0, m);
            X_act.middleRows(i0, m) = X_rot;

            Eigen::MatrixXd U2 = U_loc.cwiseProduct(U_loc);
            Eigen::VectorXd wa_rot = U2.transpose() * wa_act.segment(i0, m);
            Eigen::VectorXd nu_rot = U2.transpose() * nu_act.segment(i0, m);
            wa_act.segment(i0, m) = wa_rot;
            nu_act.segment(i0, m) = nu_rot;
        }
        i0 = i1;
    }

    RowMatrixXd Q_act = X_act.cwiseProduct(X_act);
    RowMatrixXd O_act(n_act, n_act);
    blas_dsyrk_a_at_rm(n_act, nk, 1.0, Q_act.data(), nk, 0.0, O_act.data(), n_act);

    Eigen::VectorXd inv_rt_d(n_act);
    for (int i = 0; i < n_act; ++i) {
        inv_rt_d[i] = 1.0 / std::sqrt(std::max(O_act(i, i), 1e-30));
    }
    RowMatrixXd W_s(n_act, n_act);
    for (int i = 0; i < n_act; ++i) {
        double rsum = 0.0;
        for (int j = 0; j < n_act; ++j) {
            const double co = std::clamp(O_act(i, j) * inv_rt_d[i] * inv_rt_d[j], 0.0, 1.0);
            const double op = ((co > 1e-4) ? (co * co) : 0.0) + ((i == j) ? 1e-24 : 0.0);
            W_s(i, j) = op;
            rsum += op;
        }
        W_s.row(i) /= rsum;
    }
    for (int st = 0; st < 4; ++st) {
        RowMatrixXd W_next = W_s * W_s;
        W_s.swap(W_next);
    }
    RowMatrixXd Rn(n_act, n_act);
    for (int i = 0; i < n_act; ++i) {
        const double rmax = std::max(W_s.row(i).maxCoeff(), 1e-30);
        Rn.row(i) = W_s.row(i) / rmax;
    }
    RowMatrixXd W_conn(n_act, n_act);
    for (int i = 0; i < n_act; ++i) {
        for (int j = 0; j < n_act; ++j) {
            const double r1 = Rn(i, j) * Rn(i, j);
            const double r2 = Rn(j, i) * Rn(j, i);
            W_conn(i, j) = 0.5 * (r1 / (r1 + 1e-8) + r2 / (r2 + 1e-8));
        }
    }

    RowMatrixXd MQ_act(n_act, nk);
    blas_dgemm_rm(false, false, n_act, nk, nk, 1.0, Q_act.data(), nk, M_thc.data(), nk, 0.0, MQ_act.data(), nk);

    double e_st = 0.0;
    const double inv_rt2 = 1.0 / std::sqrt(2.0);
    const double rt2 = std::sqrt(2.0);
    Eigen::VectorXd d1(nk), d1_Mq(nk), d2(nk), Md2(nk);

    for (int i = 0; i < n_act; ++i) {
        for (int j = i + 1; j < n_act; ++j) {
            const double w_pq = wa_act[i] * wa_act[j] * W_conn(i, j);
            if (w_pq < 1e-10) continue;
            double mono = 1.0;
            for (int k = 0; k < n_act; ++k) {
                if (k != i && k != j) {
                    mono *= std::max(0.0, 1.0 - 0.5 * (W_conn(i, k) + W_conn(j, k)) * nu_act[k]);
                }
            }
            const double weight = w_pq * mono;
            if (weight < 1e-10) continue;

            d1 = (Q_act.row(i) - Q_act.row(j)).transpose() * inv_rt2;
            d1_Mq = (MQ_act.row(i) - MQ_act.row(j)).transpose() * inv_rt2;
            d2 = (X_act.row(i).cwiseProduct(X_act.row(j))).transpose() * rt2;
            cblas_dgemv(CblasRowMajor, CblasNoTrans, nk, nk, 1.0, M_thc.data(), nk, d2.data(), 1, 0.0, Md2.data(), 1);

            const double G11 = std::max(0.0, d1_Mq.dot(d1));
            const double G12 = d1_Mq.dot(d2);
            const double G22 = std::max(0.0, d2.dot(Md2));
            const double tr = G11 + G22;
            const double det = std::max(0.0, G11 * G22 - G12 * G12);
            if (tr > 1e-18) {
                const double gap = (harmonic_order == 1)
                    ? (det / tr)
                    : (8.0 * (det * det) / (tr * tr * tr));
                e_st += weight * gap;
            }
        }
    }
    return e_st;
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd, double> map_fermi_occupations(
    const Eigen::Ref<const Eigen::VectorXd>& t_vec,
    double target_n
) {
    const int norb = static_cast<int>(t_vec.size());
    if (target_n <= 1e-12) {
        return {Eigen::VectorXd::Zero(norb), Eigen::VectorXd::Zero(norb), 1.0};
    }
    if (target_n >= static_cast<double>(norb) - 1e-12) {
        return {Eigen::VectorXd::Ones(norb), Eigen::VectorXd::Zero(norb), 1.0};
    }

    double lo = t_vec.minCoeff() - 50.0;
    double hi = t_vec.maxCoeff() + 50.0;
    for (int iter = 0; iter < 64; ++iter) {
        const double mid = 0.5 * (lo + hi);
        double sum_n = 0.0;
        for (int p = 0; p < norb; ++p) {
            const double arg = std::clamp(t_vec[p] - mid, -50.0, 50.0);
            sum_n += 1.0 / (1.0 + std::exp(arg));
        }
        if (sum_n < target_n) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    const double mu = 0.5 * (lo + hi);
    Eigen::VectorXd n(norb);
    Eigen::VectorXd wt(norb);
    double wsum = 0.0;
    for (int p = 0; p < norb; ++p) {
        const double arg = std::clamp(t_vec[p] - mu, -50.0, 50.0);
        const double np_val = 1.0 / (1.0 + std::exp(arg));
        const double wtp = np_val * (1.0 - np_val);
        n[p] = np_val;
        wt[p] = wtp;
        wsum += wtp;
    }
    return {n, wt, wsum + 1e-30};
}

RowMatrixXd thc_exchange_matrix(
    const Eigen::Ref<const RowMatrixXd>& X_mo,
    const Eigen::Ref<const RowMatrixXd>& M_thc,
    const Eigen::Ref<const Eigen::VectorXd>& f_vec
) {
    const int norb = static_cast<int>(X_mo.rows());
    const int nk = static_cast<int>(X_mo.cols());

    std::vector<int> act;
    act.reserve(norb);
    bool all_nonneg = true;
    for (int p = 0; p < norb; ++p) {
        if (std::abs(f_vec[p]) > 1e-14) {
            act.push_back(p);
            if (f_vec[p] < 0.0) all_nonneg = false;
        }
    }
    if (act.empty()) {
        return RowMatrixXd::Zero(norb, norb);
    }

    const int n_act = static_cast<int>(act.size());
    RowMatrixXd D_KL(nk, nk);
    if (all_nonneg) {
        RowMatrixXd X_scaled(n_act, nk);
        for (int i = 0; i < n_act; ++i) {
            const int p = act[i];
            X_scaled.row(i) = std::sqrt(f_vec[p]) * X_mo.row(p);
        }
        blas_dsyrk_at_a_rm(nk, n_act, 1.0, X_scaled.data(), nk, 0.0, D_KL.data(), nk);
    } else {
        RowMatrixXd X_a(n_act, nk);
        RowMatrixXd X_af(n_act, nk);
        for (int i = 0; i < n_act; ++i) {
            const int p = act[i];
            X_a.row(i) = X_mo.row(p);
            X_af.row(i) = f_vec[p] * X_mo.row(p);
        }
        blas_dgemm_rm(true, false, nk, nk, n_act, 1.0, X_af.data(), nk, X_a.data(), nk, 0.0, D_KL.data(), nk);
    }

    RowMatrixXd W_KL = M_thc.cwiseProduct(D_KL);
    RowMatrixXd Y(norb, nk);
    blas_dgemm_rm(false, false, norb, nk, nk, 1.0, X_mo.data(), nk, W_KL.data(), nk, 0.0, Y.data(), nk);
    RowMatrixXd K_mat(norb, norb);
    blas_dgemm_rm(false, true, norb, norb, nk, 1.0, Y.data(), nk, X_mo.data(), nk, 0.0, K_mat.data(), norb);
    return K_mat;
}

namespace {

struct SkewWorkspace {
    int norb = 0;
    bool is_zero = true;
    Eigen::VectorXd w;
    ColMatrixXcd V;
    ColMatrixXcd tmp1;
    ColMatrixXcd tmp2;
    ColMatrixXcd M_c;

    explicit SkewWorkspace(int n)
        : norb(n),
          w(n),
          V(n, n),
          tmp1(n, n),
          tmp2(n, n),
          M_c(n, n) {}

    void compute_expm(const RowMatrixXd& K, RowMatrixXd& U) {
        if (norb == 0) return;
        double max_k = 0.0;
        for (int i = 0; i < norb * norb; ++i) {
            max_k = std::max(max_k, std::abs(K.data()[i]));
        }
        if (max_k == 0.0) {
            is_zero = true;
            U.setIdentity(norb, norb);
            return;
        }
        is_zero = false;
        // H = i * K is complex Hermitian
        for (int q = 0; q < norb; ++q) {
            for (int p = 0; p < norb; ++p) {
                V(p, q) = std::complex<double>(0.0, K(p, q));
            }
        }
        int info = oblas_zheevd('V', 'U', norb, V.data(), norb, w.data());
        if (info != 0) {
            throw std::runtime_error("oblas_zheevd failed in SkewWorkspace::compute_expm");
        }

        // tmp1 = V * diag(exp(-i * w))
        for (int k = 0; k < norb; ++k) {
            const std::complex<double> ew_neg(std::cos(w[k]), -std::sin(w[k]));
            tmp1.col(k) = V.col(k) * ew_neg;
        }

        const std::complex<double> one(1.0, 0.0);
        const std::complex<double> zero(0.0, 0.0);
        cblas_zgemm(
            CblasColMajor, CblasNoTrans, CblasConjTrans,
            norb, norb, norb,
            &one, tmp1.data(), norb,
            V.data(), norb,
            &zero, tmp2.data(), norb
        );
        U.resize(norb, norb);
        for (int p = 0; p < norb; ++p) {
            for (int q = 0; q < norb; ++q) {
                U(p, q) = tmp2(p, q).real();
            }
        }
    }

    void compute_frechet_triu(const RowMatrixXd& M, double* grad_k_out) {
        if (norb <= 1) return;
        if (is_zero) {
            int idx = 0;
            for (int p = 0; p < norb; ++p) {
                for (int q = p + 1; q < norb; ++q) {
                    grad_k_out[idx++] = M(p, q) - M(q, p);
                }
            }
            return;
        }

        for (int q = 0; q < norb; ++q) {
            for (int p = 0; p < norb; ++p) {
                M_c(p, q) = std::complex<double>(M(p, q), 0.0);
            }
        }

        const std::complex<double> one(1.0, 0.0);
        const std::complex<double> zero(0.0, 0.0);
        // tmp1 = M_c * V
        cblas_zgemm(
            CblasColMajor, CblasNoTrans, CblasNoTrans,
            norb, norb, norb,
            &one, M_c.data(), norb,
            V.data(), norb,
            &zero, tmp1.data(), norb
        );
        // tmp2 = V^H * tmp1 = V^H * M * V
        cblas_zgemm(
            CblasColMajor, CblasConjTrans, CblasNoTrans,
            norb, norb, norb,
            &one, V.data(), norb,
            tmp1.data(), norb,
            &zero, tmp2.data(), norb
        );

        std::vector<std::complex<double>> ew(norb);
        for (int a = 0; a < norb; ++a) {
            ew[a] = std::complex<double>(std::cos(w[a]), std::sin(w[a]));
        }

        const std::complex<double> minus_i(0.0, -1.0);
        const std::complex<double> plus_i(0.0, 1.0);
        for (int b = 0; b < norb; ++b) {
            for (int a = 0; a < norb; ++a) {
                const double dw = w[a] - w[b];
                std::complex<double> D;
                if (std::abs(dw) > 1e-7) {
                    D = (ew[a] - ew[b]) / dw;
                } else {
                    D = plus_i * ew[a] * std::complex<double>(1.0, -0.5 * dw);
                }
                tmp2(a, b) *= (minus_i * D);
            }
        }

        // tmp1 = V * tmp2
        cblas_zgemm(
            CblasColMajor, CblasNoTrans, CblasNoTrans,
            norb, norb, norb,
            &one, V.data(), norb,
            tmp2.data(), norb,
            &zero, tmp1.data(), norb
        );
        // M_c = tmp1 * V^H
        cblas_zgemm(
            CblasColMajor, CblasNoTrans, CblasConjTrans,
            norb, norb, norb,
            &one, tmp1.data(), norb,
            V.data(), norb,
            &zero, M_c.data(), norb
        );

        int idx = 0;
        for (int p = 0; p < norb; ++p) {
            for (int q = p + 1; q < norb; ++q) {
                grad_k_out[idx++] = M_c(p, q).real() - M_c(q, p).real();
            }
        }
    }
};

/**
 * Deterministic O(N^3) localized frontier rotation + spin-separated occupation guess (Seed 2).
 * Matches _deterministic_localized_seed_rotation in static_scf.py via LAPACKE_dgeqp3 + SVD polar factor.
 */
std::tuple<RowMatrixXd, Eigen::VectorXd, Eigen::VectorXd> deterministic_localized_seed_rotation(
    const RowMatrixXd& X0,
    const RowMatrixXd& h0,
    int n_alpha,
    int n_beta
) {
    const int norb = static_cast<int>(X0.rows());
    const int nk = static_cast<int>(X0.cols());
    const bool is_singlet = (n_alpha == n_beta);
    const int k_w = std::min({3, n_beta, norb - n_alpha});
    const int k_lo = is_singlet ? (n_beta - k_w) : std::max(0, n_beta - 3);
    const int k_hi = is_singlet ? (n_alpha + k_w) : std::min(norb, n_alpha + 3);
    const int m_sub = k_hi - k_lo;

    RowMatrixXd U_seed = RowMatrixXd::Identity(norb, norb);
    if (m_sub > 1) {
        ColMatrixXd X_sub = X0.middleRows(k_lo, m_sub); // (m_sub, nk) in ColMajor
        std::vector<int> jpvt(nk, 0);
        Eigen::VectorXd tau(std::min(m_sub, nk));
        oblas_dgeqp3(m_sub, nk, X_sub.data(), m_sub, jpvt.data(), tau.data());

        // Re-extract original unfactored columns at top m_sub pivots
        Eigen::MatrixXd P_pts(m_sub, m_sub);
        for (int j = 0; j < m_sub; ++j) {
            const int col_idx = static_cast<int>(jpvt[j]) - 1;
            for (int i = 0; i < m_sub; ++i) {
                P_pts(i, j) = X0(k_lo + i, col_idx);
            }
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(P_pts, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::MatrixXd U_loc = svd.matrixU() * svd.matrixV().transpose();
        Eigen::MatrixXd H_loc_sub = U_loc.transpose() * h0.block(k_lo, k_lo, m_sub, m_sub) * U_loc;
        std::vector<int> loc_ord(m_sub);
        std::iota(loc_ord.begin(), loc_ord.end(), 0);
        std::stable_sort(loc_ord.begin(), loc_ord.end(), [&](int a, int b) {
            return H_loc_sub(a, a) < H_loc_sub(b, b);
        });
        Eigen::MatrixXd U_sorted(m_sub, m_sub);
        for (int c = 0; c < m_sub; ++c) {
            U_sorted.col(c) = U_loc.col(loc_ord[c]);
        }
        if (U_sorted.determinant() < 0.0) {
            U_sorted.col(m_sub - 1) *= -1.0;
        }
        U_seed.block(k_lo, k_lo, m_sub, m_sub) = U_sorted;
    }

    const RowMatrixXd H_rot = U_seed.transpose() * h0 * U_seed;
    const Eigen::VectorXd h_loc = H_rot.diagonal();

    if (is_singlet) {
        Eigen::VectorXd t0 = Eigen::VectorXd::Constant(norb, 5.0);
        t0.head(n_alpha).setConstant(-5.0);
        if (m_sub == 1) {
            t0[k_lo] = 0.0;
        } else if (m_sub > 1) {
            const auto h_seg = h_loc.segment(k_lo, m_sub);
            const double h_mean = h_seg.mean();
            const double h_span = std::max(h_seg.maxCoeff() - h_seg.minCoeff(), 0.20);
            for (int i = 0; i < m_sub; ++i) {
                t0[k_lo + i] = 0.16 * (h_loc[k_lo + i] - h_mean) / h_span;
            }
        }
        return {U_seed, t0, t0};
    } else {
        Eigen::VectorXd ta0 = Eigen::VectorXd::Constant(norb, 5.0);
        Eigen::VectorXd tb0 = Eigen::VectorXd::Constant(norb, 5.0);
        ta0.head(k_lo).setConstant(-5.0);
        tb0.head(k_lo).setConstant(-5.0);

        RowMatrixXd H_sub = H_rot.block(k_lo, k_lo, m_sub, m_sub);
        for (int i = 0; i < m_sub; ++i) H_sub(i, i) = 0.0;

        Eigen::VectorXd hop_norm(m_sub);
        double min_h_sub = std::numeric_limits<double>::infinity();
        for (int i = 0; i < m_sub; ++i) {
            hop_norm[i] = H_sub.row(i).norm();
            min_h_sub = std::min(min_h_sub, h_loc[k_lo + i]);
        }

        const int n_unpaired = n_alpha - n_beta;
        std::vector<int> sub_order(m_sub);
        std::iota(sub_order.begin(), sub_order.end(), 0);
        std::sort(sub_order.begin(), sub_order.end(), [&](int a, int b) {
            const double sa = hop_norm[a] + 0.1 * (h_loc[k_lo + a] - min_h_sub);
            const double sb = hop_norm[b] + 0.1 * (h_loc[k_lo + b] - min_h_sub);
            return sa < sb;
        });

        std::set<int> rad_set;
        for (int i = 0; i < std::min(n_unpaired, m_sub); ++i) {
            const int idx = k_lo + sub_order[i];
            rad_set.insert(idx);
            ta0[idx] = -5.0;
            tb0[idx] = +5.0;
        }

        std::vector<int> rem_idx;
        for (int i = 0; i < m_sub; ++i) {
            const int idx = k_lo + i;
            if (rad_set.find(idx) == rad_set.end()) {
                rem_idx.push_back(idx);
            }
        }
        std::sort(rem_idx.begin(), rem_idx.end(), [&](int a, int b) {
            return h_loc[a] < h_loc[b];
        });

        const int n_pairs_rem = n_beta - k_lo;
        for (int rank = 0; rank < static_cast<int>(rem_idx.size()); ++rank) {
            const int idx = rem_idx[rank];
            if (rank < n_pairs_rem - 1) {
                ta0[idx] = -3.0;
                tb0[idx] = -3.0;
            } else if (rank <= n_pairs_rem) {
                const double val = -0.15 + 0.30 * static_cast<double>(rank - (n_pairs_rem - 1));
                ta0[idx] = val;
                tb0[idx] = val;
            } else {
                ta0[idx] = +4.0;
                tb0[idx] = +4.0;
            }
        }
        return {U_seed, ta0, tb0};
    }
}

/**
 * High-performance Fock-Curvature Preconditioned L-BFGS Optimizer (Strong Wolfe line search, history m=12).
 */
struct PreconditionedLBFGSResult {
    double f_opt = 0.0;
    Eigen::VectorXd x_opt;
    int iters = 0;
    int fevals = 0;
};

template <typename EvalFunc, typename EarlyStopFunc>
PreconditionedLBFGSResult minimize_preconditioned_lbfgs(
    EvalFunc&& eval_fg,
    EarlyStopFunc&& should_early_stop,
    const Eigen::VectorXd& x0,
    int max_iter,
    int m_hist,
    double gtol,
    double ftol,
    bool use_precond
) {
    const int dim = static_cast<int>(x0.size());
    Eigen::VectorXd x = x0;
    Eigen::VectorXd g(dim);
    Eigen::VectorXd diag_h0 = Eigen::VectorXd::Ones(dim);

    int fevals = 0;
    double f = eval_fg(x, g, diag_h0);
    ++fevals;

    double best_f = f;
    Eigen::VectorXd best_x = x;

    if (dim == 0 || g.lpNorm<Eigen::Infinity>() <= gtol) {
        return {best_f, best_x, 0, fevals};
    }

    auto make_pscale = [&](const Eigen::VectorXd& h0_v) {
        Eigen::VectorXd ps = Eigen::VectorXd::Ones(dim);
        if (use_precond) {
            for (int i = 0; i < dim; ++i) {
                ps[i] = std::sqrt(std::clamp(h0_v[i], 1e-4, 250.0));
            }
        }
        return ps;
    };

    Eigen::VectorXd p_scale = make_pscale(diag_h0);

    struct HistPair {
        Eigen::VectorXd s_hat;
        Eigen::VectorXd y_hat;
        double rho;
    };
    std::deque<HistPair> hist;

    Eigen::VectorXd g_hat = p_scale.cwiseProduct(g);
    Eigen::VectorXd d_hat(dim);
    Eigen::VectorXd d_x(dim);
    Eigen::VectorXd q_vec(dim);
    Eigen::VectorXd x_probe(dim);
    Eigen::VectorXd g_probe(dim);
    Eigen::VectorXd h0_probe(dim);
    Eigen::VectorXd x_acc_vec(dim);
    Eigen::VectorXd g_acc_vec(dim);
    Eigen::VectorXd h0_acc_vec(dim);
    std::vector<double> alpha_hist(m_hist);

    int it = 0;
    for (; it < max_iter; ++it) {
        if (g.lpNorm<Eigen::Infinity>() <= gtol) {
            break;
        }
        if (it >= 8 && should_early_stop(it, f, x, g)) {
            break;
        }

        // Refresh diagonal metric P = sqrt(H0) at 24-step epoch boundaries
        if (use_precond && it > 0 && (it % 24 == 0)) {
            hist.clear();
            p_scale = make_pscale(diag_h0);
            g_hat = p_scale.cwiseProduct(g);
        }

        q_vec = g_hat;
        const int cur_m = static_cast<int>(hist.size());
        for (int i = cur_m - 1; i >= 0; --i) {
            alpha_hist[i] = hist[i].rho * hist[i].s_hat.dot(q_vec);
            q_vec.noalias() -= alpha_hist[i] * hist[i].y_hat;
        }

        double gamma = 1.0;
        if (cur_m > 0) {
            const auto& last = hist.back();
            const double sy = last.s_hat.dot(last.y_hat);
            const double yy = last.y_hat.squaredNorm();
            if (yy > 1e-30 && sy > 0.0) {
                gamma = std::clamp(sy / yy, 1e-4, 1e4);
            }
        }
        Eigen::VectorXd r_vec = gamma * q_vec;
        for (int i = 0; i < cur_m; ++i) {
            const double beta = hist[i].rho * hist[i].y_hat.dot(r_vec);
            r_vec.noalias() += hist[i].s_hat * (alpha_hist[i] - beta);
        }
        d_hat = -r_vec;

        double dg0 = d_hat.dot(g_hat);
        if (!(dg0 < -1e-15 * d_hat.norm() * g_hat.norm())) {
            hist.clear();
            d_hat = -g_hat;
            dg0 = d_hat.dot(g_hat);
            if (!(dg0 < 0.0)) break;
        }

        d_x = p_scale.cwiseProduct(d_hat);
        const double dx_inf = d_x.lpNorm<Eigen::Infinity>();

        double step = 1.0;
        if ((it % 24 == 0) && dx_inf > 1.0) {
            step = 1.0 / dx_inf;
        } else if (dx_inf > 8.0) {
            step = 8.0 / dx_inf;
        }

        // Cubic-interpolated Weak Wolfe line search (c1 = 1e-4, c2 = 0.9)
        constexpr double c1 = 1e-4;
        constexpr double c2 = 0.9;
        const double f0 = f;

        bool have_armijo = false;
        double best_ls_f = f0;

        auto eval_phi = [&](double a, double& fa, double& dfa) {
            x_probe = x + a * d_x;
            fa = eval_fg(x_probe, g_probe, h0_probe);
            ++fevals;
            if (fa < best_f) {
                best_f = fa;
                best_x = x_probe;
            }
            dfa = d_x.dot(g_probe);
            if (fa <= f0 + c1 * a * dg0 && fa < best_ls_f) {
                have_armijo = true;
                best_ls_f = fa;
                x_acc_vec = x_probe;
                g_acc_vec = g_probe;
                h0_acc_vec = h0_probe;
            }
        };

        auto cubic_interp = [](double a0, double f_a0, double df_a0,
                               double a1, double f_a1, double df_a1) -> double {
            const double da = a1 - a0;
            if (std::abs(da) < 1e-20) return 0.5 * (a0 + a1);
            const double d1 = df_a0 + df_a1 - 3.0 * (f_a0 - f_a1) / da;
            const double disc = d1 * d1 - df_a0 * df_a1;
            const double lo_b = std::min(a0, a1) + 0.1 * std::abs(da);
            const double hi_b = std::max(a0, a1) - 0.1 * std::abs(da);
            if (disc >= 0.0) {
                const double d2 = (da > 0.0 ? 1.0 : -1.0) * std::sqrt(disc);
                const double denom = df_a1 - df_a0 + 2.0 * d2;
                if (std::abs(denom) > 1e-24) {
                    const double a_c = a1 - da * (df_a1 + d2 - d1) / denom;
                    if (a_c >= lo_b && a_c <= hi_b) return a_c;
                }
            }
            return 0.5 * (a0 + a1);
        };

        double a_lo = 0.0, f_lo = f0, df_lo = dg0;
        double a_hi = 0.0, f_hi = f0, df_hi = dg0;
        bool bracketed = false;
        bool wolfe_ok = false;
        double a_curr = step;

        for (int ls_it = 0; ls_it < 20; ++ls_it) {
            double f_curr = 0.0, df_curr = 0.0;
            eval_phi(a_curr, f_curr, df_curr);

            if (!bracketed) {
                if (f_curr > f0 + c1 * a_curr * dg0 || (ls_it > 0 && f_curr >= f_lo)) {
                    bracketed = true;
                    a_hi = a_curr; f_hi = f_curr; df_hi = df_curr;
                } else if (df_curr >= c2 * dg0) {
                    x_acc_vec = x_probe;
                    g_acc_vec = g_probe;
                    h0_acc_vec = h0_probe;
                    best_ls_f = f_curr;
                    wolfe_ok = true;
                    break;
                } else {
                    a_lo = a_curr; f_lo = f_curr; df_lo = df_curr;
                    if (ls_it < 5 && a_curr * dx_inf < 12.0) {
                        a_curr *= 2.5;
                        continue;
                    } else {
                        wolfe_ok = true;
                        break;
                    }
                }
            } else {
                if (f_curr > f0 + c1 * a_curr * dg0 || f_curr >= f_lo) {
                    a_hi = a_curr; f_hi = f_curr; df_hi = df_curr;
                } else if (df_curr >= c2 * dg0) {
                    x_acc_vec = x_probe;
                    g_acc_vec = g_probe;
                    h0_acc_vec = h0_probe;
                    best_ls_f = f_curr;
                    wolfe_ok = true;
                    break;
                } else {
                    if (df_curr * (a_hi - a_lo) >= 0.0) {
                        a_hi = a_lo; f_hi = f_lo; df_hi = df_lo;
                    }
                    a_lo = a_curr; f_lo = f_curr; df_lo = df_curr;
                }
            }

            if (bracketed) {
                if (std::abs(a_hi - a_lo) * dx_inf < 1e-14) break;
                a_curr = cubic_interp(a_lo, f_lo, df_lo, a_hi, f_hi, df_hi);
            }
        }

        if (!wolfe_ok && !have_armijo) {
            if (!hist.empty()) {
                hist.clear();
                continue;
            }
            if ((p_scale.array() - 1.0).abs().maxCoeff() > 1e-12) {
                p_scale.setOnes();
                g_hat = g;
                continue;
            }
            break;
        }

        Eigen::VectorXd s_hat = (x_acc_vec - x).cwiseQuotient(p_scale);
        Eigen::VectorXd g_hat_new = p_scale.cwiseProduct(g_acc_vec);
        Eigen::VectorXd y_hat = g_hat_new - g_hat;
        const double sy = s_hat.dot(y_hat);
        if (sy > 1e-14 * s_hat.norm() * y_hat.norm()) {
            if (static_cast<int>(hist.size()) >= m_hist) {
                hist.pop_front();
            }
            hist.push_back({std::move(s_hat), std::move(y_hat), 1.0 / sy});
        }

        const double rel_df = (f - best_ls_f) / std::max({std::abs(f), std::abs(best_ls_f), 1.0});
        x.swap(x_acc_vec);
        g.swap(g_acc_vec);
        g_hat.swap(g_hat_new);
        diag_h0.swap(h0_acc_vec);
        f = best_ls_f;

        if (rel_df <= ftol && g.lpNorm<Eigen::Infinity>() <= 1e-5) {
            break;
        }
    }

    return {best_f, best_x, it, fevals};
}

} // namespace

std::pair<RowMatrixXd, Eigen::VectorXd> skew_expm_and_frechet(
    const Eigen::Ref<const RowMatrixXd>& K,
    const Eigen::Ref<const RowMatrixXd>& M
) {
    const int norb = static_cast<int>(K.rows());
    const int n_rot = norb * (norb - 1) / 2;
    SkewWorkspace ws(norb);
    RowMatrixXd U(norb, norb);
    Eigen::VectorXd grad_k(n_rot);
    RowMatrixXd K_copy = K;
    RowMatrixXd M_copy = M;
    ws.compute_expm(K_copy, U);
    ws.compute_frechet_triu(M_copy, grad_k.data());
    return {U, grad_k};
}

StaticSCFOutput solve_static_scf_cpp(
    const Eigen::Ref<const RowMatrixXd>& C0,
    const Eigen::Ref<const RowMatrixXd>& h1_ao,
    const Eigen::Ref<const RowMatrixXd>& X_ao,
    const Eigen::Ref<const RowMatrixXd>& M_thc,
    double enuc,
    int n_alpha,
    int n_beta,
    double e_hf_init,
    const Eigen::Ref<const Eigen::VectorXd>& mo_energy,
    const StaticSCFConfig& cfg
) {
    const int nao = static_cast<int>(C0.rows());
    const int norb = static_cast<int>(C0.cols());
    const int nk = static_cast<int>(X_ao.cols());
    const bool is_singlet = (n_alpha == n_beta);

    auto [lam_q, alpha_q] = build_phi2_laplace_quadrature(cfg.n_quad_phi2, cfg.L_exp_phi2);
    const int n_quad = cfg.n_quad_phi2;
    const int n_rot = norb * (norb - 1) / 2;
    const int dim_t = is_singlet ? norb : (2 * norb);
    const int dim_x = n_rot + dim_t;

    // Precompute initial MO integrals on C0
    RowMatrixXd tmp_h(nao, norb);
    blas_dgemm_rm(false, false, nao, norb, nao, 1.0, h1_ao.data(), nao, C0.data(), norb, 0.0, tmp_h.data(), norb);
    RowMatrixXd h0(norb, norb);
    blas_dgemm_rm(true, false, norb, norb, nao, 1.0, C0.data(), norb, tmp_h.data(), norb, 0.0, h0.data(), norb);

    RowMatrixXd X0(norb, nk);
    blas_dgemm_rm(true, false, norb, nk, nao, 1.0, C0.data(), norb, X_ao.data(), nk, 0.0, X0.data(), nk);

    struct SeedSpec {
        RowMatrixXd C_base;
        Eigen::VectorXd x_init;
        int max_it;
    };
    std::vector<SeedSpec> seeds;
    seeds.reserve(4);

    // Seed 0: Saturated canonical RHF/ROHF idempotent state
    {
        Eigen::VectorXd x_s0 = Eigen::VectorXd::Zero(dim_x);
        if (is_singlet) {
            Eigen::VectorXd t0 = Eigen::VectorXd::Constant(norb, 38.0);
            t0.head(n_alpha).setConstant(-38.0);
            x_s0.tail(norb) = t0;
        } else {
            Eigen::VectorXd ta0 = Eigen::VectorXd::Constant(norb, 38.0);
            ta0.head(n_alpha).setConstant(-38.0);
            Eigen::VectorXd tb0 = Eigen::VectorXd::Constant(norb, 38.0);
            tb0.head(n_beta).setConstant(-38.0);
            x_s0.segment(n_rot, norb) = ta0;
            x_s0.segment(n_rot + norb, norb) = tb0;
        }
        seeds.push_back({C0, std::move(x_s0), 120});
    }

    // Seed 1: Frontier-mixed canonical seed
    {
        Eigen::VectorXd x_s1 = Eigen::VectorXd::Zero(dim_x);
        if (n_rot > 0) {
            int idx = 0;
            for (int p = 0; p < norb; ++p) {
                for (int q = p + 1; q < norb; ++q) {
                    if (std::abs(p - n_alpha) <= 3 || std::abs(q - n_alpha) <= 3) {
                        x_s1[idx] = 0.14 * std::sin(1.7 * static_cast<double>(p + 1) + 2.3 * static_cast<double>(q + 1));
                    }
                    ++idx;
                }
            }
        }
        if (is_singlet) {
            const int k_op = std::min({3, n_alpha, norb - n_alpha});
            Eigen::VectorXd t1 = Eigen::VectorXd::Constant(norb, 4.5);
            t1.head(n_alpha - k_op).setConstant(-4.5);
            if (k_op > 0) {
                const int len_w = 2 * k_op;
                for (int i = 0; i < len_w; ++i) {
                    t1[n_alpha - k_op + i] = 0.08 * (-1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(len_w - 1));
                }
            }
            x_s1.tail(norb) = t1;
        } else {
            Eigen::VectorXd eps_guess = mo_energy;
            if (eps_guess.size() != norb) {
                eps_guess = h0.diagonal();
            }
            const double ehomo_a = eps_guess[std::max(0, n_alpha - 1)];
            Eigen::VectorXd ta1 = (eps_guess.array() - ehomo_a) / 0.14;
            Eigen::VectorXd tb1 = Eigen::VectorXd::Constant(norb, 4.5);
            if (n_beta > 0) {
                const double ehomo_b = eps_guess[std::max(0, n_beta - 1)];
                tb1 = (eps_guess.array() - ehomo_b) / 0.14;
            }
            x_s1.segment(n_rot, norb) = ta1;
            x_s1.segment(n_rot + norb, norb) = tb1;
        }
        seeds.push_back({C0, std::move(x_s1), 280});
    }

    // Seed 2: Spatially localized frontier basis C0 @ U_loc
    if (n_beta > 0 && norb > 1) {
        auto [U_loc, ta2, tb2] = deterministic_localized_seed_rotation(X0, h0, n_alpha, n_beta);
        Eigen::VectorXd x_s2 = Eigen::VectorXd::Zero(dim_x);
        if (is_singlet) {
            x_s2.tail(norb) = ta2;
        } else {
            x_s2.segment(n_rot, norb) = ta2;
            x_s2.segment(n_rot + norb, norb) = tb2;
        }
        RowMatrixXd C_s2(nao, norb);
        blas_dgemm_rm(false, false, nao, norb, norb, 1.0, C0.data(), norb, U_loc.data(), norb, 0.0, C_s2.data(), norb);
        seeds.push_back({std::move(C_s2), std::move(x_s2), 280});
    }

    // Seed 3: Core-Hamiltonian eigenbasis seed for singlets
    if (is_singlet && n_alpha >= 2 && (norb - n_alpha) >= 2) {
        RowMatrixXd h0_sym = 0.5 * (h0 + h0.transpose());
        ColMatrixXd U_hc = h0_sym;
        Eigen::VectorXd evals(norb);
        oblas_dsyevd('V', 'U', norb, U_hc.data(), norb, evals.data());
        if (U_hc.determinant() < 0.0) {
            U_hc.col(norb - 1) *= -1.0;
        }
        RowMatrixXd U_hc_rm = U_hc;
        RowMatrixXd C_s3(nao, norb);
        blas_dgemm_rm(false, false, nao, norb, norb, 1.0, C0.data(), norb, U_hc_rm.data(), norb, 0.0, C_s3.data(), norb);

        Eigen::VectorXd x_s3 = Eigen::VectorXd::Zero(dim_x);
        const int k_op3 = std::min({2, n_alpha, norb - n_alpha});
        Eigen::VectorXd t3 = Eigen::VectorXd::Constant(norb, 5.0);
        t3.head(n_alpha - k_op3).setConstant(-5.0);
        const int len_w3 = 2 * k_op3;
        for (int i = 0; i < len_w3; ++i) {
            t3[n_alpha - k_op3 + i] = 0.05 * (-1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(len_w3 - 1));
        }
        x_s3.tail(norb) = t3;
        seeds.push_back({std::move(C_s3), std::move(x_s3), 280});
    }

    // Pre-allocate ALL matrix & OpenMP per-thread workspaces ONCE outside optimization loops!
    SkewWorkspace skew_ws(norb);
    RowMatrixXd Krot(norb, norb);
    RowMatrixXd U(norb, norb);
    RowMatrixXd h_base(norb, norb);
    RowMatrixXd X_base(norb, nk);
    RowMatrixXd h_tmp(norb, norb);
    RowMatrixXd h_mo(norb, norb);
    RowMatrixXd X_mo(norb, nk);
    RowMatrixXd Q_mo(norb, nk);
    Eigen::VectorXd d_K(nk);
    Eigen::VectorXd v_K(nk);
    RowMatrixXd J_tmp(norb, nk);
    RowMatrixXd J_mo(norb, norb);

    // Pre-allocated workspace for mean-field Ka / Kb exchange builds
    RowMatrixXd ex_X_scaled(norb, nk);
    RowMatrixXd ex_W_KL(nk, nk);
    RowMatrixXd ex_Y(norb, nk);
    RowMatrixXd Ka(norb, norb);
    RowMatrixXd Kb(norb, norb);

    auto build_thc_k_noalloc = [&](const Eigen::VectorXd& f_vec, RowMatrixXd& K_out) {
        int n_act = 0;
        for (int p = 0; p < norb; ++p) {
            if (f_vec[p] > 1e-14) {
                ex_X_scaled.row(n_act++) = std::sqrt(f_vec[p]) * X_mo.row(p);
            }
        }
        if (n_act == 0) {
            K_out.setZero();
            return;
        }
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, nk, n_act, 1.0, ex_X_scaled.data(), nk, 0.0, ex_W_KL.data(), nk);
        for (int K = 0; K < nk; ++K) {
            ex_W_KL(K, K) *= M_thc(K, K);
            for (int L = K + 1; L < nk; ++L) {
                const double w_kl = ex_W_KL(K, L) * M_thc(K, L);
                ex_W_KL(K, L) = w_kl;
                ex_W_KL(L, K) = w_kl;
            }
        }
        blas_dgemm_rm(false, false, norb, nk, nk, 1.0, X_mo.data(), nk, ex_W_KL.data(), nk, 0.0, ex_Y.data(), nk);
        blas_dgemm_rm(false, true, norb, norb, nk, 1.0, ex_Y.data(), nk, X_mo.data(), nk, 0.0, K_out.data(), norb);
    };

    struct CumulantThreadWS {
        double e_cum_loc = 0.0;
        Eigen::VectorXd dE_dwa_loc;
        Eigen::VectorXd dE_dwb_loc;
        RowMatrixXd G_cum_T_loc; // (norb, nk) RowMajor: row p is nk contiguous doubles!
        RowMatrixXd X_scaled;
        RowMatrixXd W_KL;
        RowMatrixXd X_sub;       // (nd, nk) RowMajor
        RowMatrixXd Y_sub;       // (nd, nk) RowMajor
        Eigen::VectorXd ua, dua, ub, dub;
        std::vector<int> act_u, act_d, act_ub, act_db;

        void init(int norb_in, int nk_in) {
            dE_dwa_loc = Eigen::VectorXd::Zero(norb_in);
            dE_dwb_loc = Eigen::VectorXd::Zero(norb_in);
            G_cum_T_loc = RowMatrixXd::Zero(norb_in, nk_in);
            X_scaled.resize(norb_in, nk_in);
            W_KL.resize(nk_in, nk_in);
            X_sub.resize(norb_in, nk_in);
            Y_sub.resize(norb_in, nk_in);
            ua.resize(norb_in); dua.resize(norb_in);
            ub.resize(norb_in); dub.resize(norb_in);
            act_u.reserve(norb_in); act_d.reserve(norb_in);
            act_ub.reserve(norb_in); act_db.reserve(norb_in);
        }
    };

    const int max_threads = std::max(1, omp_get_max_threads());
    std::vector<CumulantThreadWS> thread_ws(max_threads);
    for (int tid = 0; tid < max_threads; ++tid) {
        thread_ws[tid].init(norb, nk);
    }

    RowMatrixXd G_cum_T(norb, nk);
    RowMatrixXd F_cum(norb, norb);
    RowMatrixXd A_mat(norb, norb);
    RowMatrixXd UA_mat(norb, norb);

    double best_E = 1e18;
    RowMatrixXd best_C = C0;
    Eigen::VectorXd best_na = Eigen::VectorXd::Zero(norb);
    Eigen::VectorXd best_nb = Eigen::VectorXd::Zero(norb);
    int total_iters = 0;
    int total_fevals = 0;

    for (const auto& seed : seeds) {
        blas_dgemm_rm(false, false, nao, norb, nao, 1.0, h1_ao.data(), nao, seed.C_base.data(), norb, 0.0, tmp_h.data(), norb);
        blas_dgemm_rm(true, false, norb, norb, nao, 1.0, seed.C_base.data(), norb, tmp_h.data(), norb, 0.0, h_base.data(), norb);
        blas_dgemm_rm(true, false, norb, nk, nao, 1.0, seed.C_base.data(), norb, X_ao.data(), nk, 0.0, X_base.data(), nk);

        auto eval_fg = [&](const Eigen::VectorXd& x, Eigen::VectorXd& grad, Eigen::VectorXd& h0_diag) -> double {
            Krot.setZero();
            int idx = 0;
            for (int p = 0; p < norb; ++p) {
                for (int q = p + 1; q < norb; ++q) {
                    const double v = x[idx++];
                    Krot(p, q) = v;
                    Krot(q, p) = -v;
                }
            }
            skew_ws.compute_expm(Krot, U);

            if (skew_ws.is_zero) {
                h_mo = h_base;
                X_mo = X_base;
            } else {
                blas_dgemm_rm(false, false, norb, norb, norb, 1.0, h_base.data(), norb, U.data(), norb, 0.0, h_tmp.data(), norb);
                blas_dgemm_rm(true, false, norb, norb, norb, 1.0, U.data(), norb, h_tmp.data(), norb, 0.0, h_mo.data(), norb);
                blas_dgemm_rm(true, false, norb, nk, norb, 1.0, U.data(), norb, X_base.data(), nk, 0.0, X_mo.data(), nk);
            }

            Q_mo = X_mo.cwiseProduct(X_mo);
            const bool use_ph_pauli = (cfg.amp_params.mode == AmplitudeMode::PHPauliS1);

            Eigen::VectorXd na, nb, wa_t, wb_t, wa, dwa, wb, dwb;
            PHPauliCache ph_a, ph_b;
            double wsa = 1.0, wsb = 1.0;
            if (is_singlet) {
                std::tie(na, wa_t, wsa) = map_fermi_occupations(x.tail(norb), static_cast<double>(n_alpha));
                nb = na;
                if (use_ph_pauli) {
                    ph_a = ph_pauli_amplitude_forward(na, Q_mo, cfg.amp_params.ph_eta);
                    wa = ph_a.w;
                    wb = wa;
                } else {
                    std::tie(wa, dwa) = pauli_amplitude_and_deriv(na, cfg.amp_params);
                    wb = wa;
                    dwb = dwa;
                }
            } else {
                std::tie(na, wa_t, wsa) = map_fermi_occupations(x.segment(n_rot, norb), static_cast<double>(n_alpha));
                std::tie(nb, wb_t, wsb) = map_fermi_occupations(x.segment(n_rot + norb, norb), static_cast<double>(n_beta));
                if (use_ph_pauli) {
                    ph_a = ph_pauli_amplitude_forward(na, Q_mo, cfg.amp_params.ph_eta);
                    ph_b = ph_pauli_amplitude_forward(nb, Q_mo, cfg.amp_params.ph_eta);
                    wa = ph_a.w;
                    wb = ph_b.w;
                } else {
                    std::tie(wa, dwa) = pauli_amplitude_and_deriv(na, cfg.amp_params);
                    std::tie(wb, dwb) = pauli_amplitude_and_deriv(nb, cfg.amp_params);
                }
            }

            const Eigen::VectorXd ntot = na + nb;
            cblas_dgemv(CblasRowMajor, CblasTrans, norb, nk, 1.0, Q_mo.data(), nk, ntot.data(), 1, 0.0, d_K.data(), 1);
            cblas_dgemv(CblasRowMajor, CblasNoTrans, nk, nk, 1.0, M_thc.data(), nk, d_K.data(), 1, 0.0, v_K.data(), 1);

            for (int p = 0; p < norb; ++p) {
                J_tmp.row(p) = X_mo.row(p).cwiseProduct(v_K.transpose());
            }
            blas_dgemm_rm(false, true, norb, norb, nk, 1.0, J_tmp.data(), nk, X_mo.data(), nk, 0.0, J_mo.data(), norb);

            build_thc_k_noalloc(na, Ka);
            if (is_singlet) {
                Kb = Ka;
            } else {
                build_thc_k_noalloc(nb, Kb);
            }

            double e_cum = 0.0;
            Eigen::VectorXd dE_dwa = Eigen::VectorXd::Zero(norb);
            Eigen::VectorXd dE_dwb = Eigen::VectorXd::Zero(norb);
            Eigen::VectorXd dE_dna_cum = Eigen::VectorXd::Zero(norb);
            Eigen::VectorXd dE_dnb_cum = Eigen::VectorXd::Zero(norb);
            F_cum.setZero();

            // Since ua[p] = wa[p]^2 * exp(...) <= wa[p]^2 and active threshold is ua > 1e-14,
            // when max(wa) <= 1e-7, ua[p] <= 1e-14 identically across all 28 Laplace nodes!
            const bool has_wa = (wa.maxCoeff() > 1e-13);
            const bool has_wb = (wb.maxCoeff() > 1e-13);

            if (has_wa && has_wb) {
                G_cum_T.setZero();
                const int eff_threads = (nk >= 140) ? max_threads : 1;
                {
                    ScopedSingleThreadBLAS blas_guard;
                    #pragma omp parallel num_threads(eff_threads) if(eff_threads > 1)
                    {
                        const int tid = omp_get_thread_num();
                        auto& ws = thread_ws[tid];
                        ws.e_cum_loc = 0.0;
                        ws.dE_dwa_loc.setZero();
                        ws.dE_dwb_loc.setZero();
                        ws.G_cum_T_loc.setZero();

                        #pragma omp for schedule(static)
                        for (int t = 0; t < n_quad; ++t) {
                            const double lam_t = lam_q[t];
                            const double a_t = alpha_q[t];

                            ws.act_u.clear();
                            ws.act_d.clear();
                            for (int p = 0; p < norb; ++p) {
                                const double wp = wa[p];
                                if (wp <= 1e-13) {
                                    ws.ua[p] = 0.0;
                                    ws.dua[p] = 0.0;
                                    continue;
                                }
                                const double wp3 = wp * wp * wp;
                                const double ea = std::exp(-std::clamp(lam_t * wp3, 0.0, 100.0));
                                ws.ua[p] = (wp * wp) * ea;
                                ws.dua[p] = (2.0 * wp - 3.0 * lam_t * wp * wp3) * ea;
                                if (ws.ua[p] > 1e-14) ws.act_u.push_back(p);
                                ws.act_d.push_back(p);
                            }

                            if (is_singlet) {
                                if (ws.act_u.empty() || ws.act_d.empty()) continue;
                                const int nu = static_cast<int>(ws.act_u.size());
                                for (int i = 0; i < nu; ++i) {
                                    const int p = ws.act_u[i];
                                    ws.X_scaled.row(i) = std::sqrt(ws.ua[p]) * X_mo.row(p);
                                }
                                cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, nk, nu, 1.0, ws.X_scaled.data(), nk, 0.0, ws.W_KL.data(), nk);
                                for (int K = 0; K < nk; ++K) {
                                    ws.W_KL(K, K) *= M_thc(K, K);
                                    for (int L = K + 1; L < nk; ++L) {
                                        const double w_kl = ws.W_KL(K, L) * M_thc(K, L);
                                        ws.W_KL(K, L) = w_kl;
                                        ws.W_KL(L, K) = w_kl;
                                    }
                                }

                                const int nd = static_cast<int>(ws.act_d.size());
                                for (int j = 0; j < nd; ++j) {
                                    ws.X_sub.row(j) = X_mo.row(ws.act_d[j]);
                                }
                                // Wide contiguous RowMajor GEMM: Y_sub (nd x nk) = X_sub (nd x nk) * W_KL (nk x nk)
                                blas_dgemm_rm(false, false, nd, nk, nk, 1.0, ws.X_sub.data(), nk, ws.W_KL.data(), nk, 0.0, ws.Y_sub.data(), nk);

                                for (int j = 0; j < nd; ++j) {
                                    const int p = ws.act_d[j];
                                    const double* xp = X_mo.data() + static_cast<std::size_t>(p) * nk;
                                    const double* yj = ws.Y_sub.data() + static_cast<std::size_t>(j) * nk;
                                    const double k_pp = cblas_ddot(nk, xp, 1, yj, 1);
                                    ws.e_cum_loc -= a_t * ws.ua[p] * k_pp;
                                    ws.dE_dwa_loc[p] -= a_t * ws.dua[p] * k_pp;
                                    const double coeff = -2.0 * a_t * ws.ua[p];
                                    if (std::abs(coeff) > 0.0) {
                                        cblas_daxpy(nk, coeff, yj, 1, ws.G_cum_T_loc.data() + static_cast<std::size_t>(p) * nk, 1);
                                    }
                                }
                            } else {
                                ws.act_ub.clear();
                                ws.act_db.clear();
                                for (int p = 0; p < norb; ++p) {
                                    const double wp = wb[p];
                                    if (wp <= 1e-13) {
                                        ws.ub[p] = 0.0;
                                        ws.dub[p] = 0.0;
                                        continue;
                                    }
                                    const double wp3 = wp * wp * wp;
                                    const double eb = std::exp(-std::clamp(lam_t * wp3, 0.0, 100.0));
                                    ws.ub[p] = (wp * wp) * eb;
                                    ws.dub[p] = (2.0 * wp - 3.0 * lam_t * wp * wp3) * eb;
                                    if (ws.ub[p] > 1e-14) ws.act_ub.push_back(p);
                                    ws.act_db.push_back(p);
                                }

                                if (!ws.act_ub.empty() && !ws.act_d.empty()) {
                                    const int nub = static_cast<int>(ws.act_ub.size());
                                    for (int i = 0; i < nub; ++i) {
                                        const int p = ws.act_ub[i];
                                        ws.X_scaled.row(i) = std::sqrt(ws.ub[p]) * X_mo.row(p);
                                    }
                                    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, nk, nub, 1.0, ws.X_scaled.data(), nk, 0.0, ws.W_KL.data(), nk);
                                    for (int K = 0; K < nk; ++K) {
                                        ws.W_KL(K, K) *= M_thc(K, K);
                                        for (int L = K + 1; L < nk; ++L) {
                                            const double w_kl = ws.W_KL(K, L) * M_thc(K, L);
                                            ws.W_KL(K, L) = w_kl;
                                            ws.W_KL(L, K) = w_kl;
                                        }
                                    }

                                    const int nda = static_cast<int>(ws.act_d.size());
                                    for (int j = 0; j < nda; ++j) {
                                        ws.X_sub.row(j) = X_mo.row(ws.act_d[j]);
                                    }
                                    blas_dgemm_rm(false, false, nda, nk, nk, 1.0, ws.X_sub.data(), nk, ws.W_KL.data(), nk, 0.0, ws.Y_sub.data(), nk);

                                    for (int j = 0; j < nda; ++j) {
                                        const int p = ws.act_d[j];
                                        const double* xp = X_mo.data() + static_cast<std::size_t>(p) * nk;
                                        const double* yj = ws.Y_sub.data() + static_cast<std::size_t>(j) * nk;
                                        const double k_pp = cblas_ddot(nk, xp, 1, yj, 1);
                                        ws.e_cum_loc -= a_t * ws.ua[p] * k_pp;
                                        ws.dE_dwa_loc[p] -= a_t * ws.dua[p] * k_pp;
                                        const double coeff = -a_t * ws.ua[p];
                                        if (std::abs(coeff) > 0.0) {
                                            cblas_daxpy(nk, coeff, yj, 1, ws.G_cum_T_loc.data() + static_cast<std::size_t>(p) * nk, 1);
                                        }
                                    }
                                }

                                if (!ws.act_u.empty() && !ws.act_db.empty()) {
                                    const int nua = static_cast<int>(ws.act_u.size());
                                    for (int i = 0; i < nua; ++i) {
                                        const int p = ws.act_u[i];
                                        ws.X_scaled.row(i) = std::sqrt(ws.ua[p]) * X_mo.row(p);
                                    }
                                    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, nk, nua, 1.0, ws.X_scaled.data(), nk, 0.0, ws.W_KL.data(), nk);
                                    for (int K = 0; K < nk; ++K) {
                                        ws.W_KL(K, K) *= M_thc(K, K);
                                        for (int L = K + 1; L < nk; ++L) {
                                            const double w_kl = ws.W_KL(K, L) * M_thc(K, L);
                                            ws.W_KL(K, L) = w_kl;
                                            ws.W_KL(L, K) = w_kl;
                                        }
                                    }

                                    const int ndb = static_cast<int>(ws.act_db.size());
                                    for (int j = 0; j < ndb; ++j) {
                                        ws.X_sub.row(j) = X_mo.row(ws.act_db[j]);
                                    }
                                    blas_dgemm_rm(false, false, ndb, nk, nk, 1.0, ws.X_sub.data(), nk, ws.W_KL.data(), nk, 0.0, ws.Y_sub.data(), nk);

                                    for (int j = 0; j < ndb; ++j) {
                                        const int p = ws.act_db[j];
                                        const double* xp = X_mo.data() + static_cast<std::size_t>(p) * nk;
                                        const double* yj = ws.Y_sub.data() + static_cast<std::size_t>(j) * nk;
                                        const double k_pp = cblas_ddot(nk, xp, 1, yj, 1);
                                        ws.dE_dwb_loc[p] -= a_t * ws.dub[p] * k_pp;
                                        const double coeff = -a_t * ws.ub[p];
                                        if (std::abs(coeff) > 0.0) {
                                            cblas_daxpy(nk, coeff, yj, 1, ws.G_cum_T_loc.data() + static_cast<std::size_t>(p) * nk, 1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                for (int tid = 0; tid < eff_threads; ++tid) {
                    e_cum += thread_ws[tid].e_cum_loc;
                    dE_dwa += thread_ws[tid].dE_dwa_loc;
                    dE_dwb += thread_ws[tid].dE_dwb_loc;
                    G_cum_T += thread_ws[tid].G_cum_T_loc;
                }
                if (use_ph_pauli) {
                    if (is_singlet) {
                        ph_pauli_amplitude_backprop(ph_a, 2.0 * dE_dwa, X_mo, Q_mo, dE_dna_cum, G_cum_T);
                    } else {
                        ph_pauli_amplitude_backprop(ph_a, dE_dwa, X_mo, Q_mo, dE_dna_cum, G_cum_T);
                        ph_pauli_amplitude_backprop(ph_b, dE_dwb, X_mo, Q_mo, dE_dnb_cum, G_cum_T);
                    }
                } else {
                    if (is_singlet) {
                        dE_dna_cum = 2.0 * dwa.cwiseProduct(dE_dwa);
                    } else {
                        dE_dna_cum = dwa.cwiseProduct(dE_dwa);
                        dE_dnb_cum = dwb.cwiseProduct(dE_dwb);
                    }
                }
                blas_dgemm_rm(false, true, norb, norb, nk, 1.0, X_mo.data(), nk, G_cum_T.data(), nk, 0.0, F_cum.data(), norb);
            }

            const Eigen::VectorXd h_diag = h_mo.diagonal();
            const Eigen::VectorXd J_diag = J_mo.diagonal();
            const Eigen::VectorXd Ka_diag = Ka.diagonal();
            const Eigen::VectorXd Kb_diag = Kb.diagonal();

            const double e1 = ntot.dot(h_diag);
            const double eJ = 0.5 * ntot.dot(J_diag);
            const double eK_mf = -0.5 * (na.dot(Ka_diag) + nb.dot(Kb_diag));
            const double E_tot = enuc + e1 + eJ + eK_mf + e_cum;

            for (int p = 0; p < norb; ++p) {
                for (int q = 0; q < norb; ++q) {
                    A_mat(p, q) = (h_mo(p, q) + J_mo(p, q)) * ntot[q]
                                - Ka(p, q) * na[q]
                                - Kb(p, q) * nb[q]
                                + F_cum(p, q);
                }
            }

            if (skew_ws.is_zero) {
                UA_mat = 2.0 * A_mat;
            } else {
                blas_dgemm_rm(false, false, norb, norb, norb, 2.0, U.data(), norb, A_mat.data(), norb, 0.0, UA_mat.data(), norb);
            }

            grad.resize(dim_x);
            skew_ws.compute_frechet_triu(UA_mat, grad.data());

            // Diagonal Fock-Curvature Preconditioner H0_rot[p,q] = 1 / max(0.15, 2*|F_pp - F_qq|*|n_p - n_q|)
            h0_diag.resize(dim_x);
            const Eigen::VectorXd F_diag = h_diag + J_diag - 0.5 * (Ka_diag + Kb_diag);
            int r_idx = 0;
            for (int p = 0; p < norb; ++p) {
                for (int q = p + 1; q < norb; ++q) {
                    const double curv = std::max(0.15, 2.0 * std::abs(F_diag[p] - F_diag[q]) * std::abs(ntot[p] - ntot[q]));
                    h0_diag[r_idx++] = 1.0 / curv;
                }
            }

            if (is_singlet) {
                const Eigen::VectorXd dE_dn = 2.0 * (h_diag + J_diag - Ka_diag) + dE_dna_cum;
                const double shift = wa_t.dot(dE_dn) / wsa;
                grad.tail(norb) = -wa_t.array() * (dE_dn.array() - shift);
                for (int p = 0; p < norb; ++p) {
                    if (wa_t[p] > 5e-8) {
                        const double curv_t = std::abs(2.0 * na[p] - 1.0) * wa_t[p] * std::abs(dE_dn[p] - shift)
                                            + 2.0 * wa_t[p] * wa_t[p];
                        h0_diag[n_rot + p] = std::clamp(1.0 / std::max(curv_t, 2e-4), 1.0, 16.0);
                    } else {
                        h0_diag[n_rot + p] = 1.0;
                    }
                }
            } else {
                const Eigen::VectorXd dE_dna = h_diag + J_diag - Ka_diag + dE_dna_cum;
                const Eigen::VectorXd dE_dnb = h_diag + J_diag - Kb_diag + dE_dnb_cum;
                double sh_a = 0.0, sh_b = 0.0;
                if (n_alpha > 0) {
                    sh_a = wa_t.dot(dE_dna) / wsa;
                    grad.segment(n_rot, norb) = -wa_t.array() * (dE_dna.array() - sh_a);
                } else {
                    grad.segment(n_rot, norb).setZero();
                }
                if (n_beta > 0) {
                    sh_b = wb_t.dot(dE_dnb) / wsb;
                    grad.segment(n_rot + norb, norb) = -wb_t.array() * (dE_dnb.array() - sh_b);
                } else {
                    grad.segment(n_rot + norb, norb).setZero();
                }
                for (int p = 0; p < norb; ++p) {
                    if (wa_t[p] > 5e-8) {
                        const double ca = std::abs(2.0 * na[p] - 1.0) * wa_t[p] * std::abs(dE_dna[p] - sh_a)
                                        + 2.0 * wa_t[p] * wa_t[p];
                        h0_diag[n_rot + p] = std::clamp(1.0 / std::max(ca, 2e-4), 1.0, 16.0);
                    } else {
                        h0_diag[n_rot + p] = 1.0;
                    }
                    if (wb_t[p] > 5e-8) {
                        const double cb = std::abs(2.0 * nb[p] - 1.0) * wb_t[p] * std::abs(dE_dnb[p] - sh_b)
                                        + 2.0 * wb_t[p] * wb_t[p];
                        h0_diag[n_rot + norb + p] = std::clamp(1.0 / std::max(cb, 2e-4), 1.0, 16.0);
                    } else {
                        h0_diag[n_rot + norb + p] = 1.0;
                    }
                }
            }

            return E_tot;
        };

        auto early_stop_cb = [&](int /*it*/, double f_cur, const Eigen::VectorXd& x_cur, const Eigen::VectorXd& g_cur) -> bool {
            // Skip redundant logit tail-crawling once current seed has collapsed to near-idempotency
            // inside the exact same Aufbau orbital basin already converged by an earlier seed!
            if (!(best_E < 1e17 && f_cur > best_E && (f_cur - best_E) < 0.08)) {
                return false;
            }
            if (n_rot > 0 && g_cur.head(n_rot).lpNorm<Eigen::Infinity>() > 2e-3) {
                return false;
            }
            const double best_idemp = std::max(
                (best_na.array() * (1.0 - best_na.array())).maxCoeff(),
                (best_nb.array() * (1.0 - best_nb.array())).maxCoeff()
            );
            if (best_idemp > 1e-6) {
                return false;
            }
            if (is_singlet) {
                auto [na_c, wt_c, wsa_c] = map_fermi_occupations(x_cur.tail(norb), static_cast<double>(n_alpha));
                (void)wsa_c;
                if (wt_c.maxCoeff() > 6e-3) return false;
                for (int p = 0; p < norb; ++p) {
                    const double gt = g_cur[n_rot + p];
                    if (na_c[p] > 0.5 && gt < -1e-9) return false;
                    if (na_c[p] <= 0.5 && gt > 1e-9) return false;
                }
                return true;
            } else {
                auto [na_c, wta_c, wsa_c] = map_fermi_occupations(x_cur.segment(n_rot, norb), static_cast<double>(n_alpha));
                auto [nb_c, wtb_c, wsb_c] = map_fermi_occupations(x_cur.segment(n_rot + norb, norb), static_cast<double>(n_beta));
                (void)wsa_c; (void)wsb_c;
                if (wta_c.maxCoeff() > 6e-3 || wtb_c.maxCoeff() > 6e-3) return false;
                for (int p = 0; p < norb; ++p) {
                    const double gta = g_cur[n_rot + p];
                    const double gtb = g_cur[n_rot + norb + p];
                    if (na_c[p] > 0.5 && gta < -1e-9) return false;
                    if (na_c[p] <= 0.5 && gta > 1e-9) return false;
                    if (nb_c[p] > 0.5 && gtb < -1e-9) return false;
                    if (nb_c[p] <= 0.5 && gtb > 1e-9) return false;
                }
                return true;
            }
        };

        auto opt_res = minimize_preconditioned_lbfgs(
            eval_fg,
            early_stop_cb,
            seed.x_init,
            seed.max_it,
            cfg.lbfgs_m,
            cfg.gtol,
            cfg.ftol,
            cfg.use_preconditioner
        );
        total_iters += opt_res.iters;
        total_fevals += opt_res.fevals;

        if (opt_res.f_opt < best_E) {
            best_E = opt_res.f_opt;
            Krot.setZero();
            int idx = 0;
            for (int p = 0; p < norb; ++p) {
                for (int q = p + 1; q < norb; ++q) {
                    const double v = opt_res.x_opt[idx++];
                    Krot(p, q) = v;
                    Krot(q, p) = -v;
                }
            }
            skew_ws.compute_expm(Krot, U);
            blas_dgemm_rm(false, false, nao, norb, norb, 1.0, seed.C_base.data(), norb, U.data(), norb, 0.0, best_C.data(), norb);

            if (is_singlet) {
                std::tie(best_na, std::ignore, std::ignore) = map_fermi_occupations(opt_res.x_opt.tail(norb), static_cast<double>(n_alpha));
                best_nb = best_na;
            } else {
                std::tie(best_na, std::ignore, std::ignore) = map_fermi_occupations(opt_res.x_opt.segment(n_rot, norb), static_cast<double>(n_alpha));
                std::tie(best_nb, std::ignore, std::ignore) = map_fermi_occupations(opt_res.x_opt.segment(n_rot + norb, norb), static_cast<double>(n_beta));
            }
        }
    }

    // Sort by descending total occupation and semi-canonicalize degenerate subspaces (|dn| < 1e-4)
    std::vector<int> order(norb);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const double ka = -(best_na[a] + best_nb[a] + 0.01 * (best_na[a] - best_nb[a]));
        const double kb = -(best_na[b] + best_nb[b] + 0.01 * (best_na[b] - best_nb[b]));
        return ka < kb;
    });

    RowMatrixXd C_opt(nao, norb);
    Eigen::VectorXd na(norb);
    Eigen::VectorXd nb(norb);
    for (int p = 0; p < norb; ++p) {
        C_opt.col(p) = best_C.col(order[p]);
        na[p] = best_na[order[p]];
        nb[p] = best_nb[order[p]];
    }
    const Eigen::VectorXd ntot = na + nb;

    blas_dgemm_rm(true, false, norb, nk, nao, 1.0, C_opt.data(), norb, X_ao.data(), nk, 0.0, X_mo.data(), nk);
    blas_dgemm_rm(false, false, nao, norb, nao, 1.0, h1_ao.data(), nao, C_opt.data(), norb, 0.0, tmp_h.data(), norb);
    blas_dgemm_rm(true, false, norb, norb, nao, 1.0, C_opt.data(), norb, tmp_h.data(), norb, 0.0, h_mo.data(), norb);

    Q_mo = X_mo.cwiseProduct(X_mo);
    d_K = Q_mo.transpose() * ntot;
    v_K = M_thc * d_K;
    for (int p = 0; p < norb; ++p) {
        J_tmp.row(p) = X_mo.row(p).cwiseProduct(v_K.transpose());
    }
    blas_dgemm_rm(false, true, norb, norb, nk, 1.0, J_tmp.data(), nk, X_mo.data(), nk, 0.0, J_mo.data(), norb);

    build_thc_k_noalloc(na, Ka);
    if (is_singlet) {
        Kb = Ka;
    } else {
        build_thc_k_noalloc(nb, Kb);
    }
    const RowMatrixXd F_semi = h_mo + J_mo - 0.5 * (Ka + Kb);

    int p_start = 0;
    while (p_start < norb) {
        int p_end = p_start + 1;
        while (p_end < norb
               && std::abs(na[p_end] - na[p_start]) < 1e-4
               && std::abs(nb[p_end] - nb[p_start]) < 1e-4) {
            ++p_end;
        }
        const int sub_len = p_end - p_start;
        const bool is_idemp = (4.0 * na[p_start] * (1.0 - na[p_start]) < 1e-6)
                           && (4.0 * nb[p_start] * (1.0 - nb[p_start]) < 1e-6);
        if (sub_len > 1 && is_idemp) {
            ColMatrixXd F_sub = 0.5 * (F_semi.block(p_start, p_start, sub_len, sub_len) +
                                       F_semi.block(p_start, p_start, sub_len, sub_len).transpose());
            Eigen::VectorXd ev_sub(sub_len);
            oblas_dsyevd('V', 'U', sub_len, F_sub.data(), sub_len, ev_sub.data());
            RowMatrixXd C_sub = C_opt.middleCols(p_start, sub_len) * F_sub;
            C_opt.middleCols(p_start, sub_len) = C_sub;
        }
        p_start = p_end;
    }

    StaticSCFOutput out;
    out.e_stat = best_E;
    out.C_no = std::move(C_opt);
    out.n_alpha = std::move(na);
    out.n_beta = std::move(nb);
    out.e_hf_init = e_hf_init;
    out.total_iters = total_iters;
    out.total_fevals = total_fevals;
    return out;
}

} // namespace pgft
