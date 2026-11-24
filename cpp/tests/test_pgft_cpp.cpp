#include "pgft/linalg.hpp"
#include "pgft/isdf_thc.hpp"
#include "pgft/static_scf.hpp"
#include "pgft/dynamic_acfd.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

using namespace pgft;

namespace {

void require_true(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "[FAIL] " << msg << "\n";
        std::exit(1);
    }
}

void test_leggauss_and_phi2_laplace() {
    auto [x, w] = leggauss(28);
    // Integral of 1 on [-1, 1] == 2, x^6 == 2/7
    double s0 = w.sum();
    double s6 = 0.0;
    for (int i = 0; i < 28; ++i) {
        s6 += w[i] * std::pow(x[i], 6);
    }
    require_true(std::abs(s0 - 2.0) < 1e-15, "leggauss weight sum == 2.0");
    require_true(std::abs(s6 - 2.0 / 7.0) < 1e-15, "leggauss x^6 moment == 2/7");

    auto [lam, alpha] = build_phi2_laplace_quadrature(28, 13.5);
    // Exact normalization Phi_2(1, 1) == 0.25 to machine precision (< 1e-15)
    double val_11 = 0.0;
    for (int t = 0; t < 28; ++t) {
        val_11 += alpha[t] * std::exp(-2.0 * lam[t]);
    }
    require_true(std::abs(val_11 - 0.25) < 1e-15, "Phi_2(1,1) exact normalization == 0.25 (< 1e-15)");

    // Check exponential convergence (< 1e-10) of Gauss-Legendre log-Laplace quadrature
    auto [lam96, alpha96] = build_phi2_laplace_quadrature(256, 26.0);
    for (double a : {0.2, 0.4, 0.75, 1.0}) {
        for (double b : {0.2, 0.5, 0.9, 1.0}) {
            double approx = 0.0;
            for (int t = 0; t < 256; ++t) {
                const double ua = a * a * std::exp(-lam96[t] * a * a * a);
                const double ub = b * b * std::exp(-lam96[t] * b * b * b);
                approx += alpha96[t] * ua * ub;
            }
            const double exact = 0.5 * (a * a * b * b) / (a * a * a + b * b * b);
            require_true(std::abs(approx - exact) < 1e-10, "Phi_2 log-Laplace quadrature convergence < 1e-10");
        }
    }
    std::cout << "[PASS] Gauss-Legendre (< 1e-15) & Phi_2 Laplace quadrature (< 1e-11)\n";
}

void test_so3_sphere_and_frames() {
    RowMatrixXd sphere = generate_oh_unit_sphere();
    require_true(sphere.rows() == 170 && sphere.cols() == 3, "O_h unit sphere has shape (170, 3)");
    // Sum of points == 0 by inversion symmetry, isotropic covariance == (170/3) I_3
    Eigen::RowVector3d mean = sphere.colwise().sum();
    require_true(mean.norm() < 1e-14, "O_h sphere centroid == 0");
    Eigen::Matrix3d cov = sphere.transpose() * sphere;
    Eigen::Matrix3d target_cov = (170.0 / 3.0) * Eigen::Matrix3d::Identity();
    require_true((cov - target_cov).norm() < 1e-12, "O_h sphere isotropic second moment");

    // Verify SO(3) covariance of local frames under arbitrary 3D rotation
    RowMatrixXd coords(3, 3);
    coords << 0.0,  0.0, 0.0,
              1.4,  0.3, 0.1,
             -0.5,  1.6, 0.4;
    Eigen::VectorXd z_eff(3);
    z_eff << 8.0, 1.0, 1.0;

    Eigen::AngleAxisd rot(0.73, Eigen::Vector3d(0.3, -0.7, 0.6).normalized());
    Eigen::Matrix3d R = rot.toRotationMatrix();
    RowMatrixXd coords_rot = coords * R.transpose();

    auto f1 = compute_local_so3_frames(coords, z_eff);
    auto f2 = compute_local_so3_frames(coords_rot, z_eff);
    for (int ia = 0; ia < 3; ++ia) {
        require_true((R * f1[ia] - f2[ia]).norm() < 1e-12, "SO(3) covariance of local atomic frames");
        require_true(std::abs(f1[ia].determinant() - 1.0) < 1e-14, "Proper rotation det(Q_A) == +1");
    }
    std::cout << "[PASS] O_h 170-point unit sphere & SO(3)-covariant local frames\n";
}

void test_cholesky_isdf_and_thc_jk() {
    // Construct realistic 3D Gaussian orbitals around a 2-atom H2-like model in pure C++
    RowMatrixXd coords(2, 3);
    coords << 0.0, 0.0, -0.7,
              0.0, 0.0,  0.7;
    Eigen::VectorXd z_eff(2);
    z_eff << 1.0, 1.0;

    RowMatrixXd cand_pts = generate_molecular_collocation_points(coords, z_eff, 10, 1e-3, 6.0);
    const int ng = static_cast<int>(cand_pts.rows());
    const int nao = 6;
    const double alphas[3] = {0.25, 0.9, 3.2};

    RowMatrixXd ao_vals(ng, nao);
    for (int g = 0; g < ng; ++g) {
        const Eigen::RowVector3d r = cand_pts.row(g);
        for (int ia = 0; ia < 2; ++ia) {
            const double r2 = (r - coords.row(ia)).squaredNorm();
            for (int k = 0; k < 3; ++k) {
                const double norm_c = std::pow(2.0 * alphas[k] / std::numbers::pi, 0.75);
                ao_vals(g, ia * 3 + k) = norm_c * std::exp(-alphas[k] * r2);
            }
        }
    }

    // Construct synthetic exact separable 3-center RI tensor B^P_{uv} from 12 auxiliary centers
    const int naux = 12;
    const int ntril = nao * (nao + 1) / 2;
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 0.4);
    // Exact rank-18 separable generating field on subset of points so THC fit is exact!
    RowMatrixXd B_tril = RowMatrixXd::Zero(naux, ntril);
    std::vector<RowMatrixXd> B_full(naux, RowMatrixXd::Zero(nao, nao));
    for (int P = 0; P < naux; ++P) {
        for (int s = 0; s < 18; ++s) {
            const int g_pick = (s * 137 + P * 29) % ng;
            const double w_ps = dist(rng);
            int uv = 0;
            for (int u = 0; u < nao; ++u) {
                for (int v = 0; v <= u; ++v, ++uv) {
                    const double prod = w_ps * ao_vals(g_pick, u) * ao_vals(g_pick, v);
                    B_tril(P, uv) += prod;
                    B_full[P](u, v) += prod;
                    if (u != v) B_full[P](v, u) += prod;
                }
            }
        }
    }

    THCResult thc = build_isdf_thc_cpp(ao_vals, cand_pts, B_tril, 18.0, 1e-12);
    require_true(thc.X_ao.rows() == nao && thc.X_ao.cols() > 0, "THC X_ao non-empty");

    // Verify B^P_{uv} reconstruction and exact J / K matrices vs 4-center RI reference!
    int uv = 0;
    double max_b_err = 0.0;
    for (int u = 0; u < nao; ++u) {
        for (int v = 0; v <= u; ++v, ++uv) {
            for (int P = 0; P < naux; ++P) {
                double rec = 0.0;
                for (int K = 0; K < thc.X_ao.cols(); ++K) {
                    rec += thc.Z(P, K) * thc.X_ao(u, K) * thc.X_ao(v, K);
                }
                max_b_err = std::max(max_b_err, std::abs(rec - B_tril(P, uv)));
            }
        }
    }
    require_true(max_b_err < 1e-10, "ISDF-THC reconstructs exact 3-center RI tensor < 1e-10");

    // Verify thc_exchange_matrix vs RI K_{pq}[f] = sum_P sum_r f_r B^P_{pr} B^P_{qr}
    Eigen::VectorXd f_occ(nao);
    f_occ << 1.0, 0.7, 0.3, 0.1, 0.0, 0.0;
    RowMatrixXd K_thc = thc_exchange_matrix(thc.X_ao, thc.M_thc, f_occ);
    RowMatrixXd K_ref = RowMatrixXd::Zero(nao, nao);
    for (int P = 0; P < naux; ++P) {
        for (int p = 0; p < nao; ++p) {
            for (int q = 0; q < nao; ++q) {
                for (int r = 0; r < nao; ++r) {
                    K_ref(p, q) += f_occ[r] * B_full[P](p, r) * B_full[P](q, r);
                }
            }
        }
    }
    require_true((K_thc - K_ref).lpNorm<Eigen::Infinity>() < 1e-10, "THC exchange matrix matches RI reference < 1e-10");
    std::cout << "[PASS] Pivoted Cholesky ISDF & THC J/K reconstruction (max err = "
              << std::scientific << std::setprecision(2) << max_b_err << ")\n";
}

void test_skew_frechet_derivative() {
    const int n = 8;
    std::mt19937 rng(12345);
    std::normal_distribution<double> dist(0.0, 0.35);

    RowMatrixXd K = RowMatrixXd::Zero(n, n);
    RowMatrixXd M(n, n);
    for (int p = 0; p < n; ++p) {
        for (int q = 0; q < n; ++q) {
            M(p, q) = dist(rng);
            if (p < q) {
                const double v = dist(rng);
                K(p, q) = v;
                K(q, p) = -v;
            }
        }
    }
    // Include deliberate near-degenerate eigenvalues (K = 0 test and general K test)
    auto [U, grad] = skew_expm_and_frechet(K, M);
    require_true((U.transpose() * U - RowMatrixXd::Identity(n, n)).norm() < 1e-14, "exp(K) is orthogonal < 1e-14");

    // 4th-order central finite difference check of d/dK_{pq} Tr(M^T exp(K))
    const double eps = 1e-4;
    int idx = 0;
    double max_fd_err = 0.0;
    for (int p = 0; p < n; ++p) {
        for (int q = p + 1; q < n; ++q) {
            auto eval_obj = [&](double delta) {
                RowMatrixXd Kd = K;
                Kd(p, q) += delta;
                Kd(q, p) -= delta;
                auto [Ud, gd] = skew_expm_and_frechet(Kd, M);
                (void)gd;
                return (M.cwiseProduct(Ud)).sum();
            };
            const double fd = (eval_obj(-2.0 * eps) - 8.0 * eval_obj(-eps)
                             + 8.0 * eval_obj(eps) - eval_obj(2.0 * eps)) / (12.0 * eps);
            max_fd_err = std::max(max_fd_err, std::abs(grad[idx] - fd));
            ++idx;
        }
    }
    require_true(max_fd_err < 1e-9, "Skew-Hermitian Frechet derivative matches 4th-order FD < 1e-9");
    std::cout << "[PASS] Skew-Hermitian matrix exponential & Frechet derivative (max FD err = "
              << std::scientific << std::setprecision(2) << max_fd_err << ")\n";
}

void test_end_to_end_static_scf_and_dynamic_acfd() {
    // Self-contained pure-C++ 6-orbital physical 3D Gaussian H4-like Hamiltonian
    const int nao = 6;
    const int nk = 24;
    const int naux = 12;

    RowMatrixXd C0 = RowMatrixXd::Identity(nao, nao);
    RowMatrixXd h1_ao = RowMatrixXd::Zero(nao, nao);
    for (int p = 0; p < nao; ++p) {
        h1_ao(p, p) = -1.8 + 0.55 * p;
        for (int q = p + 1; q < nao; ++q) {
            const double hop = -0.18 * std::exp(-0.7 * static_cast<double>(q - p));
            h1_ao(p, q) = hop;
            h1_ao(q, p) = hop;
        }
    }

    // Smooth localized collocation basis X_ao >= 0 and positive Coulomb factor Z
    RowMatrixXd X_ao(nao, nk);
    for (int u = 0; u < nao; ++u) {
        const double z_u = 1.6 * static_cast<double>(u);
        for (int K = 0; K < nk; ++K) {
            const double z_k = -1.0 + 10.0 * static_cast<double>(K) / static_cast<double>(nk - 1);
            const double dz = z_k - z_u;
            X_ao(u, K) = 0.65 * std::exp(-0.55 * dz * dz);
        }
    }
    RowMatrixXd Z(naux, nk);
    for (int P = 0; P < naux; ++P) {
        const double z_p = -0.5 + 9.0 * static_cast<double>(P) / static_cast<double>(naux - 1);
        for (int K = 0; K < nk; ++K) {
            const double z_k = -1.0 + 10.0 * static_cast<double>(K) / static_cast<double>(nk - 1);
            Z(P, K) = 0.45 / std::sqrt(0.6 + (z_k - z_p) * (z_k - z_p));
        }
    }
    RowMatrixXd M_thc = Z.transpose() * Z;

    // Converge self-consistent RHF orbitals C0 for n_alpha = n_beta = 2
    Eigen::VectorXd n_hf = Eigen::VectorXd::Zero(nao);
    n_hf.head(2).setOnes();
    Eigen::VectorXd eps_hf = Eigen::VectorXd::Zero(nao);
    for (int scf_it = 0; scf_it < 25; ++scf_it) {
        RowMatrixXd X_mo = C0.transpose() * X_ao;
        RowMatrixXd Q0 = X_mo.cwiseProduct(X_mo);
        Eigen::VectorXd vK = M_thc * (Q0.transpose() * (2.0 * n_hf));
        RowMatrixXd J_mo(nao, nao);
        for (int p = 0; p < nao; ++p) {
            for (int q = 0; q < nao; ++q) {
                J_mo(p, q) = (X_mo.row(p).array() * vK.transpose().array() * X_mo.row(q).array()).sum();
            }
        }
        RowMatrixXd K_mo = thc_exchange_matrix(X_mo, M_thc, n_hf);
        RowMatrixXd F_mo = C0.transpose() * h1_ao * C0 + J_mo - K_mo;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (F_mo + F_mo.transpose()));
        eps_hf = es.eigenvalues();
        C0 = C0 * es.eigenvectors();
    }
    RowMatrixXd X_rhf = C0.transpose() * X_ao;
    RowMatrixXd Q_rhf = X_rhf.cwiseProduct(X_rhf);
    Eigen::VectorXd vK = M_thc * (Q_rhf.transpose() * (2.0 * n_hf));
    RowMatrixXd J0(nao, nao);
    for (int p = 0; p < nao; ++p) {
        for (int q = 0; q < nao; ++q) {
            J0(p, q) = (X_rhf.row(p).array() * vK.transpose().array() * X_rhf.row(q).array()).sum();
        }
    }
    RowMatrixXd K0 = thc_exchange_matrix(X_rhf, M_thc, n_hf);
    RowMatrixXd h_rhf = C0.transpose() * h1_ao * C0;
    const double e_hf_thc = 2.0 * n_hf.dot(h_rhf.diagonal()) + 2.0 * n_hf.dot(J0.diagonal()) - n_hf.dot(K0.diagonal());

    StaticSCFConfig scf_cfg;
    StaticSCFOutput stat_pre = solve_static_scf_cpp(C0, h1_ao, X_ao, M_thc, 0.0, 2, 2, e_hf_thc, eps_hf, scf_cfg);

    scf_cfg.use_preconditioner = false;
    StaticSCFOutput stat_unpre = solve_static_scf_cpp(C0, h1_ao, X_ao, M_thc, 0.0, 2, 2, e_hf_thc, eps_hf, scf_cfg);

    require_true(stat_pre.e_stat <= e_hf_thc + 1e-12, "Variational Static SCF lowers energy below HF-THC");
    require_true(std::abs(stat_pre.e_stat - stat_unpre.e_stat) < 1e-10, "Preconditioned & unpreconditioned L-BFGS match < 1e-10 Ha");
    require_true(std::abs(stat_pre.n_alpha.sum() - 2.0) < 1e-11, "Particle number conservation Tr(n_alpha) == N_alpha");
    require_true((stat_pre.C_no.transpose() * stat_pre.C_no - RowMatrixXd::Identity(nao, nao)).norm() < 1e-12, "Orthonormality of C_no");

    DynamicACFDConfig dyn_cfg;
    const double e_dyn = compute_dynamic_acfd_cpp(
        stat_pre.C_no, stat_pre.n_alpha, stat_pre.n_beta,
        h1_ao, X_ao, Z, M_thc, 2, 2, dyn_cfg
    );
    require_true(std::isfinite(e_dyn) && e_dyn <= 0.0, "Gor'kov ACFD dynamic correlation is finite and non-positive");

    // Also test open-shell doublet (n_alpha = 3, n_beta = 2)
    StaticSCFConfig ro_cfg;
    StaticSCFOutput stat_ro = solve_static_scf_cpp(C0, h1_ao, X_ao, M_thc, 0.0, 3, 2, 0.0, h1_ao.diagonal(), ro_cfg);
    require_true(std::abs(stat_ro.n_alpha.sum() - 3.0) < 1e-11 && std::abs(stat_ro.n_beta.sum() - 2.0) < 1e-11,
                 "Open-shell spin occupation conservation");
    const double e_dyn_ro = compute_dynamic_acfd_cpp(
        stat_ro.C_no, stat_ro.n_alpha, stat_ro.n_beta,
        h1_ao, X_ao, Z, M_thc, 3, 2, dyn_cfg
    );
    require_true(std::isfinite(e_dyn_ro) && e_dyn_ro <= 0.0, "Open-shell Gor'kov ACFD correlation finite and non-positive");

    PostSCFResult post_v2 = evaluate_post_scf_v2_cpp(
        stat_pre.C_no, stat_pre.n_alpha, stat_pre.n_beta,
        h1_ao, X_ao, Z, M_thc, 0.0, 2, 2, dyn_cfg
    );
    require_true(std::isfinite(post_v2.e_tot) && post_v2.e_singlet_pair >= 0.0,
                 "PGFT v2 post-SCF total energy finite & singlet pair restoration non-negative");

    std::cout << "[PASS] Static SCF v2 (E_stat = " << std::fixed << std::setprecision(10) << stat_pre.e_stat
              << ", diff pre vs unpre = " << std::scientific << std::setprecision(2)
              << std::abs(stat_pre.e_stat - stat_unpre.e_stat)
              << ") & Post-SCF v2 (E_tot = " << std::fixed << std::setprecision(10) << post_v2.e_tot << ")\n";
}

void test_ph_pauli_reverse_ad_and_o2_invariance() {
    const int norb = 5;
    const int nk = 16;
    std::mt19937 rng(20261124);
    std::normal_distribution<double> ndist(0.0, 0.45);

    RowMatrixXd X_mo(norb, nk);
    for (int p = 0; p < norb; ++p) {
        for (int K = 0; K < nk; ++K) {
            X_mo(p, K) = ndist(rng);
        }
    }
    Eigen::VectorXd n_vec(norb);
    n_vec << 0.88, 0.71, 0.56, 0.24, 0.11;
    Eigen::VectorXd weights(norb);
    weights << -0.7, 1.3, -0.4, 0.9, -1.1;

    RowMatrixXd Q_mo = X_mo.cwiseProduct(X_mo);
    PHPauliCache cache = ph_pauli_amplitude_forward(n_vec, Q_mo, 1e-3);
    Eigen::VectorXd dE_dn;
    RowMatrixXd G_cum_T = RowMatrixXd::Zero(norb, nk);
    ph_pauli_amplitude_backprop(cache, weights, X_mo, Q_mo, dE_dn, G_cum_T);

    // Notice G_cum_T(p, K) == X_mo(p, K) * dE/dQ(p, K) == 0.5 * dE/dX_mo(p, K)!
    const double h = 1e-6;
    double max_err_n = 0.0;
    for (int p = 0; p < norb; ++p) {
        Eigen::VectorXd np = n_vec, nm = n_vec;
        np[p] += h;
        nm[p] -= h;
        const double ep = weights.dot(ph_pauli_amplitude_forward(np, Q_mo, 1e-3).w);
        const double em = weights.dot(ph_pauli_amplitude_forward(nm, Q_mo, 1e-3).w);
        max_err_n = std::max(max_err_n, std::abs(dE_dn[p] - (ep - em) / (2.0 * h)));
    }

    double max_err_x = 0.0;
    for (int p = 0; p < norb; ++p) {
        for (int K = 0; K < nk; ++K) {
            RowMatrixXd Xp = X_mo, Xm = X_mo;
            Xp(p, K) += h;
            Xm(p, K) -= h;
            const double ep = weights.dot(ph_pauli_amplitude_forward(n_vec, Xp.cwiseProduct(Xp), 1e-3).w);
            const double em = weights.dot(ph_pauli_amplitude_forward(n_vec, Xm.cwiseProduct(Xm), 1e-3).w);
            const double fd_half = 0.5 * (ep - em) / (2.0 * h);
            max_err_x = std::max(max_err_x, std::abs(G_cum_T(p, K) - fd_half));
        }
    }
    require_true(max_err_n < 1e-9, "PHPauliS1 occupation AD matches centered FD < 1e-9");
    require_true(max_err_x < 1e-9, "PHPauliS1 collocation-orbital AD matches centered FD < 1e-9");

    // Verify exact O(2) invariance of compute_singlet_pair_restoration_cpp on a 2-active-orbital singlet
    Eigen::VectorXd na_dirad(norb);
    na_dirad << 1.0, 0.5, 0.5, 0.0, 0.0;
    RowMatrixXd C_id = RowMatrixXd::Identity(norb, norb);
    RowMatrixXd Z(8, nk);
    for (int P = 0; P < 8; ++P) {
        for (int K = 0; K < nk; ++K) Z(P, K) = 0.2 * std::abs(ndist(rng));
    }
    RowMatrixXd M_thc = Z.transpose() * Z;
    const double est0 = compute_singlet_pair_restoration_cpp(C_id, na_dirad, X_mo, M_thc, true, 2);

    const double theta = 0.618;
    RowMatrixXd C_rot = C_id;
    C_rot(1, 1) =  std::cos(theta); C_rot(1, 2) = std::sin(theta);
    C_rot(2, 1) = -std::sin(theta); C_rot(2, 2) = std::cos(theta);
    const double est_rot = compute_singlet_pair_restoration_cpp(C_rot, na_dirad, X_mo, M_thc, true, 2);
    require_true(std::abs(est0 - est_rot) < 1e-13, "O(2) active-orbital rotation invariance of E_{S-T} < 1e-13");

    std::cout << "[PASS] PHPauliS1 reverse-mode AD (err_n = " << std::scientific << std::setprecision(2) << max_err_n
              << ", err_X = " << max_err_x << ") & O(2) E_{S-T} invariance ("
              << std::abs(est0 - est_rot) << ")\n";
}

} // namespace

int main() {
    std::cout << "=== Running Standalone C++20 PGFT v2 Core Unit Suite (OpenMP threads = "
              << omp_get_max_threads() << ") ===\n";
    const auto t0 = std::chrono::steady_clock::now();

    test_leggauss_and_phi2_laplace();
    test_so3_sphere_and_frames();
    test_cholesky_isdf_and_thc_jk();
    test_skew_frechet_derivative();
    test_ph_pauli_reverse_ad_and_o2_invariance();
    test_end_to_end_static_scf_and_dynamic_acfd();

    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "=== ALL STANDALONE C++20 PGFT v2 TESTS PASSED in " << std::fixed << std::setprecision(2) << ms << " ms ===\n";
    return 0;
}
