#include "pgft/isdf_thc.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <numeric>
#include <set>
#include <tuple>

namespace pgft {

RowMatrixXd generate_oh_unit_sphere() {
    std::vector<Eigen::Vector3d> generators;
    const int m = 6;
    const double half_pi = 0.5 * std::numbers::pi;
    for (int i = m; i >= 0; --i) {
        for (int j = m - i; j >= 0; --j) {
            const int k = m - i - j;
            if (i >= j && j >= k && k >= 0) {
                Eigen::Vector3d v(
                    std::sin(half_pi * static_cast<double>(i) / static_cast<double>(m)),
                    std::sin(half_pi * static_cast<double>(j) / static_cast<double>(m)),
                    std::sin(half_pi * static_cast<double>(k) / static_cast<double>(m))
                );
                v.normalize();
                generators.push_back(v);
            }
        }
    }
    Eigen::Vector3d v_arc(5.0, 5.0, 2.0);
    v_arc.normalize();
    generators.push_back(v_arc);

    constexpr std::array<std::array<int, 3>, 6> perms = {{
        {{0, 1, 2}}, {{0, 2, 1}}, {{1, 0, 2}},
        {{1, 2, 0}}, {{2, 0, 1}}, {{2, 1, 0}}
    }};
    constexpr std::array<std::array<double, 3>, 8> signs = {{
        {{ 1.0,  1.0,  1.0}}, {{ 1.0,  1.0, -1.0}},
        {{ 1.0, -1.0,  1.0}}, {{ 1.0, -1.0, -1.0}},
        {{-1.0,  1.0,  1.0}}, {{-1.0,  1.0, -1.0}},
        {{-1.0, -1.0,  1.0}}, {{-1.0, -1.0, -1.0}}
    }};

    // Keep first occurrence in deterministic loop order after rounding to 11 decimals
    // (matches np.round(..., 11) + np.unique(..., return_index=True) + np.sort(unique_idx))
    std::set<std::tuple<long long, long long, long long>> seen;
    std::vector<Eigen::Vector3d> unique_pts;
    unique_pts.reserve(170);

    auto round11 = [](double x) -> long long {
        // Match IEEE round-to-nearest at 1e11 scale (avoid -0)
        long long r = std::llround(x * 1e11);
        return r;
    };

    for (const auto& g : generators) {
        for (const auto& p : perms) {
            const Eigen::Vector3d gp(g[p[0]], g[p[1]], g[p[2]]);
            for (const auto& s : signs) {
                Eigen::Vector3d pt(gp[0] * s[0], gp[1] * s[1], gp[2] * s[2]);
                auto key = std::make_tuple(round11(pt[0]), round11(pt[1]), round11(pt[2]));
                if (seen.insert(key).second) {
                    pt.normalize();
                    unique_pts.push_back(pt);
                }
            }
        }
    }

    RowMatrixXd sphere(static_cast<Eigen::Index>(unique_pts.size()), 3);
    for (std::size_t i = 0; i < unique_pts.size(); ++i) {
        sphere.row(static_cast<Eigen::Index>(i)) = unique_pts[i].transpose();
    }
    return sphere;
}

std::vector<Eigen::Matrix3d> compute_local_so3_frames(
    const Eigen::Ref<const RowMatrixXd>& coords_bohr,
    const Eigen::Ref<const Eigen::VectorXd>& z_eff
) {
    const int natm = static_cast<int>(coords_bohr.rows());
    std::vector<Eigen::Matrix3d> frames(natm, Eigen::Matrix3d::Identity());

    for (int ia = 0; ia < natm; ++ia) {
        Eigen::Matrix3d I_loc = Eigen::Matrix3d::Zero();
        Eigen::Vector3d u_vec = Eigen::Vector3d::Zero();

        for (int ib = 0; ib < natm; ++ib) {
            if (ib == ia) continue;
            const Eigen::Vector3d d = coords_bohr.row(ib).transpose() - coords_bohr.row(ia).transpose();
            const double r2 = d.squaredNorm();
            double w = z_eff[ib] * std::exp(-r2 / 16.0);
            if (w < 1e-30) w = 0.0;
            if (w == 0.0) continue;

            I_loc.noalias() += w * (d * d.transpose());
            const double idx_asym = 1.0 + 0.0137 * static_cast<double>(ib + 1);
            u_vec.noalias() += (w * idx_asym) * d;
        }

        if (I_loc.trace() < 1e-24) {
            frames[ia] = Eigen::Matrix3d::Identity();
            continue;
        }

        Eigen::Matrix3d Q = I_loc;
        Eigen::Vector3d evals;
        oblas_dsyevd('V', 'U', 3, Q.data(), 3, evals.data());

        const Eigen::Vector3d projs = Q.transpose() * u_vec;
        for (int m = 0; m < 3; ++m) {
            if (std::abs(projs[m]) > 1e-10 && projs[m] < 0.0) {
                Q.col(m) *= -1.0;
            }
        }

        if (Q.determinant() < 0.0) {
            const Eigen::Vector3d abs_p = (Q.transpose() * u_vec).cwiseAbs();
            int m_flip = 0;
            if (abs_p[1] < abs_p[m_flip]) m_flip = 1;
            if (abs_p[2] < abs_p[m_flip]) m_flip = 2;
            Q.col(m_flip) *= -1.0;
        }
        frames[ia] = Q;
    }
    return frames;
}

RowMatrixXd generate_molecular_collocation_points(
    const Eigen::Ref<const RowMatrixXd>& coords_bohr,
    const Eigen::Ref<const Eigen::VectorXd>& z_eff,
    int n_rad,
    double r_min,
    double r_max
) {
    const int natm = static_cast<int>(coords_bohr.rows());
    const RowMatrixXd sphere = generate_oh_unit_sphere();
    const int n_ang = static_cast<int>(sphere.rows());
    const auto frames = compute_local_so3_frames(coords_bohr, z_eff);

    Eigen::VectorXd radii(n_rad);
    const double ratio = r_max / r_min;
    for (int k = 1; k <= n_rad; ++k) {
        const double t_cheb = 0.5 * (1.0 - std::cos((2.0 * k - 1.0) * std::numbers::pi / (2.0 * n_rad)));
        radii[k - 1] = r_min * std::pow(ratio, t_cheb);
    }

    RowMatrixXd pts(static_cast<Eigen::Index>(natm) * n_rad * n_ang, 3);
    for (int ia = 0; ia < natm; ++ia) {
        const RowMatrixXd rot_sphere = sphere * frames[ia].transpose();
        const double cx = coords_bohr(ia, 0);
        const double cy = coords_bohr(ia, 1);
        const double cz = coords_bohr(ia, 2);
        const Eigen::Index atom_offset = static_cast<Eigen::Index>(ia) * n_rad * n_ang;
        for (int ir = 0; ir < n_rad; ++ir) {
            const double rad = radii[ir];
            const Eigen::Index shell_offset = atom_offset + static_cast<Eigen::Index>(ir) * n_ang;
            for (int ig = 0; ig < n_ang; ++ig) {
                const Eigen::Index row_idx = shell_offset + ig;
                pts(row_idx, 0) = cx + rad * rot_sphere(ig, 0);
                pts(row_idx, 1) = cy + rad * rot_sphere(ig, 1);
                pts(row_idx, 2) = cz + rad * rot_sphere(ig, 2);
            }
        }
    }
    return pts;
}

namespace {

// Internal column-major Pivoted Cholesky ISDF engine (each point g is a contiguous column!)
std::vector<int> pivoted_cholesky_colmajor(
    const ColMatrixXd& Phi_orig,
    const Eigen::VectorXd& col_norm,
    int max_rank,
    double tol
) {
    const int nao = static_cast<int>(Phi_orig.rows());
    const int ng = static_cast<int>(Phi_orig.cols());
    if (ng == 0 || max_rank <= 0) return {};
    max_rank = std::min(max_rank, ng);

    const double max_cn = col_norm.maxCoeff();
    Eigen::VectorXd tb_act = (1e-15 / (max_cn + 1e-30)) * col_norm;
    Eigen::VectorXd diag = Eigen::VectorXd::Ones(ng);
    std::vector<int> active_idx(ng);
    std::iota(active_idx.begin(), active_idx.end(), 0);

    ColMatrixXd Phi_act = Phi_orig; // (nao, ng_act) ColMajor: column g is contiguous!
    int ng_act = ng;

    int alloc_rows = std::min(max_rank, 256);
    // RowMajor (alloc_rows, ng_act): row j is ng_act contiguous doubles for 4-way AVX2 dgemv_n streaming!
    RowMatrixXd L_act = RowMatrixXd::Zero(alloc_rows, ng_act);

    Eigen::VectorXd col_buf(ng_act);
    Eigen::VectorXd lp_buf(max_rank);
    std::vector<int> pivots;
    pivots.reserve(max_rank);
    const int n_threads = std::max(1, omp_get_max_threads());
    ScopedSingleThreadBLAS blas_guard;

    for (int k = 0; k < max_rank; ++k) {
        int piv_local = 0;
        if (k == 0) {
            double best = diag[0] + tb_act[0];
            for (int g = 1; g < ng_act; ++g) {
                const double v = diag[g] + tb_act[g];
                if (v > best) {
                    best = v;
                    piv_local = g;
                }
            }
        } else {
            double best = diag[0];
            for (int g = 1; g < ng_act; ++g) {
                if (diag[g] > best) {
                    best = diag[g];
                    piv_local = g;
                }
            }
        }

        const double d_val = diag[piv_local];
        if (d_val < tol) {
            break;
        }

        if (k >= alloc_rows) {
            const int new_rows = std::min(max_rank, alloc_rows + 256);
            RowMatrixXd L_new = RowMatrixXd::Zero(new_rows, ng_act);
            std::memcpy(
                L_new.data(),
                L_act.data(),
                static_cast<std::size_t>(k) * static_cast<std::size_t>(ng_act) * sizeof(double)
            );
            L_act.swap(L_new);
            alloc_rows = new_rows;
        }

        const int piv_global = active_idx[piv_local];
        pivots.push_back(piv_global);

        const double* phi_p = Phi_orig.data() + static_cast<std::size_t>(piv_global) * nao;
        for (int j = 0; j < k; ++j) {
            lp_buf[j] = L_act.data()[static_cast<std::size_t>(j) * ng_act + piv_local];
        }

        const double inv_lkk = 1.0 / std::sqrt(std::max(d_val, 1e-30));
        double* L_base = L_act.data();
        double* d_ptr = diag.data();
        double* cb = col_buf.data();

        const bool use_omp_step = (n_threads > 1) &&
            (static_cast<long long>(ng_act) * static_cast<long long>(nao + k) >= 4000000LL);
        #pragma omp parallel num_threads(n_threads) if(use_omp_step)
        {
            const int tid = omp_get_thread_num();
            const int nth = omp_get_num_threads();
            const int g0 = (ng_act * tid) / nth;
            const int g1 = (ng_act * (tid + 1)) / nth;
            const int len_g = g1 - g0;
            if (len_g > 0) {
                cblas_dgemv(
                    CblasColMajor, CblasTrans,
                    nao, len_g,
                    1.0, Phi_act.data() + static_cast<std::size_t>(g0) * nao, nao,
                    phi_p, 1,
                    0.0, cb + g0, 1
                );
                for (int g = g0; g < g1; ++g) {
                    cb[g] *= cb[g];
                }
                if (k > 0) {
                    cblas_dgemv(
                        CblasRowMajor, CblasTrans,
                        k, len_g,
                        -1.0, L_base + g0, ng_act,
                        lp_buf.data(), 1,
                        1.0, cb + g0, 1
                    );
                }
                double* row_k = L_base + static_cast<std::size_t>(k) * ng_act;
                for (int g = g0; g < g1; ++g) {
                    const double rk = cb[g] * inv_lkk;
                    row_k[g] = rk;
                    const double nd = d_ptr[g] - rk * rk;
                    d_ptr[g] = (nd > 0.0) ? nd : 0.0;
                }
            }
        }
        d_ptr[piv_local] = 0.0;

        // Exact dead-column compaction (columns with diag < tol can never become pivots)
        if ((k + 1) % 32 == 0 && ng_act > 1024) {
            int alive_cnt = 0;
            for (int g = 0; g < ng_act; ++g) {
                if (d_ptr[g] >= tol) ++alive_cnt;
            }
            if (alive_cnt == 0) {
                break;
            }
            if (alive_cnt < static_cast<int>(0.80 * ng_act)) {
                std::vector<int> keep_cols;
                keep_cols.reserve(alive_cnt);
                for (int src = 0; src < ng_act; ++src) {
                    if (d_ptr[src] >= tol) {
                        const int dst = static_cast<int>(keep_cols.size());
                        keep_cols.push_back(src);
                        if (dst != src) {
                            d_ptr[dst] = d_ptr[src];
                            active_idx[dst] = active_idx[src];
                            tb_act[dst] = tb_act[src];
                            std::memcpy(
                                Phi_act.data() + static_cast<std::size_t>(dst) * nao,
                                Phi_act.data() + static_cast<std::size_t>(src) * nao,
                                static_cast<std::size_t>(nao) * sizeof(double)
                            );
                        }
                    }
                }
                RowMatrixXd L_comp(alloc_rows, alive_cnt);
                #pragma omp parallel for schedule(static) num_threads(n_threads) if(k >= 64 && alive_cnt >= 2048)
                for (int j = 0; j <= k; ++j) {
                    const double* src_r = L_base + static_cast<std::size_t>(j) * ng_act;
                    double* dst_r = L_comp.data() + static_cast<std::size_t>(j) * alive_cnt;
                    for (int dst = 0; dst < alive_cnt; ++dst) {
                        dst_r[dst] = src_r[keep_cols[dst]];
                    }
                }
                L_act.swap(L_comp);
                ng_act = alive_cnt;
            }
        }
    }

    return pivots;
}

} // namespace

std::vector<int> pivoted_cholesky_isdf(
    const Eigen::Ref<const RowMatrixXd>& Phi_unit,
    const Eigen::Ref<const Eigen::VectorXd>& col_norm,
    int max_rank,
    double tol
) {
    ColMatrixXd Phi_col = Phi_unit; // Convert once to contiguous column-per-point layout
    return pivoted_cholesky_colmajor(Phi_col, col_norm, max_rank, tol);
}

std::pair<RowMatrixXd, RowMatrixXd> fit_thc_z(
    const Eigen::Ref<const RowMatrixXd>& X_ao,
    const Eigen::Ref<const RowMatrixXd>& B_tril,
    double schwarz_tol,
    double reg
) {
    const int nao = static_cast<int>(X_ao.rows());
    const int nk = static_cast<int>(X_ao.cols());
    const int naux = static_cast<int>(B_tril.rows());
    const int ntril = static_cast<int>(B_tril.cols());
    if (ntril != nao * (nao + 1) / 2) {
        throw std::invalid_argument("fit_thc_z: B_tril cols must equal nao*(nao+1)/2");
    }

    // 1. All-pair Khatri-Rao Gram matrix AtA = (X^T X)^{\odot 2} via dsyrk
    RowMatrixXd XtX(nk, nk);
    blas_dsyrk_at_a_rm(nk, nao, 1.0, X_ao.data(), nk, 0.0, XtX.data(), nk);
    RowMatrixXd AtA = XtX.cwiseProduct(XtX);

    // 2. Compute pair norms sum_P (B^P_{uv})^2 streaming row-wise along B_tril
    Eigen::VectorXd pair_norms = Eigen::VectorXd::Zero(ntril);
    double* pn_ptr = pair_norms.data();
    for (int P = 0; P < naux; ++P) {
        const double* b_row = B_tril.data() + static_cast<std::size_t>(P) * ntril;
        for (int uv = 0; uv < ntril; ++uv) {
            const double val = b_row[uv];
            pn_ptr[uv] += val * val;
        }
    }

    struct SigPair {
        int u;
        int v;
        int uv;
        double mult;
    };
    std::vector<SigPair> sig_pairs;
    sig_pairs.reserve(ntril);

    int uv_idx = 0;
    for (int u = 0; u < nao; ++u) {
        for (int v = 0; v <= u; ++v, ++uv_idx) {
            if (pn_ptr[uv_idx] > schwarz_tol) {
                sig_pairs.push_back({u, v, uv_idx, (u == v) ? 1.0 : 2.0});
            }
        }
    }

    const int n_sig = static_cast<int>(sig_pairs.size());
    // BtA has shape (naux, nk) RowMajor == (nk, naux) ColMajor!
    RowMatrixXd Z = RowMatrixXd::Zero(naux, nk);

    if (n_sig > 0) {
        constexpr int BLK_SIG = 2048;
        RowMatrixXd A_blk( std::min(n_sig, BLK_SIG), nk );
        RowMatrixXd B_blk( naux, std::min(n_sig, BLK_SIG) );

        for (int s0 = 0; s0 < n_sig; s0 += BLK_SIG) {
            const int blk = std::min(BLK_SIG, n_sig - s0);

            #pragma omp parallel for schedule(static) if(blk >= 256)
            for (int i = 0; i < blk; ++i) {
                const auto& sp = sig_pairs[s0 + i];
                const double* xu = X_ao.data() + static_cast<std::size_t>(sp.u) * nk;
                const double* xv = X_ao.data() + static_cast<std::size_t>(sp.v) * nk;
                double* a_row = A_blk.data() + static_cast<std::size_t>(i) * nk;
                const double m = sp.mult;
                for (int K = 0; K < nk; ++K) {
                    a_row[K] = m * xu[K] * xv[K];
                }
            }

            #pragma omp parallel for schedule(static) if(naux * blk >= 16384)
            for (int P = 0; P < naux; ++P) {
                const double* b_src = B_tril.data() + static_cast<std::size_t>(P) * ntril;
                double* b_dst = B_blk.data() + static_cast<std::size_t>(P) * blk;
                for (int i = 0; i < blk; ++i) {
                    b_dst[i] = b_src[sig_pairs[s0 + i].uv];
                }
            }

            blas_dgemm_rm(
                false, false,
                naux, nk, blk,
                1.0, B_blk.data(), blk,
                A_blk.data(), nk,
                1.0, Z.data(), nk
            );
        }
    }

    // 3. Diagonal equilibration & SPD Cholesky solve in-place on Z ((naux, nk) RowMajor == (nk, naux) ColMajor)
    Eigen::VectorXd d_scale(nk);
    for (int K = 0; K < nk; ++K) {
        d_scale[K] = std::sqrt(std::max(AtA(K, K), 1e-30));
    }

    RowMatrixXd AtA_norm(nk, nk);
    for (int K = 0; K < nk; ++K) {
        const double dk = d_scale[K];
        for (int L = 0; L < nk; ++L) {
            AtA_norm(K, L) = AtA(K, L) / (dk * d_scale[L]);
        }
        AtA_norm(K, K) += reg;
    }

    for (int P = 0; P < naux; ++P) {
        double* z_row = Z.data() + static_cast<std::size_t>(P) * nk;
        for (int K = 0; K < nk; ++K) {
            z_row[K] /= d_scale[K];
        }
    }

    // Solve AtA_norm * Z^T = AtB_scaled via partially-pivoted LU (oblas_dgesv, identical to np.linalg.solve)
    std::vector<int> ipiv(nk);
    int info = oblas_dgesv(nk, naux, AtA_norm.data(), nk, ipiv.data(), Z.data(), nk);
    if (info != 0) {
        Eigen::MatrixXd rhs_t = Z.transpose();
        RowMatrixXd AtA_fresh(nk, nk);
        for (int K = 0; K < nk; ++K) {
            const double dk = d_scale[K];
            for (int L = 0; L < nk; ++L) {
                AtA_fresh(K, L) = AtA(K, L) / (dk * d_scale[L]);
            }
            AtA_fresh(K, K) += reg;
        }
        Eigen::MatrixXd sol = AtA_fresh.partialPivLu().solve(rhs_t);
        Z = sol.transpose();
    }

    for (int P = 0; P < naux; ++P) {
        double* z_row = Z.data() + static_cast<std::size_t>(P) * nk;
        for (int K = 0; K < nk; ++K) {
            z_row[K] /= d_scale[K];
        }
    }

    // 4. Symmetric effective Coulomb kernel M_thc = Z^T Z via dsyrk
    RowMatrixXd M_thc(nk, nk);
    blas_dsyrk_at_a_rm(nk, naux, 1.0, Z.data(), nk, 0.0, M_thc.data(), nk);

    return {std::move(Z), std::move(M_thc)};
}

THCResult build_isdf_thc_cpp(
    const Eigen::Ref<const RowMatrixXd>& ao_vals_ng_nao,
    const Eigen::Ref<const RowMatrixXd>& coords_raw,
    const Eigen::Ref<const RowMatrixXd>& B_tril,
    double c_thc,
    double tol
) {
    const int ng_raw = static_cast<int>(ao_vals_ng_nao.rows());
    const int nao = static_cast<int>(ao_vals_ng_nao.cols());
    const int naux = static_cast<int>(B_tril.rows());

    // Filter candidate points with col_norm > 1e-10 directly into ColMajor Phi_unit (nao, ng)
    std::vector<int> keep_idx;
    keep_idx.reserve(ng_raw);
    std::vector<double> keep_norms;
    keep_norms.reserve(ng_raw);

    for (int g = 0; g < ng_raw; ++g) {
        const double* row = ao_vals_ng_nao.data() + static_cast<std::size_t>(g) * nao;
        double s2 = 0.0;
        for (int u = 0; u < nao; ++u) {
            s2 += row[u] * row[u];
        }
        const double cn = std::sqrt(s2);
        if (cn > 1e-10) {
            keep_idx.push_back(g);
            keep_norms.push_back(cn);
        }
    }

    const int ng = static_cast<int>(keep_idx.size());
    ColMatrixXd Phi_unit(nao, ng);
    Eigen::VectorXd col_norm(ng);

    #pragma omp parallel for schedule(static) if(ng >= 1024)
    for (int j = 0; j < ng; ++j) {
        const int g = keep_idx[j];
        const double cn = keep_norms[j];
        col_norm[j] = cn;
        const double inv_cn = 1.0 / cn;
        const double* src = ao_vals_ng_nao.data() + static_cast<std::size_t>(g) * nao;
        double* dst = Phi_unit.data() + static_cast<std::size_t>(j) * nao;
        for (int u = 0; u < nao; ++u) {
            dst[u] = src[u] * inv_cn;
        }
    }

    const int max_rank = std::min(ng, std::max(static_cast<int>(std::llround(c_thc * nao)), 3 * naux));
    std::vector<int> pivots = pivoted_cholesky_colmajor(Phi_unit, col_norm, max_rank, tol);
    const int nk = static_cast<int>(pivots.size());

    RowMatrixXd X_ao(nao, nk);
    RowMatrixXd pts(nk, 3);
    for (int K = 0; K < nk; ++K) {
        const int raw_g = keep_idx[pivots[K]];
        const double* src = ao_vals_ng_nao.data() + static_cast<std::size_t>(raw_g) * nao;
        for (int u = 0; u < nao; ++u) {
            X_ao(u, K) = src[u];
        }
        pts.row(K) = coords_raw.row(raw_g);
    }

    auto [Z, M_thc] = fit_thc_z(X_ao, B_tril);

    THCResult res;
    res.X_ao = std::move(X_ao);
    res.Z = std::move(Z);
    res.M_thc = std::move(M_thc);
    res.pts = std::move(pts);
    res.pivots = std::move(pivots);
    return res;
}

} // namespace pgft
