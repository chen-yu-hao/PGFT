#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cblas.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

// Direct Fortran LAPACK symbols exported natively by AVX2 libopenblas
extern "C" {
void dsyevd_(
    const char* jobz, const char* uplo, const int* n,
    double* a, const int* lda, double* w,
    double* work, const int* lwork, int* iwork, const int* liwork, int* info
);
void zheevd_(
    const char* jobz, const char* uplo, const int* n,
    std::complex<double>* a, const int* lda, double* w,
    std::complex<double>* work, const int* lwork,
    double* rwork, const int* lrwork,
    int* iwork, const int* liwork, int* info
);
void dgesv_(
    const int* n, const int* nrhs,
    double* a, const int* lda, int* ipiv,
    double* b, const int* ldb, int* info
);
void dgeqp3_(
    const int* m, const int* n,
    double* a, const int* lda, int* jpvt, double* tau,
    double* work, const int* lwork, int* info
);
void dgelsd_(
    const int* m, const int* n, const int* nrhs,
    double* a, const int* lda, double* b, const int* ldb,
    double* s, const double* rcond, int* rank,
    double* work, const int* lwork, int* iwork, int* info
);
}

namespace pgft {

// Row-major matrix matches C-contiguous NumPy arrays with zero copy
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ColMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using RowMatrixXcd = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ColMatrixXcd = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

/**
 * RAII guard that disables internal OpenBLAS & Eigen multi-threading
 * prior to entering an outer OpenMP parallel region, and restores the
 * previous thread count upon scope exit. Prevents nested-thread oversubscription.
 */
class ScopedSingleThreadBLAS {
public:
    ScopedSingleThreadBLAS() noexcept
        : prev_openblas_(openblas_get_num_threads()),
          prev_eigen_(Eigen::nbThreads()) {
        openblas_set_num_threads(1);
        Eigen::setNbThreads(1);
    }

    ~ScopedSingleThreadBLAS() noexcept {
        openblas_set_num_threads(prev_openblas_);
        Eigen::setNbThreads(prev_eigen_);
    }

    ScopedSingleThreadBLAS(const ScopedSingleThreadBLAS&) = delete;
    ScopedSingleThreadBLAS& operator=(const ScopedSingleThreadBLAS&) = delete;

private:
    int prev_openblas_;
    int prev_eigen_;
};

// Direct OpenBLAS-backed ColMajor LAPACK wrappers (never touches Netlib reference LAPACK!)
inline int oblas_dsyevd(char jobz, char uplo, int n, double* a, int lda, double* w) {
    if (n <= 0) return 0;
    int lwork = -1, liwork = -1, info = 0;
    double wk_opt = 0.0;
    int iwk_opt = 0;
    dsyevd_(&jobz, &uplo, &n, a, &lda, w, &wk_opt, &lwork, &iwk_opt, &liwork, &info);
    lwork = std::max(1, static_cast<int>(wk_opt));
    liwork = std::max(1, iwk_opt);
    std::vector<double> work(lwork);
    std::vector<int> iwork(liwork);
    dsyevd_(&jobz, &uplo, &n, a, &lda, w, work.data(), &lwork, iwork.data(), &liwork, &info);
    return info;
}

inline int oblas_zheevd(char jobz, char uplo, int n, std::complex<double>* a, int lda, double* w) {
    if (n <= 0) return 0;
    int lwork = -1, lrwork = -1, liwork = -1, info = 0;
    std::complex<double> wk_opt(0.0, 0.0);
    double rwk_opt = 0.0;
    int iwk_opt = 0;
    zheevd_(&jobz, &uplo, &n, a, &lda, w, &wk_opt, &lwork, &rwk_opt, &lrwork, &iwk_opt, &liwork, &info);
    lwork = std::max(1, static_cast<int>(wk_opt.real()));
    lrwork = std::max(1, static_cast<int>(rwk_opt));
    liwork = std::max(1, iwk_opt);
    std::vector<std::complex<double>> work(lwork);
    std::vector<double> rwork(lrwork);
    std::vector<int> iwork(liwork);
    zheevd_(&jobz, &uplo, &n, a, &lda, w, work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, &info);
    return info;
}

inline int oblas_dgesv(int n, int nrhs, double* a, int lda, int* ipiv, double* b, int ldb) {
    int info = 0;
    dgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}

inline int oblas_dgeqp3(int m, int n, double* a, int lda, int* jpvt, double* tau) {
    int lwork = -1, info = 0;
    double wk_opt = 0.0;
    dgeqp3_(&m, &n, a, &lda, jpvt, tau, &wk_opt, &lwork, &info);
    lwork = std::max(1, static_cast<int>(wk_opt));
    std::vector<double> work(lwork);
    dgeqp3_(&m, &n, a, &lda, jpvt, tau, work.data(), &lwork, &info);
    return info;
}

inline int oblas_dgelsd(int m, int n, int nrhs, double* a, int lda, double* b, int ldb, double* s, double rcond, int* rank) {
    int lwork = -1, info = 0;
    double wk_opt = 0.0;
    int iwk_opt = 0;
    dgelsd_(&m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, rank, &wk_opt, &lwork, &iwk_opt, &info);
    lwork = std::max(1, static_cast<int>(wk_opt));
    const int min_mn = std::max(1, std::min(m, n));
    const int liwork = std::max({1, iwk_opt, 3 * min_mn * 32 + 11 * min_mn});
    std::vector<double> work(lwork);
    std::vector<int> iwork(liwork, 0);
    dgelsd_(&m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, rank, work.data(), &lwork, iwork.data(), &info);
    return info;
}

/**
 * Row-major CBLAS dgemm wrapper: C = alpha * op(A) * op(B) + beta * C
 */
inline void blas_dgemm_rm(
    bool transA,
    bool transB,
    int m,
    int n,
    int k,
    double alpha,
    const double* A,
    int lda,
    const double* B,
    int ldb,
    double beta,
    double* C,
    int ldc
) noexcept {
    if (m <= 0 || n <= 0) return;
    if (k <= 0) {
        if (beta == 0.0) {
            for (int i = 0; i < m; ++i) {
                std::fill_n(C + static_cast<std::size_t>(i) * ldc, n, 0.0);
            }
        } else if (beta != 1.0) {
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    C[static_cast<std::size_t>(i) * ldc + j] *= beta;
                }
            }
        }
        return;
    }
    cblas_dgemm(
        CblasRowMajor,
        transA ? CblasTrans : CblasNoTrans,
        transB ? CblasTrans : CblasNoTrans,
        m, n, k,
        alpha, A, lda, B, ldb,
        beta, C, ldc
    );
}

/**
 * Row-major symmetric rank-k update: C = alpha * A^T * A + beta * C (fills both triangles)
 */
inline void blas_dsyrk_at_a_rm(
    int n,
    int k,
    double alpha,
    const double* A,
    int lda,
    double beta,
    double* C,
    int ldc
) noexcept {
    if (n <= 0) return;
    if (k <= 0) {
        if (beta == 0.0) {
            for (int i = 0; i < n; ++i) {
                std::fill_n(C + static_cast<std::size_t>(i) * ldc, n, 0.0);
            }
        }
        return;
    }
    cblas_dsyrk(
        CblasRowMajor,
        CblasUpper,
        CblasTrans,
        n, k,
        alpha, A, lda,
        beta, C, ldc
    );
    for (int i = 1; i < n; ++i) {
        double* row_i = C + static_cast<std::size_t>(i) * ldc;
        for (int j = 0; j < i; ++j) {
            row_i[j] = C[static_cast<std::size_t>(j) * ldc + i];
        }
    }
}

/**
 * Row-major symmetric rank-k update: C = alpha * A * A^T + beta * C (fills both triangles)
 */
inline void blas_dsyrk_a_at_rm(
    int n,
    int k,
    double alpha,
    const double* A,
    int lda,
    double beta,
    double* C,
    int ldc
) noexcept {
    if (n <= 0) return;
    if (k <= 0) {
        if (beta == 0.0) {
            for (int i = 0; i < n; ++i) {
                std::fill_n(C + static_cast<std::size_t>(i) * ldc, n, 0.0);
            }
        }
        return;
    }
    cblas_dsyrk(
        CblasRowMajor,
        CblasUpper,
        CblasNoTrans,
        n, k,
        alpha, A, lda,
        beta, C, ldc
    );
    for (int i = 1; i < n; ++i) {
        double* row_i = C + static_cast<std::size_t>(i) * ldc;
        for (int j = 0; j < i; ++j) {
            row_i[j] = C[static_cast<std::size_t>(j) * ldc + i];
        }
    }
}

/**
 * High-precision (80-bit long double Newton-Raphson) Gauss-Legendre quadrature on [-1, 1].
 */
inline std::pair<Eigen::VectorXd, Eigen::VectorXd> leggauss(int n) {
    if (n <= 0) {
        throw std::invalid_argument("leggauss: n must be positive");
    }
    Eigen::VectorXd x(n);
    Eigen::VectorXd w(n);
    const int m = (n + 1) / 2;
    const long double pi = std::numbers::pi_v<long double>;

    for (int i = 0; i < m; ++i) {
        long double z = std::cos(pi * (static_cast<long double>(i) + 0.75L) /
                                 (static_cast<long double>(n) + 0.5L));
        long double pp = 0.0L;
        for (int iter = 0; iter < 32; ++iter) {
            long double p1 = 1.0L;
            long double p2 = 0.0L;
            for (int j = 1; j <= n; ++j) {
                const long double p3 = p2;
                p2 = p1;
                p1 = ((2.0L * j - 1.0L) * z * p2 - (j - 1.0L) * p3) / static_cast<long double>(j);
            }
            pp = static_cast<long double>(n) * (z * p1 - p2) / (z * z - 1.0L);
            const long double dz = p1 / pp;
            z -= dz;
            if (std::abs(dz) < 1e-19L) break;
        }
        const long double wi = 2.0L / ((1.0L - z * z) * pp * pp);
        x[i] = static_cast<double>(-z);
        x[n - 1 - i] = static_cast<double>(z);
        w[i] = static_cast<double>(wi);
        w[n - 1 - i] = static_cast<double>(wi);
    }
    if (n % 2 == 1) {
        x[n / 2] = 0.0;
    }
    return {x, w};
}

} // namespace pgft
