#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "pgft/isdf_thc.hpp"
#include "pgft/static_scf.hpp"
#include "pgft/dynamic_acfd.hpp"

namespace py = pybind11;
using namespace pgft;

PYBIND11_MODULE(_pgft_cpp, m) {
    m.doc() = "PGFT v2 High-Performance C++20 + OpenMP + OpenBLAS/LAPACKE/Eigen3 Core Engine";

    // Enums & Configuration Structs
    py::enum_<AmplitudeMode>(m, "AmplitudeMode")
        .value("Pade13", AmplitudeMode::Pade13)
        .value("LinearS", AmplitudeMode::LinearS)
        .value("RationalPade", AmplitudeMode::RationalPade)
        .value("PowerPade", AmplitudeMode::PowerPade)
        .value("PHPauliS1", AmplitudeMode::PHPauliS1)
        .export_values();

    py::class_<AmplitudeParams>(m, "AmplitudeParams")
        .def(py::init<>())
        .def_readwrite("mode", &AmplitudeParams::mode)
        .def_readwrite("alpha", &AmplitudeParams::alpha)
        .def_readwrite("power", &AmplitudeParams::power)
        .def_readwrite("ph_eta", &AmplitudeParams::ph_eta);

    py::class_<StaticSCFConfig>(m, "StaticSCFConfig")
        .def(py::init<>())
        .def_readwrite("n_quad_phi2", &StaticSCFConfig::n_quad_phi2)
        .def_readwrite("L_exp_phi2", &StaticSCFConfig::L_exp_phi2)
        .def_readwrite("lbfgs_m", &StaticSCFConfig::lbfgs_m)
        .def_readwrite("gtol", &StaticSCFConfig::gtol)
        .def_readwrite("ftol", &StaticSCFConfig::ftol)
        .def_readwrite("use_preconditioner", &StaticSCFConfig::use_preconditioner)
        .def_readwrite("amp_params", &StaticSCFConfig::amp_params);

    py::enum_<QuasiparticleMode>(m, "QuasiparticleMode")
        .value("MarkovLocalMu", QuasiparticleMode::MarkovLocalMu)
        .value("GlobalGapMu", QuasiparticleMode::GlobalGapMu)
        .value("ScaledPairingGap", QuasiparticleMode::ScaledPairingGap)
        .value("ROHFHalfGap", QuasiparticleMode::ROHFHalfGap)
        .export_values();

    py::enum_<SurvivalKernelMode>(m, "SurvivalKernelMode")
        .value("LinearSelfScreeningFree", SurvivalKernelMode::LinearSelfScreeningFree)
        .value("SmoothRational", SurvivalKernelMode::SmoothRational)
        .value("PowerSelfScreeningFree", SurvivalKernelMode::PowerSelfScreeningFree)
        .value("HyperbolicSelfScreeningFree", SurvivalKernelMode::HyperbolicSelfScreeningFree)
        .export_values();

    py::class_<DynamicACFDConfig>(m, "DynamicACFDConfig")
        .def(py::init<>())
        .def_readwrite("n_omega", &DynamicACFDConfig::n_omega)
        .def_readwrite("n_tau", &DynamicACFDConfig::n_tau)
        .def_readwrite("amp_params", &DynamicACFDConfig::amp_params)
        .def_readwrite("qp_mode", &DynamicACFDConfig::qp_mode)
        .def_readwrite("survival_mode", &DynamicACFDConfig::survival_mode)
        .def_readwrite("pairing_gap_scale", &DynamicACFDConfig::pairing_gap_scale)
        .def_readwrite("survival_exponent", &DynamicACFDConfig::survival_exponent)
        .def_readwrite("hyperbolic_alpha", &DynamicACFDConfig::hyperbolic_alpha)
        .def_readwrite("dual_range_dispersion", &DynamicACFDConfig::dual_range_dispersion)
        .def_readwrite("use_sub_projector", &DynamicACFDConfig::use_sub_projector)
        .def_readwrite("coulomb_reg", &DynamicACFDConfig::coulomb_reg)
        .def_readwrite("relax_active_subspace", &DynamicACFDConfig::relax_active_subspace)
        .def_readwrite("singlet_pair_order", &DynamicACFDConfig::singlet_pair_order);

    // Output Containers
    py::class_<THCResult>(m, "THCResult")
        .def_readonly("X_ao", &THCResult::X_ao)
        .def_readonly("Z", &THCResult::Z)
        .def_readonly("M_thc", &THCResult::M_thc)
        .def_readonly("pts", &THCResult::pts)
        .def_readonly("pivots", &THCResult::pivots);

    py::class_<StaticSCFOutput>(m, "StaticSCFOutput")
        .def_readonly("e_stat", &StaticSCFOutput::e_stat)
        .def_readonly("C_no", &StaticSCFOutput::C_no)
        .def_readonly("n_alpha", &StaticSCFOutput::n_alpha)
        .def_readonly("n_beta", &StaticSCFOutput::n_beta)
        .def_readonly("e_hf_init", &StaticSCFOutput::e_hf_init)
        .def_readonly("total_iters", &StaticSCFOutput::total_iters)
        .def_readonly("total_fevals", &StaticSCFOutput::total_fevals);

    py::class_<PostSCFResult>(m, "PostSCFResult")
        .def_readonly("e_stat", &PostSCFResult::e_stat)
        .def_readonly("e_stat_base", &PostSCFResult::e_stat_base)
        .def_readonly("e_singlet_pair", &PostSCFResult::e_singlet_pair)
        .def_readonly("e_dyn", &PostSCFResult::e_dyn)
        .def_readonly("e_tot", &PostSCFResult::e_tot)
        .def_readonly("n_alpha", &PostSCFResult::n_alpha)
        .def_readonly("n_beta", &PostSCFResult::n_beta);

    // Kernel 1: ISDF-THC functions
    m.def("generate_oh_unit_sphere", &generate_oh_unit_sphere,
          "Generate deterministic 170-point O_h-symmetric unit sphere.");

    m.def("generate_molecular_collocation_points", &generate_molecular_collocation_points,
          py::arg("coords_bohr"),
          py::arg("z_eff"),
          py::arg("n_rad") = 28,
          py::arg("r_min") = 3e-4,
          py::arg("r_max") = 9.0,
          py::call_guard<py::gil_scoped_release>(),
          "Generate deterministic SO(3)-covariant candidate collocation points (natm * n_rad * 170, 3).");

    m.def("pivoted_cholesky_isdf", &pivoted_cholesky_isdf,
          py::arg("Phi_unit"),
          py::arg("col_norm"),
          py::arg("max_rank"),
          py::arg("tol") = 1e-11,
          py::call_guard<py::gil_scoped_release>(),
          "Fast O(N^3) lazy Pivoted Cholesky ISDF pivot selection.");

    m.def("fit_thc_z", &fit_thc_z,
          py::arg("X_ao"),
          py::arg("B_tril"),
          py::arg("schwarz_tol") = 1e-18,
          py::arg("reg") = 1e-13,
          py::call_guard<py::gil_scoped_release>(),
          "Fit auxiliary-to-THC matrix Z and kernel M_thc = Z^T Z from X_ao and packed B_tril.");

    m.def("build_isdf_thc_cpp", &build_isdf_thc_cpp,
          py::arg("ao_vals_ng_nao"),
          py::arg("coords_raw"),
          py::arg("B_tril"),
          py::arg("c_thc") = 18.0,
          py::arg("tol") = 1e-11,
          py::call_guard<py::gil_scoped_release>(),
          "Zero-copy O(N^3) ISDF-THC builder from PySCF ao_vals (ng, nao), coords (ng, 3), and B_tril.");

    // Kernel 2: Static SCF & Singlet Pair Restoration functions
    m.def("build_phi2_laplace_quadrature", &build_phi2_laplace_quadrature,
          py::arg("n_quad") = 28,
          py::arg("L_exp") = 13.5);

    m.def("pauli_amplitude_and_deriv", &pauli_amplitude_and_deriv,
          py::arg("n_vec"),
          py::arg("params") = AmplitudeParams{});

    m.def("thc_exchange_matrix", &thc_exchange_matrix,
          py::arg("X_mo"),
          py::arg("M_thc"),
          py::arg("f_vec"),
          py::call_guard<py::gil_scoped_release>());

    m.def("skew_expm_and_frechet", &skew_expm_and_frechet,
          py::arg("K"),
          py::arg("M"),
          py::call_guard<py::gil_scoped_release>());

    m.def("compute_singlet_pair_restoration_cpp", &compute_singlet_pair_restoration_cpp,
          py::arg("C_no"),
          py::arg("n_alpha_vec"),
          py::arg("X_ao"),
          py::arg("M_thc"),
          py::arg("is_singlet") = true,
          py::arg("harmonic_order") = 2,
          py::arg("amp_params") = AmplitudeParams{},
          py::call_guard<py::gil_scoped_release>(),
          "Strictly O(N^3), O(2)- & SO(3)-invariant Monogamy-Gated Singlet Pair Restoration.");

    m.def("solve_static_scf_cpp", &solve_static_scf_cpp,
          py::arg("C0"),
          py::arg("h1_ao"),
          py::arg("X_ao"),
          py::arg("M_thc"),
          py::arg("enuc"),
          py::arg("n_alpha"),
          py::arg("n_beta"),
          py::arg("e_hf_init"),
          py::arg("mo_energy") = Eigen::VectorXd(),
          py::arg("cfg") = StaticSCFConfig{},
          py::call_guard<py::gil_scoped_release>(),
          "High-performance O(N^3) C++20 + OpenMP + Fock-Preconditioned L-BFGS Static SCF solver.");

    // Kernel 3: Dynamic Gor'kov ACFD & Unified PGFT v2 Post-SCF functions
    m.def("build_laplace_cos_weights", &build_laplace_cos_weights,
          py::arg("omegas"),
          py::arg("emin"),
          py::arg("emax"),
          py::arg("n_tau") = 38,
          py::call_guard<py::gil_scoped_release>());

    m.def("compute_dynamic_acfd_cpp", &compute_dynamic_acfd_cpp,
          py::arg("C_no"),
          py::arg("n_alpha_vec"),
          py::arg("n_beta_vec"),
          py::arg("h1_ao"),
          py::arg("X_ao"),
          py::arg("Z"),
          py::arg("M_thc"),
          py::arg("n_alpha_elec"),
          py::arg("n_beta_elec"),
          py::arg("cfg") = DynamicACFDConfig{},
          py::call_guard<py::gil_scoped_release>(),
          "High-performance O(N^3) C++20 + OpenMP Gor'kov ACFD correlation solver.");

    m.def("evaluate_post_scf_v2_cpp", &evaluate_post_scf_v2_cpp,
          py::arg("C_no"),
          py::arg("n_alpha_init"),
          py::arg("n_beta_init"),
          py::arg("h1_ao"),
          py::arg("X_ao"),
          py::arg("Z"),
          py::arg("M_thc"),
          py::arg("enuc"),
          py::arg("n_alpha_elec"),
          py::arg("n_beta_elec"),
          py::arg("cfg") = DynamicACFDConfig{},
          py::call_guard<py::gil_scoped_release>(),
          "Unified PGFT v2 C++20 Post-Static Active-Subspace Variational Relaxation + ACFD + Singlet Pair Restoration.");
}
