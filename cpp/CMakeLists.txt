cmake_minimum_required(VERSION 3.20)
project(pgft_core VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find Eigen3
find_package(Eigen3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Core PATHS /usr/include/eigen3 /usr/local/include/eigen3 REQUIRED)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
endif()

# Link directly against AVX2 OpenBLAS (which exports both CBLAS and full Fortran LAPACK symbols:
# dsyevd_, zheevd_, dgelsd_, dgesv_, dgeqp3_), completely avoiding slow Netlib liblapack/libblas!
find_path(CBLAS_INCLUDE_DIR NAMES cblas.h PATHS /usr/include/x86_64-linux-gnu /usr/include REQUIRED)
find_library(OPENBLAS_LIB NAMES openblas
    PATHS /usr/lib/x86_64-linux-gnu/openblas-pthread /usr/lib/x86_64-linux-gnu /usr/local/lib
    NO_DEFAULT_PATH REQUIRED)

if(EXISTS "/usr/lib/x86_64-linux-gnu/openblas-pthread")
    set(CMAKE_BUILD_RPATH "/usr/lib/x86_64-linux-gnu/openblas-pthread")
    set(CMAKE_INSTALL_RPATH "/usr/lib/x86_64-linux-gnu/openblas-pthread")
    add_link_options("-Wl,--disable-new-dtags")
endif()

# High-performance C++20 compiler flags
set(PGFT_CXX_FLAGS -O3 -march=native -funroll-loops -Wall -Wextra -DEIGEN_NO_DEBUG)

# 1. Core C++20 Library: libpgft_core
add_library(pgft_core STATIC
    src/isdf_thc.cpp
    src/static_scf.cpp
    src/dynamic_acfd.cpp
)
target_include_directories(pgft_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${CBLAS_INCLUDE_DIR}
)
target_compile_options(pgft_core PRIVATE ${PGFT_CXX_FLAGS})
target_link_libraries(pgft_core
    PUBLIC
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
        ${OPENBLAS_LIB}
)

# 2. Standalone C++20 Unit & Benchmark Test Binary: test_pgft_cpp
enable_testing()
add_executable(test_pgft_cpp tests/test_pgft_cpp.cpp)
target_compile_options(test_pgft_cpp PRIVATE ${PGFT_CXX_FLAGS})
target_link_libraries(test_pgft_cpp PRIVATE pgft_core)
add_test(NAME test_pgft_cpp COMMAND test_pgft_cpp)

# 3. Zero-copy pybind11 Python Extension: _pgft_cpp
if(NOT DEFINED PYTHON_EXECUTABLE)
    if(EXISTS "/home/antigravity/.venv/bin/python")
        set(Python3_EXECUTABLE "/home/antigravity/.venv/bin/python")
    endif()
endif()
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

if(NOT pybind11_DIR)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_RES
    )
    if(PYBIND11_RES EQUAL 0)
        set(pybind11_DIR "${PYBIND11_CMAKE_DIR}")
    endif()
endif()

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_pgft_cpp MODULE src/pybind_module.cpp)
target_compile_options(_pgft_cpp PRIVATE ${PGFT_CXX_FLAGS})
target_link_libraries(_pgft_cpp PRIVATE pgft_core)

# Copy _pgft_cpp extension directly into src/pgft/ for seamless import
set(PGFT_PYTHON_PKG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/pgft")
add_custom_command(TARGET _pgft_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:_pgft_cpp>"
        "${PGFT_PYTHON_PKG_DIR}/$<TARGET_FILE_NAME:_pgft_cpp>"
    COMMENT "Copying _pgft_cpp extension module to ${PGFT_PYTHON_PKG_DIR}"
)
