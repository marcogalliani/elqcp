cmake_minimum_required(VERSION 3.10)
project(elqcp C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -Wall -Wextra")

include_directories(include)

# ----------------------------------------------------------------------------
# Core library (Riccati + dense KKT + linear algebra)
# ----------------------------------------------------------------------------
add_library(elqcp_lib
    src/linalg.c
    src/riccati.c
    src/kkt_solver.c
)
target_link_libraries(elqcp_lib m)

# ----------------------------------------------------------------------------
# Optional: sparse KKT solver via UMFPACK (SuiteSparse)
#
# The KKT system is symmetric indefinite (saddle-point).  A proper symmetric
# indefinite factorization requires 2x2 Bunch-Kaufman pivoting (MA57, PARDISO)
# which is not available in SuiteSparse.  CHOLMOD's simplicial LDL^T stops at
# any non-positive pivot and therefore cannot handle this system.  UMFPACK
# (general sparse LU with partial pivoting) is the correct SuiteSparse solver.
#
# Install on macOS:  brew install suite-sparse
# Install on Ubuntu: apt-get install libsuitesparse-dev
# ----------------------------------------------------------------------------
find_path(SUITESPARSE_INCLUDE_DIR umfpack.h
    PATHS
        /opt/homebrew/include/suitesparse
        /opt/homebrew/include
        /usr/local/include/suitesparse
        /usr/local/include
        /usr/include/suitesparse
        /usr/include
)

find_library(UMFPACK_LIB umfpack
    PATHS /opt/homebrew/lib /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
)

find_library(SUITESPARSECONFIG_LIB suitesparseconfig
    PATHS /opt/homebrew/lib /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
)

if(SUITESPARSE_INCLUDE_DIR AND UMFPACK_LIB AND SUITESPARSECONFIG_LIB)
    message(STATUS "UMFPACK found — sparse KKT solver enabled")
    message(STATUS "  Headers : ${SUITESPARSE_INCLUDE_DIR}")
    message(STATUS "  Library : ${UMFPACK_LIB}")

    target_sources(elqcp_lib PRIVATE src/kkt_sparse_solver.c)
    target_include_directories(elqcp_lib PUBLIC ${SUITESPARSE_INCLUDE_DIR})
    target_link_libraries(elqcp_lib ${UMFPACK_LIB} ${SUITESPARSECONFIG_LIB})
    target_compile_definitions(elqcp_lib PUBLIC HAVE_UMFPACK)
else()
    message(WARNING
        "UMFPACK (SuiteSparse) not found — sparse KKT solver disabled.\n"
        "  macOS:  brew install suite-sparse\n"
        "  Ubuntu: apt-get install libsuitesparse-dev"
    )
endif()

# ----------------------------------------------------------------------------
# Example: compare all solvers
# ----------------------------------------------------------------------------
add_executable(compare_solvers examples/compare_solvers.c)
target_link_libraries(compare_solvers elqcp_lib)
