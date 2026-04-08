cmake_minimum_required(VERSION 3.14)
project(elqcp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra")

include_directories(include)

# ----------------------------------------------------------------------------
# Eigen3 (header-only linear algebra library)
#
# Install on macOS:  brew install eigen
# Install on Ubuntu: apt-get install libeigen3-dev
# ----------------------------------------------------------------------------
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# ----------------------------------------------------------------------------
# Core library — Riccati recursion + sparse KKT (Eigen::SparseLU)
#
# No external solver dependency: Eigen is header-only and ships its own
# SparseLU implementation.
# ----------------------------------------------------------------------------
add_library(elqcp_lib
    src/riccati.cpp
    src/kkt_sparse_solver.cpp
)
target_link_libraries(elqcp_lib Eigen3::Eigen)

# ----------------------------------------------------------------------------
# Example: compare Riccati vs sparse KKT solver
# ----------------------------------------------------------------------------
add_executable(compare_solvers examples/compare_solvers.cpp)
target_link_libraries(compare_solvers elqcp_lib)
