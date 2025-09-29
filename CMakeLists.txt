cmake_minimum_required(VERSION 3.14)
project(consensus_admm)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

# Include directories
include_directories(include)

# Create C++ library
file(GLOB_RECURSE CPP_SOURCES "src/cpp/*.cpp")
add_library(consensus_admm_core STATIC ${CPP_SOURCES})

# Link Eigen3
target_link_libraries(consensus_admm_core Eigen3::Eigen)

# Set compile options
target_compile_options(consensus_admm_core PRIVATE -O3 -march=native)

# Create Python binding
pybind11_add_module(consensus_admm_cpp src/python/binding.cpp)
target_link_libraries(consensus_admm_cpp PRIVATE consensus_admm_core)

# Compiler-specific options for Python bindings
target_compile_definitions(consensus_admm_cpp PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# Enable testing
enable_testing()
add_subdirectory(tests)