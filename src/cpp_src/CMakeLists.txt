cmake_minimum_required(VERSION 3.10)
project(ekf_code) 

set(CMAKE_CXX_STANDARD 17)       # or 14 if you prefer
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(thirdparty/pybind11)
include_directories(thirdparty/eigen)

# # Include the FetchContent module
# include(FetchContent)

# # 1. Fetch Eigen
# FetchContent_Declare(
#     Eigen3
#     GIT_REPOSITORY https://gitlab.com
#     GIT_TAG        3.4.0 # Or your preferred stable version
# )
# FetchContent_MakeAvailable(Eigen3)

# # 2. Fetch pybind11
# FetchContent_Declare(
#     pybind11
#     GIT_REPOSITORY https://github.com
#     GIT_TAG        v2.13.6 # Or your preferred stable version
# )
# FetchContent_MakeAvailable(pybind11)

pybind11_add_module(ekf_cpp wrappers.cpp EKF.h)

target_sources(ekf_cpp
  PRIVATE
    EKF.h
)

