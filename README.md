# Extended Kalman Filter

Python EKF code and unit tests, translated to C++ with pybind11 wrappers so the same tests can exercise both implementations.

## Motivation

This project explores estimating the orientation, velocity, and position of a 3D thrust-vector-controlled (TVC) rocket from sensor measurements. Python provides a convenient environment for developing the filter equations and plotting their behavior. The C++ port implements the EKF with Eigen, while pybind11 exposes it to the existing Python test harness.

The core workflow is:

1. Implement the filter in Python using NumPy.
2. Write Python unit tests for its interface and state handling.
3. Port the EKF to C++ using Eigen for linear algebra.
4. Expose the C++ class through pybind11 and run the same tests against both versions.

A separate Python multiplicative EKF (MEKF) explores an error-state formulation with gyroscope and accelerometer biases. It currently has no C++ implementation.

## Implementations

| Implementation | State representation | Tests |
| --- | --- | --- |
| Python EKF | 10 values: quaternion (4), velocity (3), position (3) | Shared EKF suite |
| C++ EKF | Same 10-value state, using Eigen | Shared EKF suite through pybind11 |
| Python MEKF | 16-value nominal state, including sensor biases; 15-dimensional error covariance | Separate MEKF suite |

Prediction uses gyroscope and accelerometer inputs. Correction accepts accelerometer, magnetometer, GPS, lidar, and barometer measurements. The example scripts plot position and velocity estimates with covariance-based uncertainty bounds for a simple stationary scenario.

## Project layout

```text
src/
  python_src/
    ekf.py                 # NumPy EKF
    multiplicative_ekf.py  # Python MEKF
    util.py                # Quaternion and matrix helpers
    test_ekf.py             # Tests parameterized over Python and C++ EKFs
    test_mekf.py            # Python MEKF tests
    main.py                # EKF plotting example
    main_mekf.py            # MEKF plotting example
  cpp_src/
    EKF.h                  # C++ EKF implementation
    wrappers.cpp           # pybind11 bindings for the ekf_cpp module
    CMakeLists.txt
    thirdparty/
      eigen/               # Vendored Eigen source
      pybind11/            # Git submodule
```

## Setup and build

Requirements: Python 3.13 or newer, [uv](https://docs.astral.sh/uv/), CMake, and a C++17 compiler. Run these commands from the repository root:

```sh
git submodule update --init --recursive
uv sync
cmake -S src/cpp_src -B src/cpp_src/build \
  -DPYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build src/cpp_src/build --parallel 2
```

The build produces the `ekf_cpp` Python extension in `src/cpp_src/build`. Use the same Python environment for building and running the tests, and rebuild after changing Python versions.

pybind11 supplies the Python headers for the selected interpreter; a separate system Python header path is not required. These build commands have been checked on macOS with Python 3.13.

## Run tests

```sh
uv run pytest src/python_src -q
```

The EKF tests import `ekf_cpp` from the CMake build directory and use `parameterized_class` to run each test against both `ekf.EKF` and `ekf_cpp.EKF`. Build the extension first, even when running the shared suite from Python.

There are currently nine test cases: three checks for each EKF implementation and three for the Python MEKF. They check initial quaternion/velocity/position values and the shapes of the state and covariance after prediction and correction. They do not yet check numerical agreement between implementations, Jacobian correctness, or estimation accuracy on flight data.

## Run examples

```sh
uv run python src/python_src/main.py
uv run python src/python_src/main_mekf.py
```

These examples use the Python implementations and open Matplotlib plots. The C++ extension is only needed for the shared EKF tests.

## Current limitations and future work

This is an experimental implementation. The basic EKF uses zero process noise and identity measurement noise by default, and the plotting examples use simplified inputs. Sensor noise, reference frames, and physical conventions need validation for the intended vehicle and sensors.

Useful next steps include numerical comparison tests for the Python and C++ EKFs, finite-difference checks of the Jacobians, quaternion normalization checks, and evaluation with simulated trajectories or recorded sensor data.
