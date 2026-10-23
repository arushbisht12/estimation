#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "EKF.h"

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(ekf_cpp, m) {
    m.doc() = "C++ EKF implementation wrappers"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");

    py::class_<EKF>(m, "EKF")
        .def(py::init<Vector4d, Vector3d, Vector3d>())
        .def("prediction", &EKF::prediction)
        .def("correction", &EKF::correction)
        .def("f", &EKF::f)
        .def("h", &EKF::h)
        .def("F", &EKF::F)
        .def("H", &EKF::H)
        .def("W", &EKF::W)
        .def("omega", &EKF::omega)
        .def("rotation_body_to_earth", &EKF::rotation_body_to_earth)
        .def("rotation_earth_to_body", &EKF::rotation_earth_to_body)
        .def_property_readonly("mean", &EKF::mean)
        .def_property_readonly("cov", &EKF::cov)
        .def_property_readonly("q", &EKF::q)
        .def_property_readonly("vel", &EKF::vel)
        .def_property_readonly("pos", &EKF::pos)
        .def("set_x", &EKF::set_x);

}


