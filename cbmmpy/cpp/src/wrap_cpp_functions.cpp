#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "Eigen/Eigen"
#include "sylvester_solver.h"
#include "convex_biclustering.h"

namespace py = pybind11;


PYBIND11_MODULE(_cbmmpy, m)
{
    m.def("sylvester_solver", &export_sylvester_solver, "Test");
    m.def("convex_biclustering", &convex_biclustering, "Test");
}
