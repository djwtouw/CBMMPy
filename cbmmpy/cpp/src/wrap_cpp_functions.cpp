#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "Eigen/Eigen"
#include "cpp_functions.h"
#include "sylvester_solver.h"

namespace py = pybind11;


PYBIND11_MODULE(_cbmmpy, m)
{
    m.def("_square_cube", &square_cube, "Test");
    m.def("_vec_sum", &vec_sum, "Test");
    m.def("sylvester", &sylvester_wrapper, "Test");
}
