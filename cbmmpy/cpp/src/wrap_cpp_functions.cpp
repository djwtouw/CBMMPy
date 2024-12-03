#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "cpp_functions.h"


PYBIND11_MODULE(_cbmmpy, m)
{
    m.def("_square_cube", &square_cube, "Test");
    m.def("_vec_sum", &vec_sum, "Test");
}
