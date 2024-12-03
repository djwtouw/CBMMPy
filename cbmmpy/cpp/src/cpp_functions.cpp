//
// Created by dtouw on 22/09/22.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "cpp_functions.h"
#include "internal_0.h"
#include "internal_1.h"
#include "Eigen/Eigen"


double square_cube(double x)
{
    return square(x) + cube(x);
}


pybind11::dict vec_sum(const Eigen::VectorXd& a, const Eigen::VectorXd& b)
{
    pybind11::dict res;
    res["vec"] = a + b;

    return res;
}
