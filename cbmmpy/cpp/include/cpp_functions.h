#ifndef PYBIND11TEMPLATE_TEMPLATE_H
#define PYBIND11TEMPLATE_TEMPLATE_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "Eigen/Eigen"


double square_cube(double x);

pybind11::dict vec_sum(const Eigen::VectorXd& a, const Eigen::VectorXd& b);

#endif //PYBIND11TEMPLATE_TEMPLATE_H
