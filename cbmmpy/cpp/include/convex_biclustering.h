#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "Eigen/Eigen"

namespace py = pybind11;


py::dict convex_biclustering(
        const Eigen::MatrixXd &X,
        const Eigen::MatrixXi &row_indices,
        const Eigen::MatrixXi &col_indices,
        const Eigen::VectorXd &row_values,
        const Eigen::VectorXd &col_values,
        int n_rows,
        int n_cols,
        double lambda_rows,
        double lambda_cols,
        double convergence_tolerance,
        int burn_in_iterations,
        int max_iterations
);
