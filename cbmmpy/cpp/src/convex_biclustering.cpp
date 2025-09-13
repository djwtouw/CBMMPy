#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "convex_biclustering.h"
#include "loss_function.h"
#include "optimization_parameters.h"
#include "update.h"
#include "Eigen/Eigen"

namespace py = pybind11;


Eigen::SparseMatrix<double> create_square_sparse_matrix(
        const Eigen::MatrixXi &indices,
        const Eigen::VectorXd &values,
        int dim
) {
    assert(indices.rows() == 2 && "Indices matrix must have 2 rows");
    assert(indices.cols() == values.size() && "Number of indices must match number of values");

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(values.size());

    for (int k = 0; k < values.size(); ++k) {
        int i = indices(0, k);
        int j = indices(1, k);
        double v = values(k);
        triplets.emplace_back(i, j, v);
    }

    Eigen::SparseMatrix<double> sparse_matrix(dim, dim);
    sparse_matrix.setFromTriplets(triplets.begin(), triplets.end());

    return sparse_matrix;
}


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
) {
    Eigen::SparseMatrix row_weights = create_square_sparse_matrix(row_indices, row_values, n_rows);
    Eigen::SparseMatrix col_weights = create_square_sparse_matrix(col_indices, col_values, n_cols);

    OptimizationParameters optimization_parameters(X, row_weights, col_weights);
    OptimizationConstants optimization_constants{
        lambda_rows,
        lambda_cols,
        convergence_tolerance,
        burn_in_iterations,
        max_iterations
    };

    // Initialize loss values
    double loss1 = convex_biclustering_loss(X, optimization_parameters, optimization_constants);
    double loss0 = 2.0 * loss1 + 1.0;
    int iteration = 0;

    // Vector for the losses
    Eigen::VectorXd losses(optimization_constants.max_iterations + 1);
    losses(0) = loss1;

    while (loss0 / loss1 - 1.0 > optimization_constants.convergence_tolerance &&
           iteration < optimization_constants.max_iterations) {
        // After the burn in iterations, apply step doubling
        if (iteration >= optimization_constants.burn_in_iterations) {
            Eigen::MatrixXd A_update = update_no_fusions(X, optimization_parameters, optimization_constants);
            optimization_parameters.A = 2.0 * A_update - optimization_parameters.A;
        } else {
            // Update without step doubling
            optimization_parameters.A = update_no_fusions(X, optimization_parameters, optimization_constants);
        }

        // Update the matrices tracking the internal distances
        optimization_parameters.update_row_distances();
        optimization_parameters.update_col_distances();

        // Bookkeeping
        loss0 = loss1;
        loss1 = convex_biclustering_loss(X, optimization_parameters, optimization_constants);
        losses(++iteration) = loss1;
    }

    losses.conservativeResize(iteration + 1);

    // Prepare the result
    py::dict result;
    result["A"] = optimization_parameters.A;
    result["losses"] = losses;

    return result;
}
