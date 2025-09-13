#include "loss_function.h"
#include "optimization_parameters.h"
#include "optimization_constants.h"


double convex_biclustering_loss(
        const Eigen::MatrixXd &X,
        const OptimizationParameters &optimization_parameters,
        const OptimizationConstants &optimization_constants
) {
    // Aliases
    const Eigen::MatrixXd &A = optimization_parameters.A;
    const Eigen::SparseMatrix row_weights = optimization_parameters.row_weights;
    const Eigen::SparseMatrix col_weights = optimization_parameters.col_weights;
    const Eigen::SparseMatrix row_distances = optimization_parameters.row_distances;
    const Eigen::SparseMatrix col_distances = optimization_parameters.col_distances;

    // First term
    double loss = 0.5 * (X - A).squaredNorm();

    // Row penalty
    double row_penalty = 0.0;

    for (int k = 0; k < row_weights.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(row_weights, k); it; ++it) {
            int obj_i = it.row();
            int obj_j = it.col();

            if (obj_i >= obj_j) continue;

            row_penalty += row_weights.coeff(obj_i, obj_j) * row_distances.coeff(obj_i, obj_j);
        }
    }

    // Column penalty
    double col_penalty = 0.0;

    for (int k = 0; k < col_weights.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(col_weights, k); it; ++it) {
            int var_i = it.row();
            int var_j = it.col();

            if (var_i >= var_j) continue;

            col_penalty += col_weights.coeff(var_i, var_j) * col_distances.coeff(var_i, var_j);
        }
    }

    return loss + optimization_constants.lambda_rows * row_penalty + optimization_constants.lambda_cols * col_penalty;
}
