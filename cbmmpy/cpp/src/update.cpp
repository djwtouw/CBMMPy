#include "update.h"
#include "optimization_parameters.h"
#include "optimization_constants.h"
#include "sylvester_solver.h"
#include "Eigen/Eigen"


Eigen::MatrixXd update_no_fusions(
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

    // TODO: Investigate whether it is better to store C0 as sparse matrix
    Eigen::MatrixXd C0(A.cols(), A.cols());
    C0.setZero();

    for (int k = 0; k < row_weights.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(row_weights, k); it; ++it) {
            int obj_i = it.row();
            int obj_j = it.col();

            if (obj_i >= obj_j) continue;

            double temp = row_weights.coeff(obj_i, obj_j) / std::max(row_distances.coeff(obj_i, obj_j), 1e-6);

            C0(obj_i, obj_j) -= temp;
            C0(obj_j, obj_i) -= temp;
            C0(obj_i, obj_i) += temp;
            C0(obj_j, obj_j) += temp;
        }
    }

    // TODO: Investigate whether it is better to store G0 as sparse matrix
    Eigen::MatrixXd G0(A.rows(), A.rows());
    G0.setZero();

    for (int k = 0; k < col_weights.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(col_weights, k); it; ++it) {
            int var_i = it.row();
            int var_j = it.col();

            if (var_i >= var_j) continue;

            double temp = col_weights.coeff(var_i, var_j) / std::max(col_distances.coeff(var_i, var_j), 1e-6);

            G0(var_i, var_j) -= temp;
            G0(var_j, var_i) -= temp;
            G0(var_i, var_i) += temp;
            G0(var_j, var_j) += temp;
        }
    }

    // Get variables to solve the Sylvester equation and update A
    Eigen::MatrixXd S1 = optimization_constants.lambda_rows * C0;
    S1.diagonal().array() += 1.0;
    Eigen::MatrixXd S2 = optimization_constants.lambda_cols * G0;

    // Solve
    return sylvester_solver(S2, S1, X);
}
