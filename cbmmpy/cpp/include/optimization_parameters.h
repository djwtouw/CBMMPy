#ifndef OPTIMIZATIONPARAMETERS_H
#define OPTIMIZATIONPARAMETERS_H

#include "Eigen/Eigen"


class OptimizationParameters {
private:
    // Distance matrices need to be initialized every time fusion have occurred as these matrices will shrink
    bool col_distances_initialized = false;
    bool row_distances_initialized = false;
public:
    Eigen::MatrixXd A;
    Eigen::SparseMatrix<double> row_weights;
    Eigen::SparseMatrix<double> col_weights;
    Eigen::SparseMatrix<double> row_distances;
    Eigen::SparseMatrix<double> col_distances;

    OptimizationParameters(const Eigen::MatrixXd& A,
                           const Eigen::SparseMatrix<double>& row_weights,
                           const Eigen::SparseMatrix<double>& col_weights)
            : A(A),
              row_weights(row_weights),
              col_weights(col_weights)
    {
        initialize_col_distances();
        initialize_row_distances();
    }

    void initialize_row_distances() {
        // Rows/objects
        row_distances.resize(row_weights.rows(), row_weights.cols());
        row_distances.reserve(row_weights.nonZeros());

        for (int k = 0; k < row_weights.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(row_weights, k); it; ++it) {
                // Due to transposition, the rows (objects) of the data set are the columns in the Eigen matrix
                row_distances.insert(it.row(), it.col()) = (A.col(it.row()) - A.col(it.col())).norm();
            }
        }

        row_distances.makeCompressed();
        row_distances_initialized = true;
    }

    void initialize_col_distances() {
        // Columns/variables
        col_distances.resize(col_weights.rows(), col_weights.cols());
        col_distances.reserve(col_weights.nonZeros());

        for (int k = 0; k < col_weights.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(col_weights, k); it; ++it) {
                // Due to transposition, the columns (variables) of the data set are the rows in the Eigen matrix
                col_distances.insert(it.row(), it.col()) = (A.row(it.row()) - A.row(it.col())).norm();
            }
        }

        col_distances.makeCompressed();
        col_distances_initialized = true;
    }

    void update_row_distances() {
        assert(row_distances_initialized);

        for (int k = 0; k < row_distances.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(row_distances, k); it; ++it) {
                int obj_i = it.row();
                int obj_j = it.col();

                row_distances.coeffRef(obj_i, obj_j) = (A.col(obj_i) - A.col(obj_j)).norm();
            }
        }
    }

    void update_col_distances() {
        assert(col_distances_initialized);

        for (int k = 0; k < col_distances.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(col_distances, k); it; ++it) {
                int var_i = it.row();
                int var_j = it.col();

                col_distances.coeffRef(var_i, var_j) = (A.row(var_i) - A.row(var_j)).norm();
            }
        }
    }

    void perform_fusions() {
        // TODO: implement
        row_distances_initialized = false;
        col_distances_initialized = false;
    }
};


#endif
