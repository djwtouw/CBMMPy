#include "Eigen/Eigen"
#include "sylvester_solver.h"


Eigen::MatrixXd sylvester_solver(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& C)
{
    // Perform spectral decompositions
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_A(A);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_B(B);

    // Create references for readability of the code
    const Eigen::MatrixXd& Q = eig_A.eigenvectors();
    const Eigen::VectorXd& R = eig_A.eigenvalues();
    const Eigen::MatrixXd& S = eig_B.eigenvectors();
    const Eigen::VectorXd& T = eig_B.eigenvalues();

    // Transform C into Z
    Eigen::MatrixXd Z = Q.transpose() * C * S;

    // Initialize intermediate result
    Eigen::MatrixXd Y(C.rows(), C.cols());

    // Compute elements of Y
    for (int j = 0; j < Y.cols(); j++) {
        for (int i = 0; i < Y.rows(); i++) {
            Y(i, j) = Z(i, j) / (R(i) + T(j));
        }
    }

    // Compute X, that solves AX+XB=C
    Eigen::MatrixXd X = Q * Y * S.transpose();

    return X;
}


Eigen::MatrixXd sylvester_solver(const Eigen::VectorXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& C)
{
    return C;
}


Eigen::MatrixXd sylvester_solver(const Eigen::MatrixXd& A, const Eigen::VectorXd& B, const Eigen::MatrixXd& C)
{
    return C;
}


Eigen::MatrixXd sylvester_solver(const Eigen::VectorXd& A, const Eigen::VectorXd& B, const Eigen::MatrixXd& C)
{
    return C;
}


Eigen::MatrixXd export_sylvester_solver(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& C)
{
    return sylvester_solver(A, B, C);
}
