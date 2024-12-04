#ifndef SYLVESTER_SOLVER_H
#define SYLVESTER_SOLVER_H

#include "Eigen/Eigen"

Eigen::MatrixXd sylvester(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& C);

Eigen::MatrixXd sylvester(const Eigen::VectorXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& C);

Eigen::MatrixXd sylvester(const Eigen::MatrixXd& A, const Eigen::VectorXd& B, const Eigen::MatrixXd& C);

Eigen::MatrixXd sylvester(const Eigen::VectorXd& A, const Eigen::VectorXd& B, const Eigen::MatrixXd& C);

Eigen::MatrixXd sylvester_wrapper(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& C);

#endif //SYLVESTER_SOLVER_H
