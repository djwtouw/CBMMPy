#ifndef UPDATE_H
#define UPDATE_H

#include "optimization_parameters.h"
#include "optimization_constants.h"
#include "Eigen/Eigen"


Eigen::MatrixXd update_no_fusions(
        const Eigen::MatrixXd &X,
        const OptimizationParameters &optimization_parameters,
        const OptimizationConstants &optimization_constants
);

#endif //UPDATE_H
