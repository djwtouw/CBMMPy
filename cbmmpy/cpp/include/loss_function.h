#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include "optimization_parameters.h"
#include "optimization_constants.h"


double convex_biclustering_loss(
        const Eigen::MatrixXd &X,
        const OptimizationParameters &optimization_parameters,
        const OptimizationConstants &optimization_constants
);


#endif //LOSS_FUNCTION_H