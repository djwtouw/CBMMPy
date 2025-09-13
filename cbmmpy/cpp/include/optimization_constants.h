#ifndef OPTIMIZATION_CONSTANTS_H
#define OPTIMIZATION_CONSTANTS_H


struct OptimizationConstants {
    double lambda_rows;
    double lambda_cols;
    double convergence_tolerance;
    int burn_in_iterations;
    int max_iterations;
};


#endif //OPTIMIZATION_CONSTANTS_H
