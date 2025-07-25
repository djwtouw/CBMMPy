import numpy as np
from ._cvxbc_solver import _cbmm_solver
from .convex_biclustering_weights import ConvexBiclusteringWeights  # Assuming you have this class


class ConvexBiclustering:
    def __init__(
            self, lambda_rows: float = 1.0, lambda_cols: float = 1.0, burnin_iterations: int = 25,
            max_iterations: int = 100, eps_convergence: float = 1e-6
    ):
        self.lambda_rows = lambda_rows
        self.lambda_cols = lambda_cols
        self.burnin_iterations = burnin_iterations
        self.max_iterations = max_iterations
        self.eps_convergence = eps_convergence

    def fit(self, X: np.ndarray, weights: ConvexBiclusteringWeights):
        self._solve_result = _cbmm_solver(
            X, self.lambda_rows, self.lambda_cols, weights, self.burnin_iterations, self.max_iterations,
            self.eps_convergence
        )

        return self
