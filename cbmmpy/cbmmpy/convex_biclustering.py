import numpy as np
from .convex_biclustering_weights import ConvexBiclusteringWeights
from ._cbmmpy import convex_biclustering


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
        result = convex_biclustering(
            X.T,
            weights._row_indices.T,
            weights._col_indices.T,
            weights._row_weights,
            weights._col_weights,
            X.shape[0],
            X.shape[1],
            self.lambda_rows,
            self.lambda_cols,
            self.eps_convergence,
            self.burnin_iterations,
            self.max_iterations,
        )

        self._solve_result = result["A"].T
        self._losses = result["losses"]

        return self
