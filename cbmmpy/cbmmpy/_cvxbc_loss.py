import numpy as np

from .convex_biclustering_weights import ConvexBiclusteringWeights


def _convex_biclustering_loss(
        X: np.ndarray, A: np.ndarray, lambda1: float, lambda2: float, weights: ConvexBiclusteringWeights
):
    row_weights = weights.to_dense_rows()
    col_weights = weights.to_dense_cols()

    # Compute first term of the loss function
    res = 0.5 * np.square(X - A).sum()

    # Row penalty
    temp_res = 0.0
    for i in range(X.shape[0]):
        for j in range(i):
            temp_res += row_weights[i, j] * np.linalg.norm(A[i, :] - A[j, :])
    res += lambda1 * temp_res

    # Column penalty
    temp_res = 0.0
    for i in range(X.shape[1]):
        for j in range(i):
            temp_res += col_weights[i, j] * np.linalg.norm(A[:, i] - A[:, j])
    res += lambda2 * temp_res

    return res
