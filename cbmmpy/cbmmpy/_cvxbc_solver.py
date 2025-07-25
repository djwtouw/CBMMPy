import numpy as np

from .convex_biclustering_weights import ConvexBiclusteringWeights
from ._cvxbc_loss import _convex_biclustering_loss
from ._cbmmpy import sylvester_solver


def _cbmm_update(
        X: np.ndarray, A: np.ndarray, lambda1: float, lambda2: float, weights: ConvexBiclusteringWeights
):
    row_weights = weights.to_dense_rows()
    col_weights = weights.to_dense_cols()

    # For majorization step: compute C0 and G0
    C0 = np.zeros((A.shape[0], A.shape[0]))
    G0 = np.zeros((A.shape[1], A.shape[1]))

    # Populate C0
    for i in range(A.shape[0]):
        for ii in range(i):
            # Compute element of interest
            temp = row_weights[i, ii] / max(np.linalg.norm(A[i, :] - A[ii, :]), 1e-6)

            # Modify C0
            C0[i, ii] -= temp
            C0[ii, i] -= temp
            C0[i, i] += temp
            C0[ii, ii] += temp

    # Populate G0
    for j in range(A.shape[1]):
        for jj in range(j):
            # Compute element of interest
            temp = col_weights[j, jj] / max(np.linalg.norm(A[:, j] - A[:, jj]), 1e-6)

            # Modify C0
            G0[j, jj] -= temp
            G0[jj, j] -= temp
            G0[j, j] += temp
            G0[jj, jj] += temp

    # Get variables to solve S1*A + A*S2 = X
    S1 = np.eye(A.shape[0]) + lambda1 * C0
    S2 = lambda2 * G0

    # Return the update for A
    return sylvester_solver(S1, S2, X)


def _cbmm_solver(X, lambda1, lambda2, weights, burnin_iterations, max_iterations, eps_convergence):
    A = X.copy()

    loss1 = _convex_biclustering_loss(X, A, lambda1, lambda2, weights)
    loss0 = 2 * loss1 + 1
    iteration = 0

    losses = [loss1]

    while loss0 / loss1 - 1 > eps_convergence and iteration < max_iterations:
        A_update = _cbmm_update(X, A, lambda1, lambda2, weights)

        if iteration > burnin_iterations:
            A = 2 * A_update - A
        else:
            A = A_update

        # Bookkeeping to check convergence
        loss0 = loss1
        loss1 = _convex_biclustering_loss(X, A, lambda1, lambda2, weights)
        losses.append(loss1)
        max_iterations += 1

    return dict(A=A, losses=losses)
