import numpy as np
from cbmmpy._cbmmpy import sylvester


def compute_row_weights(X, phi=1):
    # Number of objects
    n = X.shape[0]

    # Initialize result
    res = np.empty((n, n))

    # Compute weights
    for i in range(n):
        res[i, i] = 0

        for j in range(i):
            # First compute distance
            res[i, j] = np.linalg.norm(X[i, :] - X[j, :])

            # Then weight
            res[i, j] = np.exp(-phi * res[i, j])

            # Add symmetric element
            res[j, i] = res[i, j]

    return res

def compute_col_weights(X, phi=1):
    # Number of variables
    p = X.shape[1]

    # Initialize result
    res = np.empty((p, p))

    # Compute weights
    for i in range(p):
        res[i, i] = 0

        for j in range(i):
            # First compute distance
            res[i, j] = np.linalg.norm(X[:, i] - X[:, j])

            # Then weight
            res[i, j] = np.exp(-phi * res[i, j])

            # Add symmetric element
            res[j, i] = res[i, j]

    return res

# %%

def loss(X, A, lambda1, lambda2, row_weights, col_weights):
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


# Initialize data
np.random.seed(123)
X = np.random.rand(6, 3)
row_weights = compute_row_weights(X)
col_weights = compute_col_weights(X)
lambda1 = 0.1
lambda2 = 2.0

# Initial solution
A = X.copy()

# Compute loss
print(loss(X, A, lambda1, lambda2, row_weights, col_weights))

# For majorization step: compute C0 and G0
C0 = np.zeros((A.shape[0], A.shape[0]))
G0 = np.zeros((A.shape[1], A.shape[1]))

# Populate C0
for i in range(A.shape[0]):
    for ii in range(i):
        # Compute element of interest
        temp = row_weights[i, ii] / np.linalg.norm(A[i, :] - A[ii, :])

        # Modify C0
        C0[i, ii] -= temp
        C0[ii, i] -= temp
        C0[i, i] += temp
        C0[ii, ii] += temp

# Populate G0
for j in range(A.shape[1]):
    for jj in range(j):
        # Compute element of interest
        temp = col_weights[j, jj] / np.linalg.norm(A[:, j] - A[:, jj])

        # Modify C0
        G0[j, jj] -= temp
        G0[jj, j] -= temp
        G0[j, j] += temp
        G0[jj, jj] += temp

# Get variables to solve S1*A + A*S2 = X
S1 = np.eye(A.shape[0]) + lambda1 * C0
S2 = lambda2 * G0

# Update A
A = sylvester(S1, S2, X)

# Compute loss
print(loss(X, A, lambda1, lambda2, row_weights, col_weights))

