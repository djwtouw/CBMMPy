import numpy as np
from cbmmpy._cbmmpy import sylvester_solver
from cbmmpy import _convex_biclustering_update

# %%

A = np.array([[1.0, 0.0, 0.0, 0.0],
              [0.0, 2.0, 0.0, 0.0],
              [0.0, 0.0, 3.0, 0.0],
              [0.0, 0.0, 0.0, 4.0]])
B = np.array([[2.0, 1.0, 1.5],
              [1.0, 2.0, 1.0],
              [1.5, 1.0, 2.0]])
C = np.array([[3.0, 1.0, 3.0],
              [2.0, 0.0, 4.0],
              [1.0, 1.0, 1.0],
              [2.0, 2.0, 2.0]])

eig_A = np.linalg.eigh(A)
eig_B = np.linalg.eigh(B)

Z = eig_A.eigenvectors.T @ C @ eig_B.eigenvectors

X = np.zeros(C.shape)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        X[i, j] = Z[i, j] / (eig_A.eigenvalues[i] + eig_B.eigenvalues[j])

X = eig_A.eigenvectors @ X @ eig_B.eigenvectors.T

print(X.round(4))
print(sylvester_solver(A, B, C).round(4))

