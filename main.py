import numpy as np
from cbmmpy import ConvexBiclusteringWeights, ConvexBiclustering


# %%

# Initialize data
np.random.seed(123)
X = np.random.rand(6, 3)
lambda1 = 0.1
lambda2 = 2.0

# Use package functions
weights = ConvexBiclusteringWeights(k_rows=X.shape[0] - 1, k_cols=X.shape[1] - 1, phi_rows=1, phi_cols=1, normalize=False)
weights.compute_weights(X)

model = ConvexBiclustering(lambda_rows=lambda1, lambda_cols=lambda2)
model.fit(X, weights)
