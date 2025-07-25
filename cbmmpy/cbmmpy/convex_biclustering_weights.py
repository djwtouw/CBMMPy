import numpy as np
from sklearn.neighbors import KDTree


def _pairwise_squared_distances(X):
    G = X @ X.T
    H = np.sum(X ** 2, axis=1)
    squared_distances = H[:, None] + H[None, :] - 2 * G

    return squared_distances[np.triu_indices_from(squared_distances, k=1)]


def _gaussian_weights(W, phi):
    return np.exp(-phi * W)


def _create_symmetric_weight_matrix(indices, weights):
    n, k = indices.shape

    # Flatten the neighbor indices and weights
    i_src = np.repeat(np.arange(n), k)
    i_dst = indices.ravel()
    w = weights.ravel()

    # Stack both (i, j) and (j, i)
    indices_symmetric = np.vstack([
        np.column_stack((i_src, i_dst)),
        np.column_stack((i_dst, i_src))
    ])

    weights_symmetric = np.concatenate([w, w])

    # Sort based on the indices
    sort_idx = np.lexsort((indices_symmetric[:, 1], indices_symmetric[:, 0]))
    sorted_indices = indices_symmetric[sort_idx]
    sorted_weights = weights_symmetric[sort_idx]

    # Filter duplicates
    _, unique_idx = np.unique(sorted_indices, axis=0, return_index=True)
    unique_indices = sorted_indices[unique_idx]
    unique_weights = sorted_weights[unique_idx]

    return unique_indices, unique_weights


class ConvexBiclusteringWeights:
    def __init__(
            self, k_rows=None, k_cols=None, phi_rows=None, phi_cols=None,
            normalize=True
    ):
        """
        Initialize the Weights class.

        Parameters:
            k_rows (int or None): Number of row-wise neighbors to compute.
            k_cols (int or None): Number of column-wise neighbors to compute.
        """
        self.k_rows = k_rows
        self.k_cols = k_cols
        self.phi_rows = phi_rows
        self.phi_cols = phi_cols
        self._normalize = normalize
        self._n_rows = None
        self._n_cols = None
        self._row_indices = None
        self._row_weights = None
        self._col_indices = None
        self._col_weights = None

    def compute_weights(self, X):
        """
        Compute the k-nearest neighbors for rows and/or columns of X.

        Parameters:
            X (numpy.ndarray): Input data matrix of shape (n_samples, n_features).
        """
        self._n_rows = X.shape[0]
        self._n_cols = X.shape[1]
        if self.k_rows is not None:
            self._compute_row_weights(X)
        if self.k_cols is not None:
            self._compute_col_weights(X)

    def _compute_row_weights(self, X):
        tree = KDTree(X)
        distances, indices = tree.query(X, k=self.k_rows + 1)

        squared_distances = distances[:, 1:] ** 2

        if self._normalize:
            median_squared_distance = np.median(_pairwise_squared_distances(X))
            normalized_squared_distances = squared_distances / median_squared_distance
        else:
            normalized_squared_distances = squared_distances

        self._row_indices, self._row_weights = _create_symmetric_weight_matrix(
            indices[:, 1:],
            _gaussian_weights(normalized_squared_distances, self.phi_rows)
        )

    def _compute_col_weights(self, X):
        X_col = X.T
        tree = KDTree(X_col)
        distances, indices = tree.query(X_col, k=self.k_cols + 1)

        squared_distances = distances[:, 1:] ** 2

        if self._normalize:
            median_squared_distance = np.median(_pairwise_squared_distances(X_col))
            normalized_squared_distances = squared_distances / median_squared_distance
        else:
            normalized_squared_distances = squared_distances

        self._col_indices, self._col_weights = _create_symmetric_weight_matrix(
            indices[:, 1:],
            _gaussian_weights(normalized_squared_distances, self.phi_cols)
        )

    def get_row_indices(self):
        return self._row_indices

    def get_row_weights(self):
        return self._row_weights

    def get_col_indices(self):
        return self._col_indices

    def get_col_weights(self):
        return self._col_weights

    def to_dense_rows(self):
        if self._row_indices is None or self._row_weights is None:
            raise ValueError("Row weights have not been computed yet.")

        W = np.zeros((self._n_rows, self._n_rows))

        for neighbors, weight in zip(self._row_indices, self._row_weights):
            W[neighbors[0], neighbors[1]] = weight

        return W

    def to_dense_cols(self):
        if self._col_indices is None or self._col_weights is None:
            raise ValueError("Col weights have not been computed yet.")

        W = np.zeros((self._n_cols, self._n_cols))

        for neighbors, weight in zip(self._col_indices, self._col_weights):
            W[neighbors[0], neighbors[1]] = weight

        return W
