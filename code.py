import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_pca_axes(X):
    """
    Compute the PCA axes (eigenvectors) of the centered point cloud X.
    Returns:
        U : (d, d) array whose columns are the eigenvectors of the covariance of X,
            sorted by decreasing eigenvalue.
    """
    # X: (n, d)
    n, d = X.shape
    # Center the data
    X_centered = X - X.mean(axis=0)
    # Compute covariance matrix (d x d)
    cov = (X_centered.T @ X_centered) / n
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort eigenvalues in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    return eigvecs  # columns are eigenvectors

def compute_1d_w2_squared(a, b):
    """
    Compute the squared 1D Wasserstein-2 distance between two empirical distributions
    supported on equal number of points, given by arrays a and b.
    That is, W2^2 = mean((sort(a) - sort(b))^2).
    """
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    return np.mean((a_sorted - b_sorted) ** 2)

def riswie_distance(X, Y):
    """
    Compute the RISWIE distance between two point clouds X and Y in R^d.
    Steps:
      1. Center X and Y.
      2. Compute PCA axes for each.
      3. Project onto PCA axes to get 1D coordinates.
      4. Build cost matrix C_{l,k} = min_s W2^2(A_l, s * B_k).
      5. Solve assignment via Hungarian algorithm.
      6. Determine optimal sign for each matched axis pair.
      7. Compute final distance = sqrt((1/d) * sum of chosen W2^2).
    Assumes X and Y have the same number of points n and same ambient dimension d.
    """
    # Ensure shapes
    assert X.ndim == 2 and Y.ndim == 2, "X and Y must be 2D arrays"
    n, d = X.shape
    assert Y.shape == (n, d), "X and Y must have the same shape (n, d)"

    # 1. Center both point clouds
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)

    # 2. Compute PCA axes
    Ux = compute_pca_axes(Xc)  # (d, d)
    Uy = compute_pca_axes(Yc)  # (d, d)

    # 3. Project onto PCA axes to get coordinates
    #    alpha: (n, d) coordinates of X in its PCA basis
    #    beta:  (n, d) coordinates of Y in its PCA basis
    alpha = Xc @ Ux  # shape (n, d)
    beta  = Yc @ Uy  # shape (n, d)

    # 4. Build cost matrix C of shape (d, d)
    #    C[l, k] = min_{s in {+1, -1}} W2^2(alpha[:, l], s * beta[:, k])
    C = np.zeros((d, d))
    for l in range(d):
        a_l = alpha[:, l]
        for k in range(d):
            b_k = beta[:, k]
            # Compute W2^2 for both sign choices
            w2_pos = compute_1d_w2_squared(a_l, b_k)
            w2_neg = compute_1d_w2_squared(a_l, -b_k)
            C[l, k] = min(w2_pos, w2_neg)

    # 5. Solve assignment problem to find optimal pairing of axes
    row_ind, col_ind = linear_sum_assignment(C)
    # row_ind[l] = l, col_ind[l] = matched k

    # 6. For each matched pair (l, k), determine optimal sign and record the chosen W2^2
    w2_chosen = np.zeros(d)
    for idx_l, idx_k in zip(row_ind, col_ind):
        a_l = alpha[:, idx_l]
        b_k = beta[:, idx_k]
        w2_pos = compute_1d_w2_squared(a_l, b_k)
        w2_neg = compute_1d_w2_squared(a_l, -b_k)
        if w2_pos <= w2_neg:
            w2_chosen[idx_l] = w2_pos
        else:
            w2_chosen[idx_l] = w2_neg

    # 7. Compute final distance
    distance = np.sqrt(np.mean(w2_chosen))
    return distance

if __name__ == "__main__":
    # Example test: generate a random point cloud in R^3, apply a random rigid transformation,
    # and check that RISWIE distance is (near) zero.

    # 1. Generate random point cloud X (n x d)
    n = 252
    d = 5
    X = np.random.randn(n, d)

    # 2. Generate a random orthonormal matrix R (via QR decomposition)
    A = np.random.randn(d, d)
    Q, _ = np.linalg.qr(A)
    R = Q  # orthonormal rotation/reflection

    # 3. Random translation vector t
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # 4. Apply the rigid transformation to X to get Y
    Y = X @ R.T + t  # shape (n, d)

    # 5. Compute RISWIE distance
    dist = riswie_distance(X, Y)
    print(f"RISWIE distance between X and its rigid transform Y: {dist:.6e}")
    # If implementation is correct, dist should be very close to 0 (up to numerical precision).
