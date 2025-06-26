"""
    1. get embeddings from vec
    2. preprocess vec into word, matrix (numpy) form
    3. for xs and xt (source and target), center, use pca to reduce dimensions
    4. sort based on eigenvalues (assume they decay), be aware of sign flip (?)
    5. match ith with ith axis
    6. for each axis, match based on percentile (same as in monge problem in 1D closed form)
    7. get final coupling (it's not a matrix)
"""
from sklearn.neighbors import NearestNeighbors
import numpy as np

def read(file, dtype='float'):
    """
    read vec data
    """
    print("Reading .vec file...")

    header = file.readline().split(' ')
    count = int(header[0])
    dim = int(header[1])

    words = []
    matrix = np.empty((count, dim), dtype=dtype)

    for i in range(count):
        word, vec = file.readline().split(' ', 1)
        words.append(word)
        matrix[i] = np.fromstring(vec, sep=' ', dtype=dtype)

    print(f"Finished reading. Matrix shape: {matrix.shape}")
    return words, matrix

def compute_pca_axes(X):
    """
    Compute the PCA axes (eigenvectors) of the centered point cloud X.
    Returns:
        U : (d, d) array whose columns are the eigenvectors of the covariance of X,
            sorted by decreasing eigenvalue.
    """
    # X: (n, d)
    n, d = X.shape
    # Compute covariance matrix (d x d)
    cov = (X.T @ X) / n
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort eigenvalues in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    print('Finished finidng pca axes')
    return eigvecs  # columns are eigenvectors

def axes_projection(X, Y):
    """
    Get projected distributions on pca axes.
    Steps:
      1. Center X and Y.
      2. Compute PCA axes for each.
      3. Project onto PCA axes to get 1D coordinates.
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
    print('Finished finding projections')
    return alpha, beta


def mutual_neighbors(X, Y, k=10, metric='cosine'):
    """
    Find mutual nearest neighbors between two point sets X and Y.

    Parameters:
        X: (n, d) numpy array
        Y: (m, d) numpy array
        k: number of nearest neighbors to consider
        metric: distance metric, e.g., 'cosine', 'euclidean'

    Returns:
        mutual_pairs: list of (i, j) index pairs where:
            - j is in the k-nearest neighbors of X[i]
            - i is in the k-nearest neighbors of Y[j]
    """
    # 1. X → Y
    nn_xy = NearestNeighbors(n_neighbors=k, metric=metric).fit(Y)
    _, indices_xy = nn_xy.kneighbors(X)  # shape: (n, k)

    # 2. Y → X
    nn_yx = NearestNeighbors(n_neighbors=k, metric=metric).fit(X)
    _, indices_yx = nn_yx.kneighbors(Y)  # shape: (m, k)

    # 3. Mutual check
    mutual_pairs = []
    for i in range(X.shape[0]):
        for j in indices_xy[i]:
            if i in indices_yx[j]:  # i ∈ NN(Y[j])
                mutual_pairs.append((i, j))
    print('Finished matching nearest neighbors')
    return mutual_pairs

def load_ground_truth_dict(filename):
    gt_dict = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            en, es = line.strip().split()
            if en not in gt_dict:
                gt_dict[en] = set()
            gt_dict[en].add(es)
    return gt_dict

def compute_accuracy(pred_pairs, gt_dict):
    correct = 0
    for en, es in pred_pairs:
        if en in gt_dict and es in gt_dict[en]:
            correct += 1
    total = len(pred_pairs)
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total

def main():
    with open('data/wiki.en.10k.vec', 'r', encoding='utf-8') as f_en, open('data/wiki.es.10k.vec', 'r', encoding='utf-8') as f_es:
        words_en, X = read(f_en)
        words_es, Y = read(f_es)

    X_proj, Y_proj = axes_projection(X, Y)

    mutual_pairs = mutual_neighbors(X_proj, Y_proj, k=10, metric='euclidean')

    pred_word_pairs = [(words_en[i], words_es[j]) for (i, j) in mutual_pairs]

    print("\nTop 100 Mutual Nearest Neighbor Pairs:")
    for idx, (en_word, es_word) in enumerate(pred_word_pairs[:100]):
        print(f"{idx+1:3d}: {en_word} ↔ {es_word}")

    gt_dict = load_ground_truth_dict("data/en-es.0-6500.txt") 
    accuracy, correct, total = compute_accuracy(pred_word_pairs, gt_dict)

    print(f"\n Accuracy over ALL {total} mutual pairs: {accuracy:.4f} ({correct} / {total} correct)")



if __name__ == "__main__":
    main()
