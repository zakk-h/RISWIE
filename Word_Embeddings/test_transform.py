import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
import ot


def read(file, dtype='float'):
    """Read .vec embedding file into word list and matrix."""
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
    """Compute PCA axes for X with eigen-decomposition."""
    n, d = X.shape
    cov = (X.T @ X) / n
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    eigvals = eigvals[idx]
    print('Finished finding PCA axes')
    return eigvecs, eigvals


# def get_transform(X, Y, k=None):
#     """
#     Compute linear transform T from source to target
#     based on PCA axes alignment.
#     """
#     assert X.ndim == 2 and Y.ndim == 2 and X.shape == Y.shape
#     Xc = X - X.mean(axis=0)
#     Yc = Y - Y.mean(axis=0)

#     Ux, _ = compute_pca_axes(Xc)
#     Uy, _ = compute_pca_axes(Yc)

#     if k is not None:
#         Ux = Ux[:, :k]
#         Uy = Uy[:, :k]

#     T = Ux @ Uy.T
#     print("Computed transform matrix T")
#     return T

def compute_1d_w2_squared(a, b):
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    return np.mean((a_sorted - b_sorted) ** 2)

def get_transform(X, Y, k=None):
    assert X.shape == Y.shape and X.ndim == 2
    n, d = X.shape

    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)

    Ux, _ = compute_pca_axes(Xc)
    Uy, _ = compute_pca_axes(Yc)

    if k is not None:
        Ux = Ux[:, :k]
        Uy = Uy[:, :k]
        d = k  # update dimension

    alpha = Xc @ Ux
    beta = Yc @ Uy

    # Axis matching by 1D W2
    C = np.zeros((d, d))
    signs = np.zeros((d, d))
    for l in range(d):
        for k_ in range(d):
            w2_pos = compute_1d_w2_squared(alpha[:, l], beta[:, k_])
            w2_neg = compute_1d_w2_squared(alpha[:, l], -beta[:, k_])
            if w2_pos <= w2_neg:
                C[l, k_] = w2_pos
                signs[l, k_] = 1
            else:
                C[l, k_] = w2_neg
                signs[l, k_] = -1

    row_ind, col_ind = linear_sum_assignment(C)

    P = np.zeros((d, d))
    S = np.zeros((d, d))
    for l, k_ in zip(row_ind, col_ind):
        P[l, k_] = 1
        S[l, l] = signs[l, k_]

    T = Ux @ S @ P @ Uy.T
    print("Computed transform matrix T")
    return T



def compute_transport_plan(X, Y, T, reg=0.1):
    """Compute Sinkhorn OT plan under cost ‖x - T y‖²."""
    n = X.shape[0]
    a = np.ones(n) / n
    b = np.ones(n) / n

    Y_trans = Y @ T.T
    C = ot.dist(X, Y_trans, metric='sqeuclidean')
    C = C / C.max()  # normalize to avoid overflow
    Gamma = ot.sinkhorn(a, b, C, reg)
    return Gamma


def load_ground_truth_dict(filename):
    """Load gold dictionary for BLI evaluation."""
    gt_dict = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            en, es = line.strip().split()
            if en not in gt_dict:
                gt_dict[en] = set()
            gt_dict[en].add(es)
    return gt_dict


def compute_accuracy(pred_pairs, gt_dict):
    """Compute precision against ground truth dictionary."""
    correct = 0
    for en, es in pred_pairs:
        if en in gt_dict and es in gt_dict[en]:
            correct += 1
    total = len(pred_pairs)
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


def main():
    # Step 1: Load bilingual embeddings
    with open('data/wiki.en.10k.vec', 'r', encoding='utf-8') as f_en, \
         open('data/wiki.es.10k.vec', 'r', encoding='utf-8') as f_es:
        words_en, X = read(f_en)
        words_es, Y = read(f_es)

    # Step 2: PCA projection alignment
    T = get_transform(X, Y, k=50)

    # Step 3: Sinkhorn optimal transport (cost = ‖x - Ty‖²)
    Gamma = compute_transport_plan(X, Y, T, reg=0.1)

    # Step 4: Predict translation by selecting max coupling
    matching = np.argmax(Gamma, axis=1)
    pred_word_pairs = [(words_en[i], words_es[j]) for i, j in enumerate(matching)]

    # Step 5: Print top 100 results
    print("\nTop 100 Matched Pairs:")
    for idx, (en_word, es_word) in enumerate(pred_word_pairs[:100]):
        print(f"{idx+1:3d}: {en_word} ↔ {es_word}")

    # Step 6: Evaluate translation accuracy
    gt_dict = load_ground_truth_dict("data/en-es.0-6500.txt")
    accuracy, correct, total = compute_accuracy(pred_word_pairs, gt_dict)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({correct} / {total} correct)")


if __name__ == "__main__":
    main()
