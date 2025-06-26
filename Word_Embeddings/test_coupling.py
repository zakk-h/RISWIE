from sklearn.neighbors import NearestNeighbors
import numpy as np

def read(file, dtype='float'):
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
    n, d = X.shape
    cov = (X.T @ X) / n
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    print('Finished finidng pca axes')
    return eigvecs, eigvals

def axes_projection(X, Y, k):
    assert X.ndim == 2 and Y.ndim == 2, "X and Y must be 2D arrays"
    n, d = X.shape
    assert Y.shape == (n, d), "X and Y must have the same shape (n, d)"

    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)

    Ux, eigvals_x = compute_pca_axes(Xc)
    Uy, eigvals_y = compute_pca_axes(Yc)

    if k is not None:
        Ux = Ux[:, :k]
        Uy = Uy[:, :k]
        eigvals_x = eigvals_x[:k]
        eigvals_y = eigvals_y[:k]

    alpha = Xc @ Ux
    beta = Yc @ Uy
    print('Finished finding projections')
    return alpha, beta, eigvals_x, eigvals_y

def riswie_coupling(alpha, beta, eigvals_x, eigvals_y):
    """
    Compute weighted RISWIE coupling using PCA eigenvalues as weights.

    Parameters:
        alpha: (n, k) projection of X on its top-k PCA axes
        beta:  (n, k) projection of Y on its top-k PCA axes
        eigvals: (k,) eigenvalues from PCA (already sorted descending)

    Returns:
        Gamma_total: (n, n) soft coupling matrix
    """
    n, k = alpha.shape
    Gamma_total = np.zeros((n, n))

    weights = (eigvals_x + eigvals_y) / 2
    weights /= weights.sum()  # normalize

    for i in range(k):
        x_i = alpha[:, i]
        y_i = beta[:, i]
        x_order = np.argsort(x_i)
        y_order = np.argsort(y_i)

        Gamma_i = np.zeros((n, n))
        for j in range(n):
            Gamma_i[x_order[j], y_order[j]] = 1.0 / n

        Gamma_total += weights[i] * Gamma_i

    return Gamma_total


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

    alpha, beta, eigvals_x, eigvals_y = axes_projection(X, Y, None)
    Gamma = riswie_coupling(alpha, beta, eigvals_x, eigvals_y)

    matching = np.argmax(Gamma, axis=1)
    pred_pairs = [(words_en[i], words_es[matching[i]]) for i in range(len(matching))]

    print("Top 100 aligned word pairs:")
    for i in range(100):
        print(f"{i+1}: {pred_pairs[i][0]} ↔ {pred_pairs[i][1]}")

    gt_dict = load_ground_truth_dict('data/en-es.0-6500.txt')  # Update path as needed
    acc, correct, total = compute_accuracy(pred_pairs, gt_dict)
    print(f"\nAccuracy against ground truth: {acc:.2%} ({correct}/{total})")

if __name__ == '__main__':
    main()
