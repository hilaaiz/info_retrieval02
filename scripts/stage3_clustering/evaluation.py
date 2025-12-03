import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def evaluate_clustering(true_labels, cluster_labels):
    """
    Evaluate clustering quality vs true labels.
    Works for:
    - KMeans (2 clusters)
    - GMM (2 clusters)
    - DBSCAN / HDBSCAN (any number of clusters, noise included)

    Strategy for DBSCAN/HDBSCAN:
      • Remove noise (-1)
      • Keep only the 2 largest clusters
      • Map them to {0,1} in both possible ways
      • Return the best accuracy
    """

    # Remove noise
    mask = cluster_labels != -1
    tl = np.array(true_labels)[mask]
    cl = np.array(cluster_labels)[mask]

    # If nothing to evaluate
    if len(cl) == 0:
        return 0, 0, 0, 0

    # Unique cluster IDs
    uniq = np.unique(cl)

    # If exactly 2 clusters → normal evaluation
    if len(uniq) == 2:
        mapping = {uniq[0]: 0, uniq[1]: 1}
        cl01 = np.array([mapping[x] for x in cl])
        return _evaluate_binary(tl, cl01)

    # If more than 2 clusters → pick the two largest clusters
    counts = {cid: np.sum(cl == cid) for cid in uniq}
    top2 = sorted(counts, key=counts.get, reverse=True)[:2]

    # Mask to keep only these two clusters
    mask2 = np.isin(cl, top2)
    tl = tl[mask2]
    cl = cl[mask2]

    mapping = {top2[0]: 0, top2[1]: 1}
    cl01 = np.array([mapping[x] for x in cl])

    return _evaluate_binary(tl, cl01)


def _evaluate_binary(true_labels, pred_labels):
    """Try both label flips and return best accuracy."""

    results = []

    for flip in [False, True]:
        p = pred_labels if not flip else 1 - pred_labels

        pr = precision_score(true_labels, p, zero_division=0)
        re = recall_score(true_labels, p, zero_division=0)
        f1 = f1_score(true_labels, p, zero_division=0)
        acc = accuracy_score(true_labels, p)

        results.append((pr, re, f1, acc))

    return max(results, key=lambda x: x[3])
