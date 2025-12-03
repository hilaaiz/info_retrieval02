# stage3_clustering/evaluation.py

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate_clustering(true_labels, cluster_labels):
    """Evaluate clustering quality vs true labels.
       Handles noise (-1) and flips label mapping automatically."""

    mask = cluster_labels != -1
    tl = np.array(true_labels)[mask]
    cl = np.array(cluster_labels)[mask]

    if len(cl) == 0 or len(np.unique(cl)) < 2:
        return 0, 0, 0, 0

    # Convert cluster IDs → 0/1
    uniq = sorted(np.unique(cl))
    mapping = {uniq[0]: 0, uniq[1]: 1}
    cl01 = np.array([mapping[x] for x in cl])

    results = []
    for flip in [False, True]:
        pred = cl01 if not flip else 1 - cl01

        p = precision_score(tl, pred)
        r = recall_score(tl, pred)
        f = f1_score(tl, pred)
        a = accuracy_score(tl, pred)

        results.append((p, r, f, a))

    return max(results, key=lambda x: x[3])
