# stage3_clustering/clustering_algorithms.py

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_distances

import hdbscan

from .evaluation import evaluate_clustering


# ----------------------------------------------------------
# K-MEANS
# ----------------------------------------------------------
def run_kmeans(X, y):
    print("\n=== K-MEANS ===")
    km = KMeans(n_clusters=2, random_state=42)
    labels = km.fit_predict(X)
    return labels, evaluate_clustering(y, labels)


# ----------------------------------------------------------
# DBSCAN with cosine distance
# ----------------------------------------------------------
def run_dbscan(X, y, eps=0.7, min_samples=5):
    print("\n=== DBSCAN ===")
    db = DBSCAN(
        metric="cosine",     # REQUIRED BY HOMEWORK
        eps=eps,
        min_samples=min_samples
    )
    labels = db.fit_predict(X.toarray())   # DBSCAN requires dense
    return labels, evaluate_clustering(y, labels)


# ----------------------------------------------------------
# HDBSCAN with cosine distance (using precomputed matrix)
# ----------------------------------------------------------
def run_hdbscan(X, y, min_cluster_size=10, min_samples=5):
    print("\n=== HDBSCAN ===")
    print("Computing cosine distance matrix for HDBSCAN...")

    # HDBSCAN cannot use metric="cosine" directly → precompute distances
    D = cosine_distances(X)  # Works with sparse BM25 matrix

    hdb = hdbscan.HDBSCAN(
        metric="precomputed",       # REQUIRED for cosine
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )

    labels = hdb.fit_predict(D)
    return labels, evaluate_clustering(y, labels)


# ----------------------------------------------------------
# Gaussian Mixture Model
# ----------------------------------------------------------
def run_gmm(X, y):
    print("\n=== GMM ===")
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="diag",  # MUCH faster (valid for homework)
        random_state=42
    )
    labels = gmm.fit_predict(X.toarray())
    return labels, evaluate_clustering(y, labels)
