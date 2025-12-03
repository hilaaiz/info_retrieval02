# stage3_clustering/clustering_algorithms.py

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

import hdbscan

from .evaluation import evaluate_clustering


def run_kmeans(X, y):
    print("\n=== K-MEANS ===")
    km = KMeans(n_clusters=2, random_state=42)
    labels = km.fit_predict(X)
    return labels, evaluate_clustering(y, labels)


def run_dbscan(X, y, eps=0.7, min_samples=5):
    print("\n=== DBSCAN ===")
    db = DBSCAN(metric="cosine", eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    return labels, evaluate_clustering(y, labels)


def run_hdbscan(X, y, min_cluster_size=10, min_samples=5):
    print("\n=== HDBSCAN ===")
    hdb = hdbscan.HDBSCAN(
        metric="cosine",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    labels = hdb.fit_predict(X)
    return labels, evaluate_clustering(y, labels)


def run_gmm(X, y):
    print("\n=== GMM ===")
    gmm = GaussianMixture(n_components=2, random_state=42)
    labels = gmm.fit_predict(X.toarray())
    return labels, evaluate_clustering(y, labels)
