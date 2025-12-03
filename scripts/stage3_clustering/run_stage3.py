# stage3_clustering/run_stage3.py

from pathlib import Path
from scipy.sparse import load_npz
import numpy as np

from clustering_algorithms import (
    run_kmeans, run_dbscan, run_hdbscan, run_gmm
)

from visualization import plot_tsne, plot_umap


def main():
    BASE = Path("../uk_us_outputs")

    print("Loading BM25 matrix...")
    X = load_npz(BASE / "X_bm25_uk_us.npz")
    y = np.load(BASE / "y_labels_num.npy")

    print(f"X shape: {X.shape}")
    print(f"y length: {len(y)}")

    # ========================
    # Run all clustering models
    # ========================
    labels_km, metrics_km  = run_kmeans(X, y)
    labels_db, metrics_db  = run_dbscan(X, y, eps=0.7, min_samples=5)
    labels_hdb, metrics_hdb = run_hdbscan(X, y, min_cluster_size=10, min_samples=5)
    labels_gmm, metrics_gmm = run_gmm(X, y)

    # Print results
    print("\n=== RESULTS ===")
    print("K-Means:   ", metrics_km)
    print("DBSCAN:    ", metrics_db)
    print("HDBSCAN:   ", metrics_hdb)
    print("GMM:       ", metrics_gmm)

    # ============
    # Visualizations
    # ============
    plot_tsne(X, labels_km, "t-SNE – KMeans")
    plot_umap(X, labels_km, "UMAP – KMeans")

    print("\nDONE: Stage 3 complete.")


if __name__ == "__main__":
    main()
