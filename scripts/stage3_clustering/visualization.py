# stage3_clustering/visualization.py

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

try:
    import umap
    has_umap = True
except:
    has_umap = False


def plot_tsne(X, labels, title="t-SNE Clustering"):
    emb = TSNE(n_components=2, random_state=42, perplexity=30)\
            .fit_transform(X.toarray())

    plt.figure(figsize=(8,6))
    plt.scatter(emb[:,0], emb[:,1], c=labels, cmap='tab10', s=8)
    plt.title(title)
    plt.show()


def plot_umap(X, labels, title="UMAP Clustering"):
    if not has_umap:
        print("UMAP not installed. Skipping.")
        return

    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42)
    emb = reducer.fit_transform(X)

    plt.figure(figsize=(8,6))
    plt.scatter(emb[:,0], emb[:,1], c=labels, cmap='tab10', s=8)
    plt.title(title)
    plt.show()
