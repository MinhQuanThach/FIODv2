import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
import random

random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

def plot_cw_sf_layer(all_factors, layer_key='layer0', pca_dim=50, tsne_perplexity=30, out_file=None):
    X_cw = np.vstack(all_factors[layer_key]['CW'])  # (N_cw, D)
    X_sf = np.vstack(all_factors[layer_key]['SF'])  # (N_sf, D)
    X_rf = np.vstack(all_factors[layer_key]['RF'])  # (N_rf, D)
    print(f"Layer {layer_key}: CW {X_cw.shape}, SF {X_sf.shape}, RF {X_rf.shape}")

    # normalize each embedding (L2) like paper did
    def l2norm_rows(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
        return X / norms
    X_cw = l2norm_rows(X_cw)
    X_sf = l2norm_rows(X_sf)
    X_rf = l2norm_rows(X_rf)

    # optionally subsample to keep TSNE fast
    max_points = 1000
    n_cw, n_sf, n_rf = X_cw.shape[0], X_sf.shape[0], X_rf.shape[0]
    if n_cw + n_sf + n_rf > max_points:
        keep_each = max_points // 3
        idx_cw = np.random.choice(n_cw, min(keep_each, n_cw), replace=False)
        idx_sf = np.random.choice(n_sf, min(keep_each, n_sf), replace=False)
        idx_rf = np.random.choice(n_rf, min(keep_each, n_rf), replace=False)
        X_cw = X_cw[idx_cw]
        X_sf = X_sf[idx_sf]
        X_rf = X_rf[idx_rf]

    X = np.vstack([X_cw, X_sf, X_rf])
    labels = np.array([0]*len(X_cw) + [1]*len(X_sf) + [2]*len(X_rf))  # 0=CW, 1=SF, 2=RF
    label_names = {0: 'CW', 1: 'SF', 2: 'RF'}

    # PCA pre-reduction
    pca_n = min(pca_dim, X.shape[1], X.shape[0]-1)
    pca = PCA(n_components=pca_n, random_state=random_seed)
    Xp = pca.fit_transform(X)

    # TSNE (2D)
    tsne = TSNE(n_components=2, perplexity=min(tsne_perplexity, max(5, Xp.shape[0]//3)),
                random_state=random_seed, init='pca', n_iter=1500)
    X2 = tsne.fit_transform(Xp)

    # Plot
    plt.figure(figsize=(8,6))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for k in [0,1,2]:
        idxs = np.where(labels == k)[0]
        plt.scatter(X2[idxs,0], X2[idxs,1], s=12, alpha=0.7, label=label_names[k], color=colors[k])
    plt.legend()
    plt.title(f"t-SNE of fog factors — {layer_key}")
    plt.xlabel('tsne-1'); plt.ylabel('tsne-2')
    if out_file:
        plt.savefig(out_file, dpi=200)
        print("Saved to", out_file)
    plt.show()

    # Quantitative metrics on original embedding space (not t-SNE)
    try:
        sil = silhouette_score(X, labels)
        print("Silhouette score:", sil)
    except Exception as e:
        print("Silhouette score failed:", e)
    # kmeans ARI
    kmeans = KMeans(n_clusters=2, random_state=random_seed).fit(X)
    ari = adjusted_rand_score(labels, kmeans.labels_)
    print("Adjusted Rand Index (labels vs KMeans):", ari)