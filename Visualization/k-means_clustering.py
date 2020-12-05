import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib import cm

data = make_blobs(n_samples=1000, n_features=4, centers=5, random_state=0)

n_cluster = 5

x, y = data

n_clusters = 6
fig, ax = plt.subplots(n_clusters - 1, 2, figsize=(12, 10))

for index, n_cluster in enumerate(range(2, n_clusters + 1)):

    cluster = KMeans(n_clusters=n_cluster, random_state=0)
    y_pred = cluster.fit_predict(x)
    s_score = silhouette_score(x, y_pred)
    i_score = cluster.inertia_
    s_values = silhouette_samples(x, y_pred)
    s_values_dict = {i: sorted(s_values[y_pred == i]) for i in range(n_cluster)}
    print('silhouette_score: %0.2f, inertia: %0.2f' % (s_score, i_score))
    y_low = 10
    for i in s_values_dict:
        y_high = y_low + len(s_values_dict[i])
        color = cm.nipy_spectral(int(i) / n_cluster)

        ax[index][0].fill_betweenx(np.arange(y_low, y_high), s_values_dict[i], facecolor=color)
        y_low = y_high + 10
        ax[index][0].axvline(x=s_score, c='r', ls='--', lw=0.5)

    color1 = cm.nipy_spectral(y_pred.astype(int) / n_cluster)
    ax[index][1].scatter(x[:, 0], x[:, 1], c=color1, s=10, alpha=0.5)
    ax[index][1].scatter(cluster.cluster_centers_[:, 0], cluster.cluster_centers_[:, 1], marker='x', s=200)

    ax[index][0].set_title('n_cluster:%d, silhouette plot for clusters' % (n_cluster))
    ax[index][1].set_title('n_cluster:%d, visualization of the clusered data' % (n_cluster))

plt.show()
