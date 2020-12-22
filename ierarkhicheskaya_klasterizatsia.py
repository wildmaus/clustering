#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.widgets import Slider, RadioButtons
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
get_ipython().run_line_magic('matplotlib', 'notebook')



# созданине данных
numberOfNodes = 1500
noisy_circles = datasets.make_circles(n_samples = numberOfNodes, factor = .1,
                                      noise = .1)
noisy_moons = datasets.make_moons(n_samples = numberOfNodes, noise = .1)
blobs = datasets.make_blobs(n_samples = numberOfNodes, random_state = 7)
no_structure = np.random.rand(numberOfNodes, 2), None

data = [noisy_circles, noisy_moons, blobs, no_structure]
methods = ['single', 'complete', 'average', 'ward']
numOfClasters = [2, 3, 4]

# результаты применения иерархической кластеризации
# с разными методами посчета рассиряния после объединения кластеров
# и разным количеством конечных класстеров
for numOfClaster in numOfClasters:
    for method in methods:
        fig, axs = plt.subplots(2, 2)
        plt.suptitle(f'Method = {method}, Number of clasters = {numOfClaster}')
        for i, dataset in enumerate(data):
            axs[i // 2, i % 2].clear()
            X, y = dataset
            Z = linkage(X, method = method, metric = 'euclidean')
            label = fcluster(Z, numOfClaster, criterion = 'maxclust')
            axs[i // 2, i % 2].scatter(X[:,0], X[:,1], c = label)
plt.show()

from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

# анализ оптимального числа кластеров
# в случае неудачного выбора программа сообщит об этом
X, y = datasets.make_blobs(n_samples = numberOfNodes, random_state = 7)
numOfClasters = [2, 3, 4, 5]

for numOfClaster in numOfClasters:
    flag = False
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(9, 4)

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (numOfClaster + 1) * 10])

    Z = linkage(X, method = 'ward', metric = 'euclidean')
    label = fcluster(Z, numOfClaster, criterion = 'maxclust')

    silhouette_avg = silhouette_score(X, label)
    print("For n_clusters =", numOfClaster,
          "The average silhouette_score is :", silhouette_avg)

    sample_silhouette_values = silhouette_samples(X, label)

    y_lower = 10
    for i in range(numOfClaster):
        ith_cluster_silhouette_values = sample_silhouette_values[label == i + 1]

        ith_cluster_silhouette_values.sort()
        if max(ith_cluster_silhouette_values) < silhouette_avg or min(ith_cluster_silhouette_values) < 0:
            flag = True


        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / numOfClaster)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor = color, edgecolor = color, alpha = 0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    if flag:
        print('Неудачный выбор числа кластеров')

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x = silhouette_avg, color = "red", linestyle = "--") 
    #если у кластера значение ниже этого =>
    #неудачный выбор количества класстеров

    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    colors = cm.nipy_spectral(label.astype(float) / numOfClaster)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for Hierarchical clustering on sample data "
                  "with n_clusters = %d" % numOfClaster),
                 fontsize=14, fontweight='bold')

plt.show()

