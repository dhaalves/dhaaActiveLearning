from sklearn.cluster import KMeans, DBSCAN
import numpy as np

from .dataset import Dataset

if __name__ == '__main__':

    dataset = Dataset('LEA-53')
    x_train, y_train, x_test, y_test, f_train, f_test = dataset.get_split()

    kmeans = KMeans(n_clusters=53)
    kmeans.fit(x_train)
    labels_kmeans = kmeans.labels_

    # DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
    # Finds core samples of high density and expands clusters from them.
    # Good for data which contains clusters of similar density.
    dbscan = DBSCAN(eps=8.0, min_samples=10)
    dbscan.fit(x_train)
    labels_dbscan = dbscan.labels_
    idx = dbscan.core_sample_indices_

    for e in np.arange(5.0, 100.0, 1.0):
        for ms in np.arange(2, 30, 2):
            dbscan = DBSCAN(eps=e, min_samples=ms)
            dbscan.fit(dataset.features)
            labels = dbscan.labels_
            noise_samples = len(list(filter(lambda x: x == -1, labels)))
            clustered_samples = list(filter(lambda x: x != -1, labels))
            n_clusters = len(set(clustered_samples))
            print('eps %.2f -  min_samples %d - n_clusters %d - noise_samples %d' % (e, ms, n_clusters, noise_samples))
