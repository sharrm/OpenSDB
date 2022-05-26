# -*- coding: utf-8 -*-
"""
@author: sharrm

Created to experiment with clustering techniques
"""

# open source functions
import numpy as np
from sklearn.cluster import KMeans


class Cluster:
    def kmeans(self, band):
        
        shape = band.shape
        
        kmeans = KMeans(n_clusters = 2, random_state=0).fit(band)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        # cluster_centers = np.reshape(cluster_centers, shape)
        
        return centers, labels





