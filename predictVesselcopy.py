# -*- coding: utf-8 -*-
"""
Vessel prediction using k-means clustering on standardized features. If the
number of vessels is not specified, assume 20 vessels.

@author: Kevin S. Xu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster, correspond
from sklearn.metrics.pairwise import haversine_distances, euclidean_distances
from scipy.spatial.distance import euclidean, squareform
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.naive_bayes import LabelBinarizer
from sklearn.neighbors import NearestNeighbors
# import tensorflow as tf
# from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import LSTM, Dense

import matplotlib.pyplot as plt
import math

def predictWithK(testFeatures, numVessels, trainFeatures=None, 
                 trainLabels=None):
    # Unsupervised prediction, so training data is unused

    def preprocess_data(data):
        # data[:, 1] = data[:, 3] * 60.0 # y (latitude) in nautical miles
        # data[:, 2] = data[:, 4] * 60.0 # x (longitude) in nautical miles
        # data[:, 3] = data[:, 5] / 36000.0 # v in nautical miles per second
        # data[:, 4] = np.deg2rad(data[:, 6] / 10.0) # theta in radians
        # Time sinusoids
        # hours = np.sin(2*np.pi* (data[:, 2].astype("datetime64[h]").astype(float)) / 24)
        # hours = hours[:, np.newaxis]
        
        # DistanceTravelled  
        # dist_diffs = np.linalg.norm(np.diff(data[:, [3,4]], axis=0), axis=1)
        # dist_diffs = np.hstack([dist_diffs, np.nan])
        # dist_diffs = dist_diffs[:, np.newaxis]

        # Neighbor speed
        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(data[:, [1, 2]])
        neighbor_indices = nn.kneighbors(data[:, [1, 2]], return_distance=False)[:,1:]
        neighbor_speed = data[neighbor_indices, 3].mean(axis=1)[:, np.newaxis]
        neighbor_course = data[neighbor_indices, 4].mean(axis=1)[:, np.newaxis]
        
        cog = data[:, -1] # course over ground
        # Difference in COG between reports
        cog_diff = np.abs(np.diff(cog))
        cog_diff = np.hstack([cog_diff, np.zeros(1)])[:,None]

        # Cyclic encoding 
        cog_sin = np.sin(2*np.pi*cog/360)[:,None]
        cog_cos = np.cos(2*np.pi*cog/360)[:,None]
        
        # Directional velocity
        velocity = data[:, 3][:,None] # speed over ground
        dir_velocity = velocity * cog_cos
        
        # # Append features
        data_ext = np.hstack([data, neighbor_speed, dir_velocity, cog_sin, cog_cos, cog_diff])
        return data_ext
    
    testFeatures = preprocess_data(testFeatures)
    # scaler = StandardScaler()
    # testFeatures = scaler.fit_transform(testFeatures)

    # scaler = StandardScaler()
    # testFeatures = scaler.fit_transform(testFeatures)

    linkage_matrix = linkage(testFeatures, method=l, metric=m)
    predVessels = fcluster(linkage_matrix, numVessels, criterion="maxclust")
    return predVessels

def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    min_score = math.inf
    optimal_k = 1
    for k in np.arange(2, 26, 1):
        labels = predictWithK(testFeatures, k, l, m, trainFeatures, trainLabels)
        if np.unique(labels).size > 1:
            score = davies_bouldin_score(testFeatures, labels)
            if min_score > score:
                min_score = score
                optimal_k = k
    return predictWithK(testFeatures, optimal_k, l, m, trainFeatures, trainLabels)

# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    
    from utils import loadData, plotVesselTracks
    # data = loadData('set2.csv')
    # features = data[:,2:]
    # labels = data[:,1]

    data3 = loadData('set2noVID.csv')
    features3 = data3[:, 2:]

    
    # numVessels = np.unique(labels).size

    l_max = None
    m_max = None
    numVessels = 1
    min_score = math.inf
    for l in ("single", "complete", "average", "weighted"):
        for m in ('braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'euclidean', 'hamming', 'jaccard', 'mahalanobis', 'minkowski', 'seuclidean', 'sqeuclidean', 'yule'):
            predVesselsWithoutK = predictWithoutK(features3, l, m)
            predNumVessels = np.unique(predVesselsWithoutK).size
            if predNumVessels > 1:
                score = davies_bouldin_score(features3, predVesselsWithoutK)
                if min_score > score:
                    min_score = score
                    l_max = l
                    m_max = m
                    numVessels = predNumVessels
                    print("min_score", min_score, "method", l_max, "metric", m_max, "pred_vessels", predNumVessels)
        
    print("Final score:", min_score)
    print("Final method:", l_max)
    print("Final metric:", m_max)
    print("Final pred_vessels", numVessels)

    # predVesselsWithK = predictWithK(features, numVessels)
    # ariWithK = adjusted_rand_score(labels, predVesselsWithK)

    # Prediction without specified number of vessels
    predVesselsWithoutK = predictWithoutK(features3)
    predNumVessels = np.unique(predVesselsWithoutK).size
    print(predNumVessels)
    # ariWithoutK = adjusted_rand_score(labels, predVesselsWithoutK)
    # score = silhouette_score(features3, predVesselsWithoutK, metric=m_max)
    # score = silhouette_score(features3, predVesselsWithoutK)
    
    # print(f'Adjusted Rand index given K = {numVessels}: {ari}')
    # print(f'Silhouette Score for estimated K = {predNumVessels}: '
    #       + f'{score}')

    # Plot all vessel tracks with no coloring
    plotVesselTracks(features3[:,[2,1]])
    plt.title('All vessel tracks')
    # Plot vessel tracks colored by prediction and actual labels
    # plotVesselTracks(features[:,[2,1]], predVesselsWithK)
    # plt.title('Vessel tracks by cluster with K')
    plotVesselTracks(features3[:,[2,1]], predVesselsWithoutK)
    plt.title('Vessel tracks by cluster without K')
    # plotVesselTracks(features[:,[2,1]], labels)
    # plt.title('Vessel tracks by label')
    # # plt.show()
    