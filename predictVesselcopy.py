# -*- coding: utf-8 -*-
"""
Vessel prediction using k-means clustering on standardized features. If the
number of vessels is not specified, assume 20 vessels.

@author: Kevin S. Xu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
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
# from sklearn.neighbors import DistanceMetric
from sklearn.naive_bayes import LabelBinarizer

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense

import matplotlib.pyplot as plt
import math

def predictWithK(testFeatures, numVessels, l, m, trainFeatures=None, 
                 trainLabels=None):
    # Unsupervised prediction, so training data is unused

    scaler = MinMaxScaler()
    testFeatures = scaler.fit_transform(testFeatures)
    linkage_matrix = linkage(testFeatures, method=l, metric=m)
    predVessels = fcluster(linkage_matrix, numVessels, criterion="maxclust")
    return predVessels

def predictWithoutK(testFeatures, l, m, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    min_score = math.inf
    optimal_k = 1
    for k in np.arange(2, 51, 1):
        labels = predictWithK(testFeatures, k, l, m, trainFeatures, trainLabels)
        if (np.unique(labels).size > 1):
            score = davies_bouldin_score(testFeatures, labels)
            if score < min_score:
                min_score = score
                optimal_k = k
    return predictWithK(testFeatures, optimal_k, l, m, trainFeatures, trainLabels)

# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    
    from utils import loadData, plotVesselTracks
    # data = loadData('set2.csv')
    # features = data[:,2:]
    # labels = data[:,1]

    data3 = loadData('set3noVID.csv')
    features3 = data3[:, 2:]

    
    # numVessels = np.unique(labels).size

    l_max = None
    m_max = None
    numVessels = 1
    min_score = math.inf
    for l in ("single", "complete", "average", "weighted"):
        for m in ('braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'):
            predVesselsWithoutK = predictWithoutK(features3, l, m)
            predNumVessels = np.unique(predVesselsWithoutK).size
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
    predVesselsWithoutK = predictWithoutK(features3, l_max, m_max)
    predNumVessels = np.unique(predVesselsWithoutK).size
    # ariWithoutK = adjusted_rand_score(labels, predVesselsWithoutK)
    score = davies_bouldin_score(features3, predVesselsWithoutK)
    
    # print(f'Adjusted Rand index given K = {numVessels}: {ari}')
    print(f'Davies Bouldin score for estimated K = {predNumVessels}: '
          + f'{score}')

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
    