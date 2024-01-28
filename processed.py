# -*- coding: utf-8 -*-
"""
Vessel prediction using k-means clustering on standardized features. If the
number of vessels is not specified, assume 20 vessels.

@author: Kevin S. Xu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score
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
from sklearn.metrics.pairwise import haversine_distances
from sklearn.neighbors import NearestNeighbors
# from sklearn.neighbors import DistanceMetric
import matplotlib.pyplot as plt
import math

def preprocess_data(data):
        # data[:, 1] = data[:, 3] * 60.0 # y (latitude) in nautical miles
        # data[:, 2] = data[:, 4] * 60.0 # x (longitude) in nautical miles
        # data[:, 3] = data[:, 5] / 36000.0 # v in nautical miles per second
        # data[:, 4] = np.deg2rad(data[:, 6] / 10.0) # theta in radians
        # Time sinusoids
    hours = np.sin(2*np.pi* (data[:, 2].astype("datetime64[h]").astype(float)) / 24)
    hours = hours[:, np.newaxis]
        
        # DistanceTravelled  
    dist_diffs = np.linalg.norm(np.diff(data[:, [3,4]], axis=0), axis=1)
    dist_diffs = np.hstack([dist_diffs, np.nan])
    dist_diffs = dist_diffs[:, np.newaxis]

        # Neighbor speed
    nn = NearestNeighbors(n_neighbors=5).fit(data[:, [1, 2]])
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

def predictWithK(testFeatures, numVessels, trainFeatures=None, 
                 trainLabels=None):
    # Unsupervised prediction, so training data is unused
    scaler = MinMaxScaler()
    testFeatures = scaler.fit_transform(testFeatures)
    linkage_matrix = linkage(testFeatures, method="complete", metric="correlation")
    predVessels = fcluster(linkage_matrix, numVessels, criterion="maxclust")
    predict = pd.DataFrame(predVessels)
    num = predict.nunique()
    print(num)
    print(predVessels)
    return predVessels

    # scaler = StandardScaler()
    # testFeatures = scaler.fit_transform(testFeatures)
    # km = KMeans(n_clusters=numVessels, init='k-means++', n_init=10, 
    #             random_state=100)
    # predVessels = km.fit_predict(testFeatures)
    # return predVessels

def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    max_score = -math.inf
    
    for k in np.arange(2, 26, 1):
        labels = predictWithK(testFeatures, k)
        if np.unique(labels).size > 1:
            score = calinski_harabasz_score(testFeatures, labels)
            print(score)
            if score > max_score:
                max_score = score
                optimal_k = k
    print(max_score)
    print(optimal_k) 
    return predictWithK(testFeatures, optimal_k)

    
    return predictWithK(testFeatures, 8, trainFeatures, trainLabels)

# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    
    from utils import loadData, plotVesselTracks
    data = loadData('set2noVID.csv')
    features = data[:,2:]
    labels = data[:,1]
    features = preprocess_data(features)

    # Plot all vessel tracks with no coloring
    # plotVesselTracks(features[:,[2,1]])
    # plt.title('All vessel tracks')
    
    # Run prediction algorithms and check accuracy
    # Prediction with specified number of vessels
    numVessels = np.unique(labels).size

    # l_max = None
    # ari = -math.inf
    # for l in ("single", "complete", "average", "weighted", "centroid", "median", "ward"):
    #     predVesselsWithK = predictWithK(features, numVessels, l)
    #     ariWithK = adjusted_rand_score(labels, predVesselsWithK)
    #     if ariWithK > ari:
    #         ari = ariWithK
    #         l_max = l
    # print("ARI:", ari)
    # print("l final:", l_max)
    
    # predVesselsWithK = predictWithK(features, numVessels)
    # ariWithK = adjusted_rand_score(labels, predVesselsWithK)

    # Prediction without specified number of vessels
    predVesselsWithoutK = predictWithoutK(features)
    predNumVessels = np.unique(predVesselsWithoutK).size
    ariWithoutK = adjusted_rand_score(labels, predVesselsWithoutK)
    
    # print(f'Adjusted Rand index given K = {numVessels}: {ari}')
    print(f'Adjusted Rand index for estimated K = {predNumVessels}: '
          + f'{ariWithoutK}')

    # Plot vessel tracks colored by prediction and actual labels
    # plotVesselTracks(features[:,[2,1]], predVesselsWithK)
    # plt.title('Vessel tracks by cluster with K')
    # plotVesselTracks(features[:,[2,1]], predVesselsWithoutK)
    # plt.title('Vessel tracks by cluster without K')
    # plotVesselTracks(features[:,[2,1]], labels)
    # plt.title('Vessel tracks by label')
    # plt.show()
    