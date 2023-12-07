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
from sklearn.metrics.pairwise import haversine_distances
# from sklearn.neighbors import DistanceMetric
import matplotlib.pyplot as plt
import math

def predictWithK(testFeatures, numVessels, gamma, affinity, n_neighbors, degree, trainFeatures=None, 
                 trainLabels=None):
    # Unsupervised prediction, so training data is unused
    # scaler = StandardScaler()
    # testFeatures = scaler.fit_transform(testFeatures)
    # km = KMeans(n_clusters=numVessels, init='k-means++', n_init=10, 
    #             random_state=100)
    # predVessels = km.fit_predict(testFeatures)
    # return predVessels




    testFeatures1 = testFeatures.copy()

    # # Custom distance metric function for spatio-temporal clustering
    # def custom_distance(X, Y):
    #     spatial_distance = haversine_distances(X[:, :2], Y[:, :2]) * 6371000  # Convert to meters using Earth radius
    #     direction_velocity_distance = cdist(X[:, 2:4], Y[:, 2:4], metric='euclidean')
        
    #     time_distance = np.abs(X[:, 4] - Y[:, 4])  # Use absolute time difference
        
    #     # Combine spatial, direction/velocity, and time distances
    #     distance_matrix = spatial_distance + direction_velocity_distance + time_distance[:, np.newaxis]
        
    #     return distance_matrix

    # # Standardize the data
    # scaler = StandardScaler()
    # data_scaled = scaler.fit_transform(data)

    # # Apply DBSCAN with custom spatio-temporal distance metric
    # eps = 0.5  # The maximum distance between two samples for one to be considered as in the neighborhood of the other
    # min_samples = 5  # The number of samples (or total weight) in a neighborhood for a point to be considered as a core point
    # dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=custom_distance)
    # labels = dbscan.fit_predict(data_scaled)

    # # Add cluster labels to the original data
    # data_with_labels = np.column_stack((data, labels))

    # # Visualize the clusters (2D plot, you may need to adapt for 3D)
    # plt.scatter(data_with_labels[:, 1], data_with_labels[:, 0], c=data_with_labels[:, -1])
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.title('Spatio-Temporal Clusters')
    # plt.show()

    # return labels



    def preprocess_data(data):
        # Sort data based on timestamps
        # data = data[data[:, 2].argsort()]
        return data

    def construct_distance_matrix(data):
        num_points = len(data)
        distances = np.zeros((num_points, num_points))

        # data[:, 2] = data[:, 2] * 60.0 * np.cos(np.deg2rad(data[:, 1])) # x (longitude) in nautical miles
        # data[:, 1] = data[:, 1] * 60 # y (latitude) in nautical miles
        data[:, 2] = data[:, 2] * 60.0 # x (longitude) in nautical miles
        # data[:, 3] = data[:, 3] / 3600.0 # v in nautical miles per second
        # data[:, 4] = np.deg2rad(data[:, 4] / 10.0) # theta in radians

        data[:, 1] = data[:, 1] * 60.0 # y (latitude) in nautical miles
        data[:, 3] = data[:, 3] / 36000.0 # v in nautical miles per second
        data[:, 4] = np.deg2rad(data[:, 4] / 10.0) # theta in radians

        t, y, x, v, theta = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
        for i in range(num_points):
            t1, y1, x1, v1, theta1 = t[i], y[i], x[i], v[i], theta[i]
            time_diff = t - t1
            acc_approx = np.where(time_diff == 0, 0, (data[:, 3] - v1) / time_diff)
            x2_pred = x1 + np.sin(theta1) * (v1 * time_diff + 0.5 * acc_approx * time_diff**2)
            y2_pred = y1 + np.cos(theta1) * (v1 * time_diff + 0.5 * acc_approx * time_diff**2)

            distances[:, i] = np.sqrt((x2_pred - data[:, 2])**2 + (y2_pred - data[:, 1])**2)
        
        distances = np.triu(distances) + np.triu(distances, k=1).T

        # distances_flat = distances.flatten()
        # scaler = MinMaxScaler()
        # std_distances_flat = scaler.fit_transform(distances_flat.reshape(-1, 1))
        # std_distances = std_distances_flat.reshape(distances.shape)

        print("First 5")
        print(distances[:5])
        print("Last 5")
        print(distances[-5:])
        return distances



    testFeatures1 = preprocess_data(testFeatures1)
    # distance_matrix = construct_distance_matrix(testFeatures1)
    # print(np.shape(distance_matrix))
    # distance_matrix = squareform(distance_matrix)
    # print("distance matrix")
    # print(distance_matrix[-5:])
    # print(distance_matrix[:5])

    # linkage_matrix = linkage(distance_matrix, method=l)
    # predVessels = fcluster(linkage_matrix, numVessels, criterion="maxclust")

    cl = SpectralClustering(n_clusters=numVessels, random_state=42, n_init=10, affinity=affinity, gamma=gamma, n_neighbors=n_neighbors, degree=degree)
    return predVessels


    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(testFeatures)
    testFeatures = pd.DataFrame(testFeatures, columns=["SEQUENCE_DTTM", "LAT", "LON", "SPEED_OVER_GROUND", "COURSE_OVER_GROUND"])

    # testFeatures["TIME_INTERVAL"] = testFeatures["SEQUENCE_DTTM"].diff()
    # testFeatures.loc[0, "TIME_INTERVAL"] = 0
    # testFeatures["SPEED_X"] = testFeatures["SPEED_OVER_GROUND"] * np.sin(np.deg2rad(testFeatures["COURSE_OVER_GROUND"] / 10))
    # testFeatures["SPEED_Y"] = testFeatures["SPEED_OVER_GROUND"] * np.cos(np.deg2rad(testFeatures["COURSE_OVER_GROUND"] / 10))

    # print(testFeatures.loc[testFeatures["ACC_X"].isnull()])
    print(testFeatures)
    testFeatures = testFeatures.to_numpy()
    # scaler = StandardScaler()
    # testFeatures = scaler.fit_transform(testFeatures)
    # kmeans = KMeans(n_clusters=numVessels, init='k-means++', n_init=10, random_state=100)
    # predVessels = kmeans.fit_predict(testFeatures)
    model = AgglomerativeClustering(n_clusters=numVessels, linkage="ward")
    predVessels = model.fit_predict(testFeatures)

    return predVessels

def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    
    # Arbitrarily assume 20 vessels
    return predictWithK(testFeatures, 20, trainFeatures, trainLabels)

# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    
    from utils import loadData, plotVesselTracks
    data = loadData('set1.csv')
    features = data[:,2:]
    labels = data[:,1]

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

    # gamma, affinity, n_neighbors, degree
    g_max = None
    a_max = None
    n_max = None
    d_max = None
    ari=-math.inf
    for g in np.arange(0.01, 1, 0.05):
        for a in ('nearest_neighbors', 'rbf', 'poly', 'polynomial'):
            if a == 'nearest_neighbors':
                for n in np.arange(5, 50):
                    
    
    # predVesselsWithK = predictWithK(features, numVessels)
    # ariWithK = adjusted_rand_score(labels, predVesselsWithK)

    # Prediction without specified number of vessels
    # predVesselsWithoutK = predictWithoutK(features)
    # predNumVessels = np.unique(predVesselsWithoutK).size
    # ariWithoutK = adjusted_rand_score(labels, predVesselsWithoutK)
    
    print(f'Adjusted Rand index given K = {numVessels}: {ari}')
    # print(f'Adjusted Rand index for estimated K = {predNumVessels}: '
    #       + f'{ariWithoutK}')

    # Plot vessel tracks colored by prediction and actual labels
    plotVesselTracks(features[:,[2,1]], predVesselsWithK)
    plt.title('Vessel tracks by cluster with K')
    # plotVesselTracks(features[:,[2,1]], predVesselsWithoutK)
    # plt.title('Vessel tracks by cluster without K')
    plotVesselTracks(features[:,[2,1]], labels)
    plt.title('Vessel tracks by label')
    # plt.show()
    