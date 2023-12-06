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
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import haversine_distances, euclidean_distances
from scipy.spatial.distance import euclidean, squareform
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler
import math

def predictWithK(testFeatures, numVessels, trainFeatures=None, 
                 trainLabels=None):
    # Unsupervised prediction, so training data is unused
    # scaler = StandardScaler()
    # testFeatures = scaler.fit_transform(testFeatures)
    # km = KMeans(n_clusters=numVessels, init='k-means++', n_init=10, 
    #             random_state=100)
    # predVessels = km.fit_predict(testFeatures)
    # return predVessels




    def preprocess_data(data):
        # Sort data based on timestamps
        # data = data[data[:, 2].argsort()]
        return data

    def construct_distance_matrix(data):
        num_points = len(data)
        distances = np.zeros((num_points, num_points))

        t, y, x, v, theta = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
        for i in range(num_points):
            t1, y1, x1, v1, theta1 = t[i], y[i], x[i], v[i], theta[i]
            later_time_indices = np.where(t > t1)[0]
            later_data = data[later_time_indices]
            time_diff = later_data[:, 0] - t1
            acc_approx = np.where(time_diff != 0, (later_data[:, 3] - v1) / time_diff, 0)
            x2_pred = x1 + np.sin(np.deg2rad(theta1 / 10)) * (v1 * time_diff + 0.5 * acc_approx * time_diff**2)
            y2_pred = y1 + np.cos(np.deg2rad(theta1 / 10)) * (v1 * time_diff + 0.5 * acc_approx * time_diff**2)
            distances[i, later_time_indices] = np.sqrt((x2_pred - later_data[:, 2])**2 + (y2_pred - later_data[:, 1])**2)
        distances = distances + distances.T
        print(distances)
        return distances

    #     distances[distances == 0] = 1
    #     affinity_matrix = np.minimum(1 / distances, 1)
    #     affinity_matrix /= affinity_matrix.sum(axis=1, keepdims=True)
    #     affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)
    #     print(affinity_matrix)
    #     return affinity_matrix



    preprocessedTestFeatures = preprocess_data(testFeatures)
    distance_matrix = construct_distance_matrix(preprocessedTestFeatures)
    print(np.shape(distance_matrix))
    # distance_matrix = squareform(distance_matrix)
    
    linkage_matrix = linkage(distance_matrix, method="single")
    predVessels = fcluster(linkage_matrix, numVessels, criterion="maxclust")
    return predVessels

    # model = SpectralClustering(n_clusters=numVessels, affinity="precomputed")
    # predVessels = model.fit_predict(affinity_matrix)


    testFeatures = pd.DataFrame(testFeatures, columns=["SEQUENCE_DTTM", "LAT", "LON", "SPEED_OVER_GROUND", "COURSE_OVER_GROUND"])

    # testFeatures["TIME_INTERVAL"] = testFeatures["SEQUENCE_DTTM"].diff()
    # testFeatures.loc[0, "TIME_INTERVAL"] = 0
    testFeatures["SPEED_X"] = testFeatures["SPEED_OVER_GROUND"] * np.sin(testFeatures["COURSE_OVER_GROUND"] / 10)
    testFeatures["SPEED_Y"] = testFeatures["SPEED_OVER_GROUND"] * np.cos(testFeatures["COURSE_OVER_GROUND"] / 10)
    testFeatures["ACC_X"] = testFeatures["SPEED_X"] / testFeatures["TIME_INTERVAL"]
    testFeatures.loc[0, "ACC_X"] = 0
    testFeatures["ACC_Y"] = testFeatures["SPEED_Y"] / testFeatures["TIME_INTERVAL"]
    testFeatures.loc[0, "ACC_Y"] = 0

    distances = np.zeros(len(testFeatures, testFeatures))
    for i in range(len(testFeatures)):
        for j in range(len(testFeatures)):
            spatial_distance = haversine_distances(np.radians([[testFeatures.loc[i, "LAT"], testFeatures.loc[i, "LON"]]]),
                                                   np.radians([[testFeatures.loc[j, "LAT"], testFeatures.loc[j, "LON"]]])).item()
            temporal_distance = euclidean(testFeatures[i], testFeatures[j])
            distances[i, j] = np.sqrt(spatial_distance ** 2 + temporal_distance ** 2)



    print(testFeatures.loc[testFeatures["ACC_X"].isnull()])
    print(testFeatures)
    testFeatures = testFeatures.to_numpy()
    # scaler = StandardScaler()
    # testFeatures = scaler.fit_transform(testFeatures)
    kmeans = KMeans(n_clusters=numVessels, init='k-means++', n_init=10, random_state=100)
    predVessels = kmeans.fit_predict(testFeatures)
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
    plotVesselTracks(features[:,[2,1]])
    plt.title('All vessel tracks')
    
    # Run prediction algorithms and check accuracy

    ari = -math.inf
    s_final = None
    # Prediction with specified number of vessels
    numVessels = np.unique(labels).size
    predVesselsWithK = predictWithK(features, numVessels)
    ariWithK = adjusted_rand_score(labels, predVesselsWithK)
    # if ariWithK > ari:
    #     ariWithK = ari
    #     s_final = s
    # print("ARI:", ari)
    # print("s final:", s_final)
    
    # Prediction without specified number of vessels
    predVesselsWithoutK = predictWithoutK(features)
    predNumVessels = np.unique(predVesselsWithoutK).size
    ariWithoutK = adjusted_rand_score(labels, predVesselsWithoutK)
    
    print(f'Adjusted Rand index given K = {numVessels}: {ariWithK}')
    print(f'Adjusted Rand index for estimated K = {predNumVessels}: '
          + f'{ariWithoutK}')

    # Plot vessel tracks colored by prediction and actual labels
    plotVesselTracks(features[:,[2,1]], predVesselsWithK)
    plt.title('Vessel tracks by cluster with K')
    plotVesselTracks(features[:,[2,1]], predVesselsWithoutK)
    plt.title('Vessel tracks by cluster without K')
    plotVesselTracks(features[:,[2,1]], labels)
    plt.title('Vessel tracks by label')
    