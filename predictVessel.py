# -*- coding: utf-8 -*-
"""
Vessel prediction using k-means clustering on standardized features. If the
number of vessels is not specified, assume 20 vessels.

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler
import math

def predictWithK(testFeatures, numVessels, s, trainFeatures=None, 
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
        data = data[data[:, 2].argsort()]
        return data

    def construct_affinity_matrix(data, s):
        num_points = len(data)
        distances = np.zeros((num_points, num_points))

        # for i in range(num_points):
        #     t1, x1, y1, v1, theta1 = unique_data[i, :]

        #     later_time_indices = np.where(unique_data[:, 0] > t1)[0]
        #     later_data = unique_data[later_time_indices, :]
        #     time_diff = later_data[:, 0] - t1

        #     acc_approx = np.where(time_diff != 0, (later_data[:, 3] - v1) / time_diff, 0)
        #     x2_pred = x1 + np.cos(theta1) * (v1 * time_diff + 0.5 * acc_approx * time_diff**2)
        #     y2_pred = y1 + np.sin(theta1) * (v1 * time_diff + 0.5 * acc_approx * time_diff**2)

        #     distances = np.sqrt((x2_pred - later_data[:, 1])**2 + (y2_pred - later_data[:, 2])**2)
        #     affinity_matrix[i, later_time_indices] = rbf_kernel(distances.reshape(1, -1))

        #     # Update the affinity matrix for the current point
        #     # affinity_matrix[i, :] = similarities

        # affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)

        t, y, x, v, theta = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
        for i in range(num_points):
            t1, y1, x1, v1, theta1 = t[i], y[i], x[i], v[i], theta[i]
            later_time_indices = np.where(t > t1)[0]
            later_data = data[later_time_indices]
            time_diff = later_data[:, 0] - t1
            acc_approx = np.where(time_diff != 0, (later_data[:, 3] - v1) / time_diff, 0)
            x2_pred = x1 + np.sin(theta1) * (v1 * time_diff + 0.5 * acc_approx * time_diff**2)
            y2_pred = y1 + np.cos(theta1) * (v1 * time_diff + 0.5 * acc_approx * time_diff**2)
            distances[i, later_time_indices] = np.sqrt((x2_pred - later_data[:, 2])**2 + (y2_pred - later_data[:, 1])**2)
        distances = 0.5 * (distances + distances.T)
        print(distances)

        # for i in range(num_points):
        #     for j in range(i + 1, num_points):
        #         t1, y1, x1, v1, theta1 = data[i,:]
        #         t2, y2, x2, v2, theta2 = data[j,:]

        #         time_diff = t2 - t1
        #         if time_diff == 0:
        #             x2_pred = x1
        #             y2_pred = y1
        #         else:
        #             acc_approx = (v2 - v1) / (time_diff)
        #             x2_pred = x1 + math.cos(theta1) * (v1 * (time_diff) + 0.5 * acc_approx * math.pow(time_diff, 2))
        #             y2_pred = y1 + math.sin(theta1) * (v1 * (time_diff) + 0.5 * acc_approx * math.pow(time_diff, 2))

        #         distance = np.sqrt((x2_pred - x2)**2 + (y2_pred - y2)**2)

        #         # Construct affinity graph
        #         distances[i, j] = distances[j, i] = distance
        
        # affinity_matrix = rbf_kernel(distances, gamma=1.0 / (2.0 * 1.0**2))
        # print(affinity_matrix)

        # min_val = np.min(distances)
        # max_val = np.max(distances)
        # scaled_distances = (distances - min_val) / (max_val - min_val)
        # affinity_matrix = np.ones_like(scaled_distances) - scaled_distances
        # squared_distances = distances ** 2
        # affinity_matrix = np.exp(-squared_distances / (2 * s**2))

        distances[distances == 0] = 1
        affinity_matrix = np.minimum(1 / distances, 1)
        affinity_matrix /= affinity_matrix.sum(axis=1, keepdims=True)
        affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)
        print(affinity_matrix)
        return affinity_matrix


    preprocessedTestFeatures = preprocess_data(testFeatures)
    # Assuming testFeatures is the preprocessed unique data
    affinity_matrix = construct_affinity_matrix(preprocessedTestFeatures, s=s)

    model = SpectralClustering(n_clusters=numVessels, affinity="precomputed")
    pred_labels = model.fit_predict(affinity_matrix)

    return pred_labels

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
    for s in np.arange(0.001, 1, 0.001):
        numVessels = np.unique(labels).size
        predVesselsWithK = predictWithK(features, numVessels, s)
        ariWithK = adjusted_rand_score(labels, predVesselsWithK)
        if ariWithK > ari:
            ariWithK = ari
            s_final = s
    print("ARI:", ari)
    print("s final:", s_final)
    
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
    