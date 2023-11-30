# -*- coding: utf-8 -*-
"""
Vessel prediction using k-means clustering on standardized features. If the
number of vessels is not specified, assume 20 vessels.

@author: Kevin S. Xu
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import adjusted_rand_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_kernels
import math

random.seed(100)

def predictWithK(testFeatures, numVessels, trainFeatures=None, 
                 trainLabels=None):
    # Unsupervised prediction, so training data is unused
    
    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(testFeatures)
    # km = KMeans(n_clusters=numVessels, init='k-means++', n_init=10, 
    #             random_state=100)
    # predVessels = km.fit_predict(testFeatures)

    def preprocess_data(data):
        # Sort data based on timestamps
        data = data[data[:, 2].argsort()]

        # Remove duplicate data with extremely close coordinates and courses over ground at the same time
        unique_data = []
        for i in range(len(data) - 1):
            if not np.allclose(data[i, 3:], data[i + 1, 3:], atol=1e-6):
                unique_data.append(data[i])
        unique_data.append(data[-1])
        unique_data = np.array(unique_data)

        return unique_data

    def construct_affinity_graph(unique_data):
        num_points = len(unique_data)
        affinity_matrix = np.zeros((num_points, num_points))

        for i in range(num_points):
            for j in range(i + 1, num_points):
                t1, x1, y1, v1, theta1 = unique_data[i, :5]
                t2, x2, y2, v2, theta2 = unique_data[j, :5]

                x2_pred = x1 + v1 * math.cos(theta1) * (t2 - t1)
                y2_pred = y2 + v1 * math.sin(theta1) * (t2 - t1)

                # Calculate similarity using Gaussian similarity function
                similarity = pairwise_kernels([[x2_pred, y2_pred]], [[x2, y2]], metric='rbf')[0, 0]

                # Calculate tolerance proportional to distance
                # distance = euclidean_distances([[x1, y1]], [[x2, y2]])[0, 0]
                # tolerance = 1.0 / (1.0 + distance)

                # Construct affinity graph
                affinity_matrix[i, j] = affinity_matrix[j, i] = similarity

        return affinity_matrix


    # Assuming testFeatures is the preprocessed unique data
    affinity_matrix = construct_affinity_graph(testFeatures)

    # Perform clustering using some clustering algorithm (e.g., KMeans)
    # You may need to install scikit-learn if you haven't already: pip install scikit-learn
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=numVessels)
    pred_labels = kmeans.fit_predict(affinity_matrix)

    return pred_labels

def predictWithoutK(testFeatures, trainFeatures, trainLabels):
    # Try range of K values
    # Define ARI scoring
    ari_scorer = make_scorer(adjusted_rand_score)

    # Gridsearch cross-validation 
    kmeans = KMeans()
    param_grid = {'n_clusters': range(2, 31)}
    grid = GridSearchCV(kmeans, param_grid, cv=5, scoring=ari_scorer)

    grid.fit(trainFeatures, trainLabels)
    best_km = grid.best_estimator_
    best_num_clusters = best_km.n_clusters
    
    return predictWithK(testFeatures, best_num_clusters)

    # # Arbitrarily assume 20 vessels
    # return predictWithK(testFeatures, 20, trainFeatures, trainLabels)

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

    # Prediction with specified number of vessels
    numVessels = np.unique(labels).size
    predVesselsWithK = predictWithK(features, numVessels)
    ariWithK = adjusted_rand_score(labels, predVesselsWithK)
    
    # Prediction without specified number of vessels
    predVesselsWithoutK = predictWithoutK(features, features, labels)
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
    # plt.show()