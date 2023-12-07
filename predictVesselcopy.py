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
from sklearn.naive_bayes import LabelBinarizer

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense

import matplotlib.pyplot as plt
import math

def create_lstm_model(num_vessels):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(None, num_features)))
        model.add(LSTM(32))
        model.add(Dense(num_vessels, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

def predictWithK(testFeatures, numVessels, gamma, affinity, n_neighbors, degree, trainFeatures=None, 
                 trainLabels=None):
    # Unsupervised prediction, so training data is unused

        # Define number of features 
        num_features = 7
        
        # Reshape inputs   
        test_sequences = np.reshape(testFeatures, (len(testFeatures), 1, num_features))
        
        # Create one-hot encoded outputs
        encoder = LabelBinarizer()
        train_targets = encoder.fit_transform(np.arange(numVessels))

        # Create and train LSTM model
        lstm_model = create_lstm_model(numVessels) 
        lstm_model.fit(test_sequences, train_targets)

        # Make predictions on test set
        preds = lstm_model.predict(test_sequences)
        pred_ids = np.argmax(preds, axis=-1)

        return encoder.inverse_transform(pred_ids)

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
    for g in np.arange(0.05, 1.05, 0.05):
        for a in ('nearest_neighbors', 'rbf', 'poly', 'polynomial', 'additive_chi2', 'linear', 'chi2', 'laplacian', 'sigmoid', 'cosine'):
            if a == 'nearest_neighbors':
                for n in np.arange(5, 31, 1):
                    predVesselsWithK = predictWithK(features, numVessels, g, a, n, 1)
                    ariWithK = adjusted_rand_score(labels, predVesselsWithK)
                    if ariWithK > ari:
                        ari = ariWithK
                        g_max = g
                        a_max = a
                        n_max = n
            elif a == 'polynomial':
                for d in np.arange(2, 11, 1):
                    predVesselsWithK = predictWithK(features, numVessels, g, a, 1, d)
                    ariWithK = adjusted_rand_score(labels, predVesselsWithK)
                    if ariWithK > ari:
                        ari = ariWithK
                        g_max = g
                        a_max = a
                        d_max = d
            else:
                predVesselsWithK = predictWithK(features, numVessels, g, a, 1, 1)
                ariWithK = adjusted_rand_score(labels, predVesselsWithK)
                if ariWithK > ari:
                    ari = ariWithK
                    g_max = g
                    a_max = a

    print("ari", ari)
    print("g_max", g_max)
    print("a_max", a_max)
    print("n_max", n_max)
    print("d_max", d_max)

    
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
    