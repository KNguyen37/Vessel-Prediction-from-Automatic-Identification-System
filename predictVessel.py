# -*- coding: utf-8 -*-
"""
Vessel prediction using k-means clustering on standardized features. If the
number of vessels is not specified, assume 20 vessels.

@author: Kevin S. Xu
"""
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense

import random
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.naive_bayes import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

random.seed(100)

# def create_lstm_model(num_vessels):
#     model = Sequential()
#     model.add(LSTM(64, return_sequences=True, input_shape=(None, num_features)))
#     model.add(LSTM(32))
#     model.add(Dense(num_vessels, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam')
#     return model

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
    nn = NearestNeighbors(5).fit(data[:, [3,4]])
    neighbor_indices = nn.kneighbors(data[:, [3,4]], return_distance=False)[:,1:]
    neighbor_speed = data[neighbor_indices, 5].mean(axis=1)[:, np.newaxis]
    neighbor_course = data[neighbor_indices, 6].mean(axis=1)[:, np.newaxis]
    
    cog = data[:, -1] # course over ground
    # Difference in COG between reports
    cog_diff = np.abs(np.diff(cog))
    cog_diff = np.hstack([cog_diff, np.zeros(1)])[:,None]

    # Cyclic encoding 
    cog_sin = np.sin(2*np.pi*cog/360)[:,None]
    cog_cos = np.cos(2*np.pi*cog/360)[:,None]
    
    # Directional velocity
    velocity = data[:, 5][:,None] # speed over ground
    dir_velocity = velocity * cog_cos
    
    # # Append features
    data_ext = np.hstack([data, neighbor_speed, dir_velocity, cog_sin, cog_cos, cog_diff])
    return data_ext
    
def predictWithK(testFeatures, numVessels, trainFeatures = None, 
                 trainLabels = None):
    # Unsupervised prediction, so training data is unused
    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(testFeatures)
    imputer = SimpleImputer(strategy='mean')
    testFeatures_imputed = imputer.fit_transform(testFeatures)
    km = KMeans(n_clusters=numVessels, init='k-means++', n_init=10, 
                random_state=100)
    predVessels = km.fit_predict(testFeatures_imputed)
    return predVessels

def predictWithoutK(testFeatures, trainFeatures, trainLabels):
    # Try different K values from 2 to 20
    k_range = range(4, 15)  
    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(testFeatures)
    imputer = SimpleImputer(strategy='mean')
    testFeatures_imputed = imputer.fit_transform(testFeatures)
    elbow_k = find_elbow_k(testFeatures_imputed, k_range) 
    
    # Predict labels using the elbow K  
    return predictWithK(testFeatures, elbow_k)
    
def find_elbow_k(train_features, k_range):
    sil_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k)  
        kmeans.fit(train_features)

        sil_score = silhouette_score(train_features, kmeans.labels_)
        sil_scores.append(sil_score)
    # Find index of max difference 
    max_diff_idx = find_max_diff_index(sil_scores)
    return k_range[max_diff_idx]

def find_max_diff_index(scores):
    score_diffs = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
    max_ix = score_diffs.index(max(score_diffs))
    return max_ix

# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    from utils import loadData, plotVesselTracks
    data = loadData('set2.csv')
    data = preprocess_data(data)
    features = data[:,2:]
    labels = data[:,1]
    
    
    #%% Plot all vessel tracks with no coloring
    plotVesselTracks(features[:,[2,1]])
    plt.title('All vessel tracks')
    
    #%% Run prediction algorithms and check accuracy
    
    #Prediction with specified number of vessels
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

    #%% Plot vessel tracks colored by prediction and actual labels
    plotVesselTracks(features[:,[2,1]], predVesselsWithK)
    plt.title('Vessel tracks by cluster with K')
    plotVesselTracks(features[:,[2,1]], predVesselsWithoutK)
    plt.title('Vessel tracks by cluster without K')
    plotVesselTracks(features[:,[2,1]], labels)
    plt.title('Vessel tracks by label')
    plt.show()
# %%
