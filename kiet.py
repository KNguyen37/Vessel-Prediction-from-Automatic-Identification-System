# -*- coding: utf-8 -*-
"""
Vessel prediction using k-means clustering on standardized features. If the
number of vessels is not specified, assume 20 vessels.

@author: Kevin S. Xu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
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
from sklearn.naive_bayes import LabelBinarizer

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense

import matplotlib.pyplot as plt
import math

def create_lstm_model(num_vessels):
        num_features = 5
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(None, num_features)))
        model.add(LSTM(32))
        model.add(Dense(num_vessels, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

def predictWithK(testFeatures, numVessels, trainFeatures=None, trainLabels=None):
    # Define number of features 
    num_features = 5
    # Reshape inputs
    print(testFeatures.shape)   
    test_sequences = np.reshape(testFeatures, (3323, 1, num_features))
        
    # Create one-hot encoded outputs
    encoder = LabelBinarizer()
    train_targets = np.zeros(len(test_sequences))
    #test_sequences = test_sequences[:8]
    #train_targets = encoder.fit_transform(np.arange(numVessels))

    # Create and train LSTM model
    lstm_model = create_lstm_model(numVessels) 
    lstm_model.fit(test_sequences, train_targets)

    # Make predictions on test set
    preds = lstm_model.predict(test_sequences)
    pred_ids = np.argmax(preds, axis=-1)

    return encoder.inverse_transform(pred_ids)

def predict_vessels(test_features, test_labels, train_features, train_labels):

    # Standardize features
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    # Encode labels 
    encoder = LabelEncoder()
    train_encoded = encoder.fit_transform(train_labels)
    test_encoded = encoder.transform(test_labels)

    # Create and fit model
    num_features = train_scaled.shape[1]
    num_vessels = len(encoder.classes_)
    model = create_lstm_model(num_features, num_vessels)
    model.fit(train_scaled, train_encoded, epochs=10)

    # Make predictions
    test_seqs = test_scaled.reshape(-1, 1, num_features)
    preds = model.predict_classes(test_seqs)
    
    return encoder.inverse_transform(preds)

def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    
    # Arbitrarily assume 20 vessels
    return predictWithK(testFeatures, 20, trainFeatures, trainLabels)

# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    
    from utils import loadData, plotVesselTracks
    data = loadData('set2.csv')
    features = data[:,2:]
    labels = data[:,1]

    # Plot all vessel tracks with no coloring
    # plotVesselTracks(features[:,[2,1]])
    # plt.title('All vessel tracks')
    
    # Run prediction algorithms and check accuracy
    # Prediction with specified number of vessels
    numVessels = np.unique(labels).size
    num_features = 5
    
    # LSTM autoencoder
    input = keras.Input(shape=(None, num_features))
    encoded = keras.layers.LSTM(64)(input)
    decoded = keras.layers.RepeatVector(num_features)(encoded)
    decoded = keras.layers.LSTM(64, return_sequences=True)(decoded)

    model = keras.Model(input, decoded)
    model.compile(loss='mse', optimizer='adam')

    # Generate sequences from features
    sequence_len = 10  
    sequences = generate_sequences(features, sequence_len)

    # Train reconstruction
    model.fit(sequences)  

    # Get vessel embeddings  
    encoder = keras.Model(input, encoded)
    embeddings = encoder.predict(sequences)

    # Cluster embeddings
    num_clusters = 20
    clusters = KMeans(n_clusters=num_clusters).fit_predict(embeddings)

    # predVesselsWithK = predictWithK(features, numVessels)
    # ariWithK = adjusted_rand_score(labels, predVesselsWithK)

    # #Prediction without specified number of vessels
    # predVesselsWithoutK = predictWithoutK(features)
    # predNumVessels = np.unique(predVesselsWithoutK).size
    # ariWithoutK = adjusted_rand_score(labels, predVesselsWithoutK)
    
    # print(f'Adjusted Rand index given K = {numVessels}: {ari}')
    # print(f'Adjusted Rand index for estimated K = {predNumVessels}: '
    #       + f'{ariWithoutK}')

    # # Plot vessel tracks colored by prediction and actual labels
    # plotVesselTracks(features[:,[2,1]], predVesselsWithK)
    # plt.title('Vessel tracks by cluster with K')
    # plotVesselTracks(features[:,[2,1]], predVesselsWithoutK)
    # plt.title('Vessel tracks by cluster without K')
    # plotVesselTracks(features[:,[2,1]], labels)
    # plt.title('Vessel tracks by label')
    # plt.show()
    