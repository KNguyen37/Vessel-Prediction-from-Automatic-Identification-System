import numpy as np
from scipy.spatial import distance

class KalmanVesselTracker:
    def __init__(self):
        # State vector [lat, lon, lat_vel, lon_vel]
        self.state = [0, 0, 0, 0]  
        
        # Uncertainty matrix
        self.covariance = np.eye(4)  
        
        # Observation matrix
        self.observation_model = np.array([[1, 0, 0, 0], 
                                           [0, 1, 0, 0]])
        
        # Motion model 
        self.motion_model = np.eye(4)
        dt = 1 # time step  
        self.motion_model[0,2] = dt
        self.motion_model[1,3] = dt

    def predict(self):
        
        # Predict to get priori estimates
        prior_state = np.dot(self.motion_model, self.state)
        prior_covariance = np.dot(self.motion_model, np.dot(self.covariance, self.motion_model.T)) + np.eye(4)
        
        return prior_state, prior_covariance
    
    def correct(self, measurement): 
        
        # Compute Kalman gain 
        innovation = measurement - np.dot(self.observation_model, self.state)
        innovation_covariance = np.dot(self.observation_model, np.dot(self.covariance, self.observation_model.T))
        kalman_gain = np.dot(self.covariance, np.dot(self.observation_model.T, np.linalg.inv(innovation_covariance)))
        
        # Correct state 
        new_state = self.state + np.dot(kalman_gain, innovation)
        new_covariance = self.covariance - np.dot(kalman_gain, np.dot(self.observation_model, self.covariance))
        
        return new_state, new_covariance