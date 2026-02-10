import numpy as np
from src.core.objects import BoundingBox3D

class KalmanBoxTracker:
    """
    A Constant Velocity (CV) Kalman Filter for 3D Bounding Boxes.
    State Vector (8x1): [x, y, z, heading, vx, vy, vz, v_heading]
    """
    def __init__(self, box: BoundingBox3D):
        # initialize state vector [x, y, z, h, 0, 0, 0, 0]
        self.x = np.zeros((8, 1))
        self.x[0] = box.x
        self.x[1] = box.y
        self.x[2] = box.z
        self.x[3] = box.heading
        
        # state transition matrix (F)
        # predict next state: pos = pos + vel * dt (dt=1)
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i+4] = 1.0 # x += vx, y += vy
            
        # Measurments matrix (H)
        # We observe [x, y, z, h], mapping them to the first 4 state variables
        self.H = np.eye(4, 8)
        
        # Covariance Matrix (P) - Uncertainity measurment
        # High uncertainty for velocity initially
        self.P = np.eye(8) * 10
        self.P[4:, 4:] *= 1000.0
        
        # Measurement Noise (R) - Trust in sensor/annotations
        # Low value = we trust the annotation coordinates
        self.R = np.eye(4) * 0.1
        
        # Process Noise (Q) - Uncertainty in the model
        self.Q = np.eye(8) * 0.01
        
    def update(self, box: BoundingBox3D):
        """
        Correction Step: Update state with a ground-truth measurement.
        """
        # Measurement Vector
        z = np.array([[box.x], [box.y], [box.z], [box.heading]])
        
        # Innovation (Residual): y = z - Hx
        y = z - (self.H @ self.x)
        
        # Fix Cyclic Heading Error (-pi to pi)
        # If prediction is 179° and measurement is -179°, difference should be 2°, not 358°
        while y[3, 0] > np.pi: 
            y[3, 0] -= 2 * np.pi
        while y[3, 0] < -np.pi: 
            y[3, 0] += 2 * np.pi
            
        # Kalman Gain: K = PH' * inv(HPH' + R)
        S = (self.H @ self.P @ self.H.T) + self.R
        
        try:
            K = (self.P @ self.H.T) @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular (rare)
            K = np.zeros((8, 4))
        
        # update state: x = x + Ky
        self.x = self.x + (K @ y)
        
        # update covariance: P = (I - KH)P
        identity = np.eye(8)
        self.P = (identity - (K @ self.H)) @ self.P
        
    def predict(self) -> BoundingBox3D:
        """
        Prediction Step: Project state forward by one step.
        Returns a 'Predicted' BoundingBox3D.
        """
        # predict state: x = Fx
        self.x = self.F @ self.x
        
        # predict covariance: P = FPF' + Q
        self.P = (self.F @ self.P @ self.F.T) + self.Q
        
        # Normalize heading in state
        while self.x[3, 0] > np.pi: 
            self.x[3, 0] -= 2 * np.pi
        while self.x[3, 0] < -np.pi: 
            self.x[3, 0] += 2 * np.pi
          
        # Return as Object
        return BoundingBox3D(
            track_id=-1, # Placeholder
            label="predicted",
            x=float(self.x[0, 0]),
            y=float(self.x[1, 0]),
            z=float(self.x[2, 0]),
            dx=0, 
            dy=0, 
            dz=0, # Dimensions are not tracked
            heading=float(self.x[3, 0])
        )