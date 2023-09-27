import numpy as np


class obs_generation():
    
    def __init__(self, an_pos : np.ndarray, noise_std):
        self.noise_std = noise_std
        self.an_pos1 = an_pos[0].reshape((2,1))
        self.an_pos2 = an_pos[1].reshape((2,1))
        self.an_pos3 = an_pos[2].reshape((2,1))

        
    def trilateration(self,gt):
        A = 2*(self.an_pos2[0] - self.an_pos1[0])
        B = 2*(self.an_pos2[1] - self.an_pos1[1])
        D = 2*(self.an_pos3[0] - self.an_pos2[0])
        E = 2*(self.an_pos3[1] - self.an_pos2[1])
        tri1 = np.array([[A, B],[D, E]]).reshape((2,2))

        r1 = np.linalg.norm(gt - self.an_pos1) + self.noise_std * np.random.randn(1)
        r2 = np.linalg.norm(gt - self.an_pos2) + self.noise_std * np.random.randn(1)
        r3 = np.linalg.norm(gt - self.an_pos3) + self.noise_std * np.random.randn(1)
        C = r1**2 - r2**2 + self.an_pos2[0]**2 - self.an_pos1[0]**2 + self.an_pos2[1]**2 - self.an_pos1[1]**2
        F = r2**2 - r3**2 + self.an_pos3[0]**2 - self.an_pos2[0]**2 + self.an_pos3[1]**2 - self.an_pos2[1]**2
        tri2 = np.array([[C],[F]]).reshape((2,1))
        obs = np.linalg.inv(tri1) @ tri2
        return obs




class LKF():
    
    def __init__(self, F, H, Q, R):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
    
    def prediction(self, prev_x, prev_P):
        self.pred_x = self.F @ prev_x
        self.pred_P = self.F @ prev_P @ self.F.T + self.Q
        return self.pred_x, self.pred_P
        
    def correction(self, pred_x, pred_P, obs):
        KG = (pred_P @ self.H.T) @ (np.linalg.inv((self.H @ pred_P @ self.H.T + self.R)))
        est_x = pred_x + KG @ (obs - self.H @ pred_x)
        est_P = pred_P - KG @ self.H @ pred_P
        return est_x, est_P
