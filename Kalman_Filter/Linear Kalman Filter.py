### import modules ###
import numpy as np
import matplotlib.pyplot as plt
from control import LKF,obs_generation

### Modeling ###
T = 100
F = np.array([[1, 0, 0.5, 0], [0, 1, 0, 0.5], [0, 0, 1, 0], [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
P = np.eye(4)
Q = np.eye(4)
R = 100*np.eye(2)
noise_std = 0.3

### Initially Given Information ###
anchors = np.array([[0, 0], [10, 10], [0, 10]])
gt = np.array([[3],[0],[1],[0]])
x_initial = np.array([[3],[0],[1],[0]])
obs_initial = np.array([[3],[0]])
ground_truth = np.hstack((x_initial,np.zeros((4,T-1))))
observation = np.hstack((obs_initial,np.zeros((2,T-1))))
estimation = np.hstack((x_initial,np.zeros((4,T-1))))

### Making Instance ###
obs = obs_generation(anchors,noise_std)
LKF = LKF(F,H,Q,R)

### Get observation and Ground truth###
for i in range(1,T):
    ground_truth[:, i] = F @ ground_truth[:, i-1] + np.random.randn(4)
    observation[:, i:i+1] = obs.trilateration(ground_truth[0:2, i].reshape((2,1)))
    
### Executing Filtering ###
for i in range(1,T):
    LKF.prediction(estimation[:,i-1],P)
    estimation[:,i],P = LKF.correction(LKF.pred_x,LKF.pred_P,observation[:,i])
    
### Plot the result ###
plt.figure
plt.plot(ground_truth[0,:],ground_truth[1,:], '-o', color='black', markerfacecolor='none', markeredgecolor='black')
plt.plot(observation[0, :], observation[1, :], "-x",color="red")
plt.plot(estimation[0, :], estimation[1, :], "-x", color="blue")
plt.legend(["GroundTruth", "Observation", "Linear Kalman Filter"])
plt.title("Linear Kalman Filter")
plt.show()
