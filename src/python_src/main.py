import numpy as np
import matplotlib.pyplot as plt
from ekf import EKF

plt.figure()

q0 = np.array([1,0,0,0]).reshape(-1,1)
v0 = np.array([0,0,0]).reshape(-1,1)
p0 = np.array([5,5,5]).reshape(-1,1)
ekf = EKF(q0, v0, p0)




DT = 0.1
NUM_STEPS = 1000

gps_meas=np.array([5,5,5]).reshape(-1,1)
steps_per_meas = 20

mus = []
covs = []

for step in range(NUM_STEPS):
    mus.append(ekf.mean)
    covs.append(ekf.cov)
    
    ekf.prediction(DT)
    
    if (step != 0) and (step % steps_per_meas == 0):
        ekf.correction(DT, gps=gps_meas)    


plt.subplot(2, 1, 1)
plt.title('Position X')
plt.plot([mu[7] for mu in mus], 'r')
plt.plot([mu[7] + 2 * np.sqrt(cov[7,7]) for mu,cov in zip(mus, covs)], 'r--') # +2 std
plt.plot([mu[7] - 2 * np.sqrt(cov[7,7]) for mu,cov in zip(mus, covs)], 'r--') # -2 std

plt.subplot(2, 1, 2)
plt.title('Velocity X')
plt.plot([mu[4] for mu in mus], 'r')
plt.plot([mu[4] + 2 * np.sqrt(cov[4,4]) for mu,cov in zip(mus, covs)], 'r--') # +2 std
plt.plot([mu[4] - 2 * np.sqrt(cov[4,4]) for mu,cov in zip(mus, covs)], 'r--') # -2 std

plt.tight_layout()
plt.show()
