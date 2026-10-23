import numpy as np
import matplotlib.pyplot as plt
from multiplicative_ekf import MEKF

plt.figure()

q0 = np.array([1, 0, 0, 0]).reshape(-1, 1)
v0 = np.array([0, 0, 0]).reshape(-1, 1)
p0 = np.array([5, 5, 5]).reshape(-1, 1)

# Covariances for MEKF (gyro, accel, gyro_bias, accel_bias)
gyro_cov = 0.0
accel_cov = 0.0
gyro_bias_cov = 0.0
accel_bias_cov = 0.0

# GPS measurement covariance (much smaller relative to IMU noise)
gps_cov = 0.001

# Initialize P with 1.0 for velocity and position, and 0.0 for orientation and biases
# This perfectly mirrors the 10-state EKF (which has no bias states and no initial orientation-velocity coupling)
P_initial = np.zeros((15, 15), dtype=float)
np.fill_diagonal(P_initial[3:9, 3:9], 1.0) # velocity and position variance = 1.0

mekf = MEKF(q0, v0, p0, gyro_cov, accel_cov, gyro_bias_cov, accel_bias_cov, gps_cov=gps_cov, P_initial=P_initial)

DT = 0.1
NUM_STEPS = 1000

gps_meas = np.array([5, 5, 5]).reshape(-1, 1)
steps_per_meas = 20

mus = []
covs = []

for step in range(NUM_STEPS):
    mus.append(mekf.mean)
    covs.append(mekf.cov)
    
    mekf.prediction(DT)
    
    if (step != 0) and (step % steps_per_meas == 0):
        mekf.correction(DT, gps=gps_meas)    

# mekf.mean returns: [roll, pitch, yaw, vx, vy, vz, px, py, pz, gyr_bx, gyr_by, gyr_bz, acc_bx, acc_by, acc_bz]
# mekf.cov (_P) is 15x15 corresponding to error states:
# - Indices 0..2: orientation error
# - Indices 3..5: velocity error (vx, vy, vz) -> Velocity X is index 3
# - Indices 6..8: position error (px, py, pz) -> Position X is index 6

plt.subplot(2, 1, 1)
plt.title('Position X')
plt.plot([mu[6] for mu in mus], 'r')
plt.plot([mu[6] + 2 * np.sqrt(cov[6, 6]) for mu, cov in zip(mus, covs)], 'r--') # +2 std
plt.plot([mu[6] - 2 * np.sqrt(cov[6, 6]) for mu, cov in zip(mus, covs)], 'r--') # -2 std

plt.subplot(2, 1, 2)
plt.title('Velocity X')
plt.plot([mu[3] for mu in mus], 'r')
plt.plot([mu[3] + 2 * np.sqrt(cov[3, 3]) for mu, cov in zip(mus, covs)], 'r--') # +2 std
plt.plot([mu[3] - 2 * np.sqrt(cov[3, 3]) for mu, cov in zip(mus, covs)], 'r--') # -2 std

plt.tight_layout()
plt.show()
