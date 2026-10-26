import numpy as np
from numpy import linalg as la
from util import skewSymmetric, quatToMatrix, quat_multiply, quat_to_euler
from pyquaternion import Quaternion

class MEKF:
        # Codex: internal vectors are columns; public vector properties remain flat.
    @staticmethod
    def _column(value, size, name):
        array = np.asarray(value, dtype=float)
        if array.size != size:
            raise ValueError(f"{name} must contain {size} elements, got {array.size}")
        allowed_shapes = [(size,), (size, 1), (1, size)]
        if size == 1:
            allowed_shapes.append(())
        if array.shape not in allowed_shapes:
            raise ValueError(f"{name} must be a vector, got shape {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
        return array.reshape(size, 1)

    @staticmethod
    def _matrix(value, shape, name):
        array = np.asarray(value, dtype=float)
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
        return array

    @classmethod
    def _direction(cls, value, name):
        vector = cls._column(value, 3, name)
        norm = la.vector_norm(vector)
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError(f"{name} must have a finite norm greater than 1e-12")
        return vector / norm
    
    def __init__(self, initial_q, initial_vel, initial_pos,
                 gyro_cov, accel_cov,         # Values usually found in IMU datasheet (units like rad/s/sqrt(Hz))
                 gyro_bias_cov, accel_bias_cov,
                 gps_cov=1.0, P_initial=None) -> None:
        initial_gyr_b = np.zeros(3)
        initial_acc_b = np.zeros(3)
        # Codex: validate each input before assembling the column state.
        self._x = np.concatenate((self._column(initial_q, 4, "initial_q"),
                                  self._column(initial_vel, 3, "initial_vel"),
                                  self._column(initial_pos, 3, "initial_pos"),
                                  initial_gyr_b.reshape(3, 1),
                                  initial_acc_b.reshape(3, 1)))
        
        #self._error_x = np.zeros(15).reshape(-1,1)     # error state vector is always 0 in predict step so not needed to be stored
        if P_initial is None:
            self._P = np.identity(15, dtype=float)
        else:
            self._P = self._matrix(P_initial, (15, 15), "P_initial").copy()
        
        # PARAMETERS
        self.dt = 0.01  #PLACEHOLDER
        self.g_const = 9.80665
        self.theta_const = 0
        self.g_vector = np.array([0,0,1]).reshape((3,1))
        self.north_vector = np.array([0, np.cos(self.theta_const), -np.sin(self.theta_const)]).reshape([3,1])
        
        self.gyro_cov_mat = gyro_cov * np.identity(3, dtype=float)
        self.accel_cov_mat = accel_cov * np.identity(3, dtype=float)
        self.gyro_bias_cov_mat = gyro_bias_cov * np.identity(3, dtype=float)
        self.accel_bias_cov_mat = accel_bias_cov * np.identity(3, dtype=float)
        self.gps_cov_mat = gps_cov * np.identity(3, dtype=float)


    def prediction(self, dt, gyr=None, acc=None):
        # Codex: convert measurements to columns before subtracting column biases.
        # subtract biases from sensor measurements
        if gyr is None:
            gyr = np.zeros(3).reshape(-1,1)
        else: 
            gyr = self._column(gyr, 3, "gyr") - self._x[10:13]
        if acc is None:
            acc = np.zeros(3).reshape(-1,1)
        else:
            acc = self._column(acc, 3, "acc") - self._x[13:16]
        
        # predict nominal values
        self._x[0:10] = self.f(gyr, acc, dt)
        
        # predict error values
        process_model = self.phi(gyr, acc, dt)
        Q = self.process_covariance(dt, acc=acc, gyr=gyr)
        self._P = process_model @ self._P @ process_model.T + Q
        
    def correction(self, dt: float, acc=None, mag=None, gps=None, lid=None, bar=None) -> None:
        # Codex: validate measurement columns and direction norms at entry.
        retained_index_set = 11 * [True]
        if acc is None:
            retained_index_set[0:3] = 3 * [False]
            acc_norm = None
        else:
            acc_norm = self._direction(acc, "acc")
        if mag is None:
            retained_index_set[3:6] = 3 * [False]
            mag_norm = None
        else:
            mag_norm = self._direction(mag, "mag")
        if gps is None:
            retained_index_set[6:9] = 3 * [False]
        else:
            gps = self._column(gps, 3, "gps")
        if lid is None:
            retained_index_set[9] = False
        else:
            lid = self._column(lid, 1, "lid")
        if bar is None:
            retained_index_set[10] = False
        else:
            bar = self._column(bar, 1, "bar")
            
        if sum(retained_index_set) == 0:
            return 0 # is this okay to do ?
        
        
        # # Codex: keep measurement dimensions separate from the 15-state correction.
        # H_k = self._matrix(H_k, (H_k.shape[0], 15), "H")
        # z = self._column(np.concatenate((acc_norm, mag_norm, gps, lid, bar)),
        #                  H_k.shape[0], "z")
        # expected = self._column(self.h(), H_k.shape[0], "h")
        # innovation = self._column(z - expected, H_k.shape[0], "innovation")
        # measurement_cov = self._matrix(self.measurement_covariance(),
        #                                (H_k.shape[0], H_k.shape[0]), "R")
            
        
        # Kalman gain
        H_k = self.H()[retained_index_set,:]
        P_Ht = self._P @ H_k.T
        # Calculate P Ht used twice
        # S = H P Ht + R
        R_k = self.measurement_covariance()[np.ix_(retained_index_set, retained_index_set)]
        S = H_k @ P_Ht + R_k
        
        # Codex: S = H_k @ P_Ht + measurement_cov
        
        # K = P Ht S^-1
        K_k = P_Ht @ np.linalg.solve(S, np.eye(S.shape[0]))
        
        # A posteriori covariance
        self._P = (np.identity(15, dtype=float) - K_k @ H_k) @ self._P
        
        # A posteriori ERROR state
        measurements = [acc_norm, mag_norm, gps, lid, bar]
        valid_measurements = [a for a in measurements if a is not None]
        z = np.concatenate(valid_measurements).reshape((sum(retained_index_set),1))
        
        h_k = self.h()[retained_index_set]
        error_state = K_k @ (z - h_k)
        
        # # Codex: validate the full error correction before mutating state or covariance.
        # error_state = self._column(K_k @ innovation, 15, "error_state")
        
        # TODO deal with error state not having shape 11 all the time
        # Fold error state back into nominal states
        small_angle_approx_q = np.concatenate(([1.0], 0.5 * error_state[0:3].ravel())).reshape(-1,1)
        small_angle_approx_q = small_angle_approx_q / la.norm(small_angle_approx_q)
        self._x[0:4] = quat_multiply(self.q, small_angle_approx_q)
        self._x[4:7] = self.vel.reshape(-1,1) + error_state[3:6]
        self._x[7:10] = self.pos.reshape(-1,1) + error_state[6:9]
        self._x[10:13] += error_state[9:12]
        self._x[13:16] += error_state[12:15]
        
        # covariance resetting around new attitude 
        small_angle_error = error_state[0:3]
        reset_error_func = np.eye(3,3) - 0.5 * skewSymmetric(small_angle_error)
        
        reset_jacobian = np.eye(15,15)
        reset_jacobian[0:3, 0:3] = reset_error_func
        
        self._P = reset_jacobian @ self._P @ reset_jacobian.T
        
        
        # # Codex: quat_multiply uses flat (4,) inputs and returns a flat quaternion here.
        # small_angle_approx_q = np.concatenate(([1.0], 0.5 * error_state[0:3, 0]))
        # new_q = self._column(quat_multiply(self.q, small_angle_approx_q), 4, "new_q")
        # q_norm = la.vector_norm(new_q)
        # if not np.isfinite(q_norm) or q_norm <= 1e-12:
        #     raise ValueError("new_q must have a finite nonzero norm")
        # updated_state = self._x.copy()
        # updated_state[0:4] = new_q / q_norm
        # updated_state[4:16] += error_state[3:15]
        # updated_state = self._column(updated_state, 16, "updated_state")
        # self._P = (np.identity(15, dtype=float) - K_k @ H_k) @ self._P
        # self._x[:] = updated_state
        
       
    #HELPER
        
    def f(self, gyr, acc, dt: float) -> np.ndarray:
        # Codex: direct helper calls follow the same column-vector contract.
        gyr = self._column(gyr, 3, "gyr")
        acc = self._column(acc, 3, "acc")
        q_prev = self._x[0:4]
        vel_prev = self._x[4:7]
        pos_prev = self._x[7:10]
        
        q_new = (np.eye(4) + 0.5 * dt * self.omega(gyr)).dot(q_prev) # switch to @
        q_new = q_new / la.norm(q_new) # normalize
        #use q_old or q_new?
        vel_new = vel_prev + self.rotation_body_to_earth(q_new).dot(acc) * dt # switch to @
        vel_new[2] -= self.g_const * dt # to account for graivty acceleration
        pos_new = pos_prev + vel_new * dt
        
        return self._column(np.concatenate((q_new, vel_new, pos_new)), 10, "predicted_state")
    
    # state transition matrix
    def phi(self, gyr, acc, dt) -> np.ndarray:
         # Codex: validate sensor vectors; util.quat_rotation_matrix receives flat self.q.
        gyr = self._column(gyr, 3, "gyr")
        acc = self._column(acc, 3, "acc")
        G = np.zeros(shape=(15, 15), dtype=float)
        
        G[0:3, 0:3] = -skewSymmetric(gyr)
        G[3:6, 0:3] = -quatToMatrix(self.q) @ skewSymmetric(acc)
        G[3:6, 12:15] = -quatToMatrix(self.q)
        G[6:9, 3:6] = np.identity(3, dtype=float)
        G[0:3, 9:12] = -np.identity(3, dtype=float)
        
        return np.identity(15, dtype=float) + G * dt 
    
    def process_covariance(self, dt, acc=None, gyr=None):
        """Use corrected body IMU readings and the predicted attitude.

        gyro_cov_mat retains its covariance-intensity units; the effective
        interval-averaged gyro sample covariance is gyro_cov_mat / dt.
        """
        # Codex: validate the discrete noise inputs; zero duration adds no noise.
        dt = float(dt)
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        acc = np.zeros((3, 1)) if acc is None else self._column(acc, 3, "acc")
        gyr = np.zeros((3, 1)) if gyr is None else self._column(gyr, 3, "gyr")
        Q = np.zeros((15, 15), dtype=float)
        if dt == 0:
            return Q
        
        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4
        dt5 = dt**5
        
        R = self.rotation_body_to_earth(self.q)

        Q[0:3, 0:3] = self.gyro_bias_cov_mat * (dt3 / 3.0)
        Q[9:12, 0:3] = -self.gyro_bias_cov_mat * (dt2 / 2.0)
        Q[0:3, 9:12] = Q[9:12, 0:3]
        Q[9:12, 9:12] = self.gyro_bias_cov_mat * dt

        Q[3:6, 3:6] = self.accel_cov_mat * dt + self.accel_bias_cov_mat * (dt3 / 3.0)
        
        Q[6:9, 3:6] = self.accel_cov_mat * (dt2 / 2.0) + self.accel_bias_cov_mat * (dt4 / 8.0)
        Q[3:6, 6:9] = Q[6:9, 3:6]
        
        Q[12:15, 3:6] = -self.accel_bias_cov_mat @ R.T * (dt2 / 2.0)
        Q[3:6, 12:15] = Q[12:15, 3:6].T
        
        Q[6:9, 6:9] = self.accel_cov_mat * (dt3 / 3.0) + self.accel_bias_cov_mat * (dt5 / 20.0)
        
        Q[12:15, 6:9] = -self.accel_bias_cov_mat @ R.T * (dt3 / 6.0)
        Q[6:9, 12:15] = Q[12:15, 6:9].T
        
        Q[12:15, 12:15] = self.accel_bias_cov_mat * dt

        # Codex: one gyro sample affects attitude, then velocity, then position in f().
        half_step = 0.5 * dt * gyr
        attitude_jacobian = (dt / (1.0 + float((half_step.T @ half_step).item()))
                             * (np.eye(3) - skewSymmetric(half_step)))
        acceleration_jacobian = -R @ skewSymmetric(acc)
        gyro_jacobian = np.zeros((15, 3), dtype=float)
        gyro_jacobian[0:3] = attitude_jacobian
        gyro_jacobian[3:6] = acceleration_jacobian @ attitude_jacobian * dt
        gyro_jacobian[6:9] = gyro_jacobian[3:6] * dt
        Q += gyro_jacobian @ (self.gyro_cov_mat / dt) @ gyro_jacobian.T

        return Q
        
    def h(self) -> np.ndarray:
        acc_ex = self.rotation_earth_to_body(self.q) @ self.g_vector
        mag_ex = self.rotation_earth_to_body(self.q) @ self.north_vector
        gps_ex = self.pos.reshape((3,1))
        lid_ex = np.array([self.pos[2]]).reshape((1,1))
        bar_ex = np.array([self.pos[2]]).reshape((1,1))
        
        return np.concatenate((acc_ex,mag_ex,gps_ex,lid_ex,bar_ex))
    
    def H(self) -> np.ndarray: 
        H = np.zeros(shape = (11,15), dtype=float)
        
        H[0:3, 0:3] = skewSymmetric( self.rotation_earth_to_body(self.q) @ self.g_vector)
        H[0:3, 12:15] = np.identity(3, dtype=float)
        H[3:6, 0:3] = skewSymmetric( self.rotation_earth_to_body(self.q) @ self.north_vector)
        H[3:6, 12:15] = np.identity(3, dtype=float)
        H[6:9, 6:9] = np.identity(3, dtype=float)
        H[9, 8] = 1.0
        H[10, 8] = 1.0
        
        return H
    
    def measurement_covariance(self):
        R = np.identity(11, dtype=float)
        R[6:9, 6:9] = self.gps_cov_mat
        return R
        
        
    @property
    def q(self) -> np.ndarray:
        return self._x[0:4,0].reshape((4,1))
    
    @property
    def vel(self) -> np.ndarray:
        return self._x[4:7,0].reshape((3,1))
    
    @property
    def pos(self) -> np.ndarray:
        return self._x[7:10,0].reshape((3,1))
    
    @property 
    def gyr_b(self) -> np.ndarray:
        return self._x[10:13,0].reshape((3,1))
    
    @property 
    def acc_b(self) -> np.ndarray:
        return self._x[13:16,0].reshape((3,1))
    
    @property
    def mean(self) -> np.ndarray:
        return np.concatenate((quat_to_euler(self._x[0:4,0]),(self._x[4:,0]))).reshape((15,1))
    
    @property
    def cov(self) -> np.ndarray:
        return self._P.reshape((15,15))
    
    
    
    
    def omega(self, gyr) -> np.ndarray: 
        # Codex: unpack validated flat components only at the scalar-formula boundary.
        gp, gq, gr = self._column(gyr, 3, "gyr")[:, 0]
        return np.array([
            [0.0, -gp, -gq, -gr],
            [gp,  0.0,  gr, -gq],
            [gq, -gr,  0.0, gp], #mistake in lorenzo's omega
            [gr,  gq, -gp,  0.0]
        ])
         
    def rotation_body_to_earth(self, q: np.ndarray) -> np.ndarray:
        # Codex: accept quaternion vectors and unpack four scalar components.
        a, b, c, d = self._column(q, 4, "q")[:, 0]
        return np.array([
            [a*a + b*b - c*c - d*d,  -2*a*d + 2*b*c,         2*a*c + 2*b*d],
            [2*a*d + 2*b*c,          a*a - b*b + c*c - d*d,  -2*a*b + 2*c*d],
            [-2*a*c + 2*b*d,         2*a*b + 2*c*d,          a*a - b*b - c*c + d*d]
    ])
        
    def rotation_earth_to_body(self, q:np.ndarray) -> np.ndarray:
        return self.rotation_body_to_earth(q).T
