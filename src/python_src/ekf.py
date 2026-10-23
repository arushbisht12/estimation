import numpy as np
from numpy import linalg as la

class EKF:
    def __init__(self, initial_q: list[float], initial_vel: list[float], 
                 initial_pos: list[float]) -> None:
        # MOVE LIST TO NPARRAY CONVERSION INSIDE INIT
         
        # mean of state GRV(Gaussian Random Variable) 
        self._x = np.concatenate((initial_q, initial_vel, initial_pos)).reshape((10,1)) #currently assumes initial vals are nparray
        self.n_states = self._x.size
        
        # covariance of state
        self._P = np.eye(self.n_states)
        
        # process noise covariance matrix
        self.process_noise = np.zeros([6,6])
        # measurement noise covariance matrix
        self.R = np.eye(11)
        
        # log of states?
        
        # update rate hz
        #self.frequency = frequency
        
        # constants
        self.g_const = 9.80665 #m/s^2
        self.g_vector = np.array([0,0,1]).reshape((3,1))
        self.theta_const = 0 #estimate theta at intiailization from stationary mag: arctan(x/y)
        self.north_vector = np.array([0, np.cos(self.theta_const), -np.sin(self.theta_const)]).reshape([3,1])
        
    #def update_EKF(self, x_prev, P_prev, gyr: list[float], acc: list[float], 
    #               mag: list[float], gps: list[float]) -> np.ndarray:
    #    pass
    

    def prediction(self, dt: float, gyr=None, acc=None) -> None:
        if gyr is None:
            gyr = np.zeros(3).reshape(-1, 1)
        if acc is None:
            acc = np.zeros(3).reshape(-1, 1)
            
        # x = f(x, u)
        self.mean = self.f(gyr, acc, dt) 
        
        # Q = W N Wt
        W_k = self.W(dt)
        Q = W_k @ self.process_noise @ W_k.T
        
        # P = F P Ft + Q
        F_k = self.F(gyr, acc, dt)
        self._P = F_k @ self._P @ F_k.T + Q
        
        # CHANGES: check if new x or P are NaN or infinity, do something about it
        
        
    def correction(self, dt: float, acc=None, mag=None, gps=None, lid=None, bar=None) -> None:
        if acc is None:
            acc_norm = np.zeros(3).reshape(-1, 1)
        else:
            acc_norm = acc / la.vector_norm(acc)
        if mag is None:
            mag_norm = np.zeros(3).reshape(-1, 1)
        else:
            mag_norm = mag / la.vector_norm(mag)
        if gps is None:
            gps = np.zeros(3).reshape(-1, 1)
        if lid is None:
            lid = np.zeros(1).reshape(-1, 1)
        if bar is None:
            bar = np.zeros(1).reshape(-1, 1)
            
        # normalize acc and mag
        
        # S = H P Ht + R
        H_k = self.H(self.theta_const)
        S = H_k @ self._P @ H_k.T + self.R
        
        # K = P Ht S^-1
        K = self._P @ H_k.T @ np.linalg.solve(S, np.eye(S.shape[0]))
        
        # x = x + K(z - h(x))
        z = np.concatenate((acc_norm, mag_norm, gps, lid, bar)).reshape((11,1))
        self.mean = self._x + K @ (z - self.h(dt))
        #self._x[0:4] /= la.vector_norm(self._x[0:4]) # normalize quaternion

        
        # P = (I - K H) P
        self._P = (np.eye(10) - K @ H_k) @ self._P
        
        # CHANGES: check if S is invertible or not
    
    
    
    # HELPER FUNCTIONS
    
    # overload update by giving the params it doesnt recieve 0
    
    def f(self, gyr, acc, dt: float) -> np.ndarray:
        q_prev = self._x[0:4]
        vel_prev = self._x[4:7]
        pos_prev = self._x[7:10]
        
        q_new = (np.eye(4) + 0.5 * dt * self.omega(gyr)).dot(q_prev) # switch to @
        #use q_old or q_new?
        vel_new = vel_prev + self.rotation_body_to_earth(q_new).dot(acc) * dt # switch to @
        vel_new[2] -= self.g_const * dt # to account for graivty acceleration
        pos_new = pos_prev + vel_new * dt
        
        return np.concatenate((q_new, vel_new, pos_new)).reshape((10,1))
    
    def h(self, dt) -> np.ndarray:
        acc_ex = self.rotation_earth_to_body(self.q) @ self.g_vector
        mag_ex = self.rotation_earth_to_body(self.q) @ self.north_vector
        gps_ex = self.pos.reshape((3,1))
        lid_ex = np.array([self.pos[2]]).reshape((1,1))
        bar_ex = np.array([self.pos[2]]).reshape((1,1))
        
        return np.concatenate((acc_ex,mag_ex,gps_ex,lid_ex,bar_ex))
    
    
    def F(self, gyr, acc, dt: float) -> np.ndarray:
        w1, w2, w3 = np.asarray(gyr).reshape(3,)
        a1, a2, a3 = np.asarray(acc).reshape(3,)
        q1, q2, q3, q4 = np.asarray(self.q).reshape(4,)

        F_sym = np.array([
            [1,        -(dt*w1)/2,  -(dt*w2)/2,  -(dt*w3)/2,  0, 0, 0, 0, 0, 0],
            [(dt*w1)/2, 1,           (dt*w2)/2,  -(dt*w3)/2,  0, 0, 0, 0, 0, 0],
            [(dt*w2)/2, -(dt*w3)/2,  1,           (dt*w1)/2,  0, 0, 0, 0, 0, 0],
            [(dt*w3)/2,  (dt*w2)/2, -(dt*w1)/2,  1,           0, 0, 0, 0, 0, 0],

            [dt*(2*a1*q1 - 2*a2*q4 + 2*a3*q3),
            dt*(2*a1*q2 + 2*a2*q3 + 2*a3*q4),
            dt*(2*a2*q2 - 2*a1*q3 + 2*a3*q1),
            -dt*(2*a2*q1 + 2*a1*q4 - 2*a3*q2),
            1, 0, 0, 0, 0, 0],

            [dt*(2*a2*q1 + a1*q4 - 2*a3*q2),
            -dt*(2*a2*q2 - a1*q3 + 2*a3*q1),
            dt*(a1*q2 + 2*a2*q3 + 2*a3*q4),
            dt*(a1*q1 - 2*a2*q4 + 2*a3*q3),
            0, 1, 0, 0, 0, 0],

            [dt*(2*a2*q2 - 2*a1*q3 + 2*a3*q1),
            dt*(2*a2*q1 + 2*a1*q4 - 2*a3*q2),
            -dt*(2*a1*q1 - 2*a2*q4 + 2*a3*q3),
            dt*(2*a1*q2 + 2*a2*q3 + 2*a3*q4),
            0, 0, 1, 0, 0, 0],

            [0, 0, 0, 0, dt, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, dt, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, dt, 0, 0, 1]
        ])

        return F_sym

    
    def H(self, theta: float) -> np.ndarray: # delete theta?
        q1, q2, q3, q4 = np.asarray(self.q).reshape(4,)

        g = self.g_const
        theta = self.theta_const # delete? 

        c = np.cos(theta)
        s = np.sin(theta)

        H_sym = np.array([
            [ 2*g*q3, -2*g*q4,  2*g*q1, -2*g*q2, 0, 0, 0, 0, 0, 0],
            [-2*g*q2, -2*g*q1, -2*g*q4, -2*g*q3, 0, 0, 0, 0, 0, 0],
            [-2*g*q1,  2*g*q2,  2*g*q3, -2*g*q4, 0, 0, 0, 0, 0, 0],

            [ q4*c + 2*q3*s,  q3*c - 2*q4*s,  q2*c + 2*q1*s,  q1*c - 2*q2*s, 0, 0, 0, 0, 0, 0],
            [ 2*q1*c - 2*q2*s, -2*q2*c - 2*q1*s, 2*q3*c - 2*q4*s, -2*q4*c - 2*q3*s, 0, 0, 0, 0, 0, 0],
            [-2*q2*c - 2*q1*s,  2*q2*s - 2*q1*c, 2*q4*c + 2*q3*s,  2*q3*c - 2*q4*s, 0, 0, 0, 0, 0, 0],

            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])

        return H_sym

    
    def W(self, dt: float) -> np.ndarray:
        q1, q2, q3, q4 = np.asarray(self.q).reshape(4,)

        W_sym = np.array([
            [-(dt*q2)/2, -(dt*q3)/2, -(dt*q4)/2, 0, 0, 0],
            [ (dt*q1)/2,  (dt*q3)/2, -(dt*q4)/2, 0, 0, 0],
            [ (dt*q4)/2,  (dt*q1)/2, -(dt*q2)/2, 0, 0, 0],
            [-(dt*q3)/2,  (dt*q2)/2,  (dt*q1)/2, 0, 0, 0],

            [0, 0, 0,
            dt*(q1**2 + q2**2 - q3**2 - q4**2),
            -dt*(2*q1*q4 - 2*q2*q3),
            dt*(2*q1*q3 + 2*q2*q4)],

            [0, 0, 0,
            dt*(q1*q4 + q2*q3),
            dt*(q1**2 - q2**2 + q3**2 - q4**2),
            -dt*(2*q1*q2 - 2*q3*q4)],

            [0, 0, 0,
            -dt*(2*q1*q3 - 2*q2*q4),
            dt*(2*q1*q2 + 2*q3*q4),
            dt*(q1**2 - q2**2 - q3**2 + q4**2)],

            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])

        return W_sym
    
    
    def omega(self, gyr) -> np.ndarray: 
        gp, gq, gr = np.asarray(gyr).reshape(3,)
        return np.array([
            [0.0, -gp, -gq, -gr],
            [gp,  0.0,  gr, -gq],
            [gq, -gr,  0.0, gp], #mistake in lorenzo's omega
            [gr,  gq, -gp,  0.0]
        ])
    
    def rotation_body_to_earth(self, q: np.ndarray) -> np.ndarray:
        a, b, c, d = np.asarray(q).reshape(4,)
        return np.array([
            [a*a + b*b - c*c - d*d,  -2*a*d + 2*b*c,         2*a*c + 2*b*d],
            [2*a*d + 2*b*c,          a*a - b*b + c*c - d*d,  -2*a*b + 2*c*d],
            [-2*a*c + 2*b*d,         2*a*b + 2*c*d,          a*a - b*b - c*c + d*d]
    ])
        
    def rotation_earth_to_body(self, q:np.ndarray) -> np.ndarray:
        return self.rotation_body_to_earth(q).T
    

    @property
    def mean(self) -> np.ndarray:
        return self._x[:,0]
    
    @property
    def cov(self) -> np.ndarray:
        return self._P
     
    @property
    def q(self) -> np.ndarray:
        return self._x[0:4,0]  
    
    @property   
    def vel(self) -> np.ndarray:
        return self._x[4:7,0]
    
    @property
    def pos(self) -> np.ndarray:
        return self._x[7:10,0]
    
    @mean.setter
    def mean(self, value: np.ndarray):
        self._x = np.asarray(value).reshape(10,1)
        self._x[0:4] /= la.vector_norm(self._x[0:4]) # normalize quaternion
        