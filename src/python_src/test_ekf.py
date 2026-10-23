import ekf as python_ekf

# import CPP wrapper

import os
cur_file_path = os.path.dirname(os.path.abspath(__file__))

import sys
#sys.path.__add__(cur_file_path + '/../cpp/build')
#print(sys.path)
sys.path.append(cur_file_path + '/../cpp_src/build')
#print(sys.path)

import ekf_cpp as cpp_ekf

# END import CPP wrapper


import numpy as np

import numpy.testing as npt
import unittest

from parameterized import parameterized_class



@parameterized_class([
   { "EKF": python_ekf.EKF},
   { "EKF": cpp_ekf.EKF}
])
class TestEKF(unittest.TestCase):

    def test_can_construct_with_initial_x(self):
        q0 = np.array([1,2,3,4])#.reshape(-1,1)
        v0 = np.array([0,0,0])#.reshape(-1,1)
        p0 = np.array([5,5,5])#.reshape(-1,1)
        ekf = self.EKF(q0, v0, p0)
        
        npt.assert_allclose(ekf.q, q0)
        npt.assert_allclose(ekf.vel, v0)
        npt.assert_allclose(ekf.pos, p0)
        
    #check sizes of f, F, W results
    def test_after_prediction_x_and_P_shape(self):
        q0 = np.array([1,2,3,4]).reshape(-1,1)
        v0 = np.array([0,0,0]).reshape(-1,1)
        p0 = np.array([5,5,5]).reshape(-1,1)
        ekf = self.EKF(q0, v0, p0)
        
        gyr = np.array([0.1,0.1,0.1]).reshape(-1,1)
        acc = np.array([0.1,0.1,0.1]).reshape(-1,1)

        ekf.prediction(0.01, gyr, acc)
        
        self.assertEqual(ekf.mean.shape, (10,))
        self.assertEqual(ekf.cov.shape, (10, 10))
    
    #sizes of h, H
    def test_after_correction_x_and_P_shape(self):
        q0 = np.array([1,2,3,4]).reshape(-1,1)
        v0 = np.array([0,0,0]).reshape(-1,1)
        p0 = np.array([5,5,5]).reshape(-1,1)
        ekf = self.EKF(q0, v0, p0)
        
        acc = np.array([0.1,0.1,0.1]).reshape(-1,1)
        mag = np.array([0.1,0.1,0.1]).reshape(-1,1)
        gps = np.array([0.1,0.1,0.1]).reshape(-1,1)
        lid = np.array([0.1]).reshape(-1,1)
        bar = np.array([0.1]).reshape(-1,1)


        ekf.correction(0.01, acc, mag, gps, lid, bar)
        
        self.assertEqual(ekf.mean.shape, (10,))
        self.assertEqual(ekf.cov.shape, (10, 10))
    
    """def test_finite_difference_jacobian_check_for_F_matrix(self):
        q0 = np.array([1,2,3,4]).reshape(-1,1)
        v0 = np.array([0,0,0]).reshape(-1,1)
        p0 = np.array([5,5,5]).reshape(-1,1)
        ekf = EKF(q0, v0, p0, frequency=100)
        
        gyr = np.array([0.1,0.1,0.1]).reshape(-1,1)
        acc = np.array([0.1,0.1,0.1]).reshape(-1,1)

        F = ekf.F(gyr, acc, dt=0.01)"""
        
    #def test_quaternion_normalized(self):
        
        
        
    
    
        