import numpy as np

import numpy.testing as npt
import unittest

import multiplicative_ekf as mekf

from parameterized import parameterized_class

@parameterized_class([
   { "MEKF": mekf.MEKF}
])
class TestMEKF(unittest.TestCase):
    
    def test_can_construct_with_initial_x(self):
        q0 = np.array([1,2,3,4])#.reshape(-1,1)
        v0 = np.array([0,0,0])#.reshape(-1,1)
        p0 = np.array([5,5,5])#.reshape(-1,1)
        ekf = self.MEKF(q0, v0, p0, 1,1,1,1)
        
        npt.assert_allclose(ekf.q, q0)
        npt.assert_allclose(ekf.vel, v0)
        npt.assert_allclose(ekf.pos, p0)
        
    def test_after_prediction_x_and_P_shape(self):
        q0 = np.array([1,2,3,4])#.reshape(-1,1)
        v0 = np.array([0,0,0])#.reshape(-1,1)
        p0 = np.array([5,5,5])#.reshape(-1,1)
        ekf = self.MEKF(q0, v0, p0, 1,1,1,1)
        
        gyr = np.array([0.1,0.1,0.1]).reshape(-1,1)
        acc = np.array([0.1,0.1,0.1]).reshape(-1,1)

        ekf.prediction(0.01, gyr, acc)
        
        self.assertEqual(ekf.mean.shape, (15,))
        self.assertEqual(ekf.cov.shape, (15, 15))
        
    def test_after_correction_x_and_P_shape(self):
        q0 = np.array([1,2,3,4])#.reshape(-1,1)
        v0 = np.array([0,0,0])#.reshape(-1,1)
        p0 = np.array([5,5,5])#.reshape(-1,1)
        ekf = self.MEKF(q0, v0, p0, 1,1,1,1)
        
        acc = np.array([0.1,0.1,0.1]).reshape(-1,1)
        mag = np.array([0.1,0.1,0.1]).reshape(-1,1)
        gps = np.array([0.1,0.1,0.1]).reshape(-1,1)
        lid = np.array([0.1]).reshape(-1,1)
        bar = np.array([0.1]).reshape(-1,1)


        ekf.correction(0.01, acc, mag, gps, lid, bar)
        
        self.assertEqual(ekf.mean.shape, (15,))
        self.assertEqual(ekf.cov.shape, (15, 15))