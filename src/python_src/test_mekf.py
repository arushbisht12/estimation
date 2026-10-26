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
        q0 = np.array([1,2,3,4]).reshape(-1,1)
        v0 = np.array([0,0,0]).reshape(-1,1)
        p0 = np.array([5,5,5]).reshape(-1,1)
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
        
        self.assertEqual(ekf.mean.shape, (15, 1))
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
        
        self.assertEqual(ekf.mean.shape, (15, 1))
        self.assertEqual(ekf.cov.shape, (15, 15))

    # Codex: consolidated coverage for vector layouts, validation, and GPS-only updates.
    def _make_filter(self, **kwargs):
        return self.MEKF([1, 0, 0, 0], [0, 0, 0], [1, 2, 3],
                         0.01, 0.02, 0.001, 0.001, **kwargs)

    def test_input_layouts_produce_the_same_prediction_and_correction(self):
        layouts = (lambda x: x, lambda x: np.array(x),
                   lambda x: np.array(x).reshape(-1, 1),
                   lambda x: np.array(x).reshape(1, -1))
        raw = dict(gyr=[0.1, 0.2, 0.3], acc=[0.2, 0.3, 9.9],
                   mag=[0, 1, 0], gps=[1, 2, 3], lid=[3], bar=[3])
        cases = [{name: layout(value) for name, value in raw.items()}
                 for layout in layouts]
        cases.append(dict(gyr=[0.1, 0.2, 0.3], acc=[[0.2], [0.3], [9.9]],
                          mag=np.array([0, 1, 0]), gps=[[1, 2, 3]],
                          lid=3.0, bar=np.array(3.0)))
        expected = None
        for index, readings in enumerate(cases):
            with self.subTest(layout=index):
                estimator = self._make_filter()
                estimator._x[10:13, 0] = [0.01, 0.02, 0.03]
                estimator._x[13:16, 0] = [0.1, 0.2, 0.3]
                estimator.prediction(0.01, readings['gyr'], readings['acc'])
                estimator.correction(0.1, **{name: readings[name]
                                             for name in ('acc', 'mag', 'gps', 'lid', 'bar')})
                self.assertTrue(np.isfinite(estimator._x).all())
                self.assertAlmostEqual(np.linalg.norm(estimator.q), 1.0)
                if expected is None:
                    expected = estimator._x.copy(), estimator.cov.copy()
                else:
                    npt.assert_allclose(estimator._x, expected[0])
                    npt.assert_allclose(estimator.cov, expected[1])

    def test_constructor_rejects_wrong_counts_layouts_and_nonfinite_values(self):
        for q, vel, pos in [([1, 0, 0], [0, 0, 0, 0], [0, 0, 0]),
                            (np.eye(2), [0, 0, 0], [0, 0, 0]),
                            ([1, 0, 0, 0], [np.nan, 0, 0], [0, 0, 0])]:
            with self.subTest(q=q, vel=vel):
                with self.assertRaises(ValueError):
                    self.MEKF(q, vel, pos, 0.01, 0.02, 0.001, 0.001)
        for covariance in (np.eye(14), np.full((15, 15), np.inf)):
            with self.assertRaises(ValueError):
                self._make_filter(P_initial=covariance)

    def test_invalid_measurements_leave_state_and_covariance_unchanged(self):
        invalid_calls = [
            lambda e: e.prediction(0.01, gyr=[0, 0]),
            lambda e: e.prediction(0.01, acc=[0, np.inf, 0]),
            lambda e: e.correction(0.1, acc=[0, 0, 0]),
            lambda e: e.correction(0.1, mag=[0, 0, np.nan]),
            lambda e: e.correction(0.1, gps=np.zeros((3, 1, 1))),
            lambda e: e.correction(0.1, lid=[1, 2]),
        ]
        for index, call in enumerate(invalid_calls):
            with self.subTest(measurement=index):
                estimator = self._make_filter()
                state, covariance = estimator._x.copy(), estimator.cov.copy()
                with self.assertRaises(ValueError):
                    call(estimator)
                npt.assert_array_equal(estimator._x, state)
                npt.assert_array_equal(estimator.cov, covariance)

    def test_direct_helpers_accept_lists_and_reject_malformed_vectors(self):
        estimator = self._make_filter()
        self.assertEqual(estimator.f([0, 0, 0], [0, 0, 9.8], 0.01).shape, (10, 1))
        self.assertEqual(estimator.phi([0, 0, 0], [0, 0, 9.8], 0.01).shape, (15, 15))
        with self.assertRaises(ValueError):
            estimator.omega([0, 0])
        with self.assertRaises(ValueError):
            estimator.rotation_body_to_earth(np.eye(2))

    def test_correction_with_only_gps_uses_three_measurement_components(self):
        import sys

        estimator = self._make_filter(gps_cov=0.25)
        before = estimator._x.copy()
        gps = np.array([[4.0], [5.0], [6.0]])
        captured = {}
        correction_code = estimator.correction.__func__.__code__

        # Capture the actual reduced matrices without replacing the correction logic.
        def capture(frame, event, arg):
            if frame.f_code is correction_code and event == 'return':
                captured.update(frame.f_locals)

        previous_profile = sys.getprofile()
        try:
            sys.setprofile(capture)
            estimator.correction(0.1, gps=gps)
        finally:
            sys.setprofile(previous_profile)

        for name, shape in {'H_k': (3, 15), 'R_k': (3, 3), 'h_k': (3, 1),
                            'z': (3, 1), 'P_Ht': (15, 3), 'S': (3, 3),
                            'K_k': (15, 3), 'error_state': (15, 1)}.items():
            with self.subTest(quantity=name):
                self.assertIn(name, captured)
                self.assertEqual(captured[name].shape, shape)

        expected_H = np.zeros((3, 15))
        expected_H[:, 6:9] = np.eye(3)
        npt.assert_array_equal(captured['H_k'], expected_H)
        npt.assert_allclose(captured['R_k'], 0.25 * np.eye(3))
        npt.assert_array_equal(captured['h_k'], before[7:10])
        npt.assert_array_equal(captured['z'], gps)
        npt.assert_allclose(estimator._x[7:10], before[7:10] + 0.8 * (gps - before[7:10]))
        expected_state = before.copy()
        expected_state[7:10] = estimator._x[7:10]
        npt.assert_allclose(estimator._x, expected_state)
        expected_covariance = np.eye(15)
        expected_covariance[6:9, 6:9] *= 0.2
        npt.assert_allclose(estimator.cov, expected_covariance, atol=1e-14)

    # Codex: validate gyro process noise against the actual discrete state prediction.
    def test_gyro_covariance_matches_prediction_finite_differences(self):
        from util import quat_multiply

        for dt in (0.01, 0.1):
            with self.subTest(dt=dt):
                q = np.array([0.8, 0.2, -0.3, 0.4])
                q /= np.linalg.norm(q)
                estimator = self.MEKF(q, [1, 2, 3], [4, 5, 6],
                                      0.02, 0.0, 0.0, 0.0)
                gyr = np.array([[0.8], [-0.5], [0.3]])
                acc = np.array([[2.0], [-1.0], [9.0]])
                nominal = estimator.f(gyr, acc, dt)
                conjugate = nominal[:4, 0].copy()
                conjugate[1:] *= -1
                jacobian = np.zeros((15, 3))
                epsilon = 1e-5
                for axis in range(3):
                    perturbation = np.zeros((3, 1))
                    perturbation[axis, 0] = epsilon
                    plus = estimator.f(gyr + perturbation, acc, dt)
                    minus = estimator.f(gyr - perturbation, acc, dt)
                    plus_angle = 2 * quat_multiply(conjugate, plus[:4, 0])[1:]
                    minus_angle = 2 * quat_multiply(conjugate, minus[:4, 0])[1:]
                    jacobian[:3, axis] = (plus_angle - minus_angle) / (2 * epsilon)
                    jacobian[3:9, axis] = ((plus[4:10] - minus[4:10]) / (2 * epsilon)).ravel()
                estimator._x[:10] = nominal
                expected = jacobian @ (np.eye(3) * 0.02 / dt) @ jacobian.T
                actual = estimator.process_covariance(dt, acc=acc, gyr=gyr)
                npt.assert_allclose(actual, expected, rtol=2e-6, atol=2e-11)

    def test_gyro_coupling_vanishes_without_force_and_scales_with_force(self):
        estimator = self.MEKF([1, 0, 0, 0], [0, 0, 0], [0, 0, 0],
                              0.02, 0.0, 0.0, 0.0)
        dt = 0.1
        no_force = estimator.process_covariance(dt, acc=[0, 0, 0])
        npt.assert_array_equal(no_force[3:, :], 0)
        npt.assert_array_equal(no_force[:, 3:], 0)
        npt.assert_allclose(no_force[:3, :3], np.eye(3) * 0.02 * dt)
        force = estimator.process_covariance(dt, acc=[0, 0, 10])
        doubled = estimator.process_covariance(dt, acc=[0, 0, 20])
        npt.assert_allclose(force[3:6, 3:6], np.diag([1, 1, 0]) * 100 * 0.02 * dt**3)
        npt.assert_allclose(force[6:9, 6:9], force[3:6, 3:6] * dt**2)
        npt.assert_allclose(doubled[3:9, 3:9], 4 * force[3:9, 3:9])
        npt.assert_allclose(doubled[3:9, :3], 2 * force[3:9, :3])

    def test_prediction_uses_corrected_imu_and_predicted_attitude_for_gyro_noise(self):
        estimator = self.MEKF([1, 0, 0, 0], [0, 0, 0], [0, 0, 0],
                              0.02, 0.0, 0.0, 0.0, P_initial=np.zeros((15, 15)))
        estimator._x[10:13] = [[0.1], [0.2], [-0.1]]
        estimator._x[13:16] = [[0.5], [-0.2], [0.3]]
        gyr = np.array([[0.4], [-0.3], [0.2]])
        acc = np.array([[2.0], [1.0], [8.0]])
        nominal = estimator.f(gyr, acc, 0.1)
        estimator.prediction(0.1, gyr + estimator.gyr_b, acc + estimator.acc_b)
        expected = estimator.process_covariance(0.1, acc=acc, gyr=gyr)
        npt.assert_allclose(estimator._x[:10], nominal)
        npt.assert_allclose(estimator.cov, expected)
        self.assertGreater(np.linalg.norm(expected[3:9, :3]), 0)

    def test_process_covariance_remains_symmetric_and_nonnegative(self):
        estimator = self._make_filter()
        q = np.array([0.8, 0.2, -0.3, 0.4])
        estimator._x[:4, 0] = q / np.linalg.norm(q)
        for dt in (0.001, 0.1):
            with self.subTest(dt=dt):
                covariance = estimator.process_covariance(dt, acc=[2, -1, 9], gyr=[0.8, -0.5, 0.3])
                npt.assert_allclose(covariance, covariance.T, atol=1e-14)
                self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-14)
        npt.assert_array_equal(estimator.process_covariance(0.0, acc=[2, -1, 9]), np.zeros((15, 15)))
        for dt in (-0.1, np.nan, np.inf):
            with self.subTest(invalid_dt=dt), self.assertRaises(ValueError):
                estimator.process_covariance(dt)
                
    def test_attitude_covariance_resetting_after_error_injection(self):
        P = np.eye(15,15)
        P[0,0] = 0.04 # roll error variance
        P[1,1] = 0.09 # pitch error variance
        P[2,2] = 0.01 # yaw error variance
        P[2,6], P[6,2] = 0.02, 0.02 # yaw / xpos covariance
        P[0,3], P[3,0] = 0.01, 0.01 # roll xvel covariance
        
        estimator = self._make_filter(P_initial=P, gps_cov=0.25)
        gps = np.array([[2], [2], [3]])
        estimator.correction(0.1, gps=gps)
        
        tol = 1e-12
        self.assertAlmostEqual(-0.00008, estimator.cov[1,3], delta=tol)
        self.assertAlmostEqual(-0.00008, estimator.cov[3,1], delta=tol)
        self.assertAlmostEqual(0.0004, estimator.cov[0,1], delta=tol)
        self.assertAlmostEqual(0.0004, estimator.cov[1,0], delta=tol)
        
        npt.assert_array_almost_equal(estimator.cov, estimator.cov.T)
        
        
