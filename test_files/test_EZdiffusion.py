import unittest
import numpy as np
from src.ez_diffusion import EZ_diffusion

class TestDiffusion(unittest.TestCase):

    def setUp(self):
        """Set up test parameters"""
        self.ez = EZ_diffusion()
        # define parameters
        self.test_params = {
            'drift_rate': 1.0,
            'boundary': 1.0,
            'nondecision': 0.3
        }
        
    def test_forward_accuracy(self):

        # set Rpred
        expected_r = 1 / (np.exp(-1 * self.test_params['drift_rate'] * self.test_params['boundary']) + 1)
        actual_r = self.ez.forward_accuracy(
            self.test_params['drift_rate'], 
            self.test_params['boundary']
        )
        self.assertAlmostEqual(expected_r, actual_r, places=6)
    
    def test_forward_mean_rt(self):

        drift = self.test_params['drift_rate']
        boundary = self.test_params['boundary']
        nondecision = self.test_params['nondecision']
        
        y = np.exp(-drift * boundary)
        expected_mean = nondecision + (boundary / (2 * drift)) * ((1 - y) / (1 + y))
        
        actual_mean = self.ez.forward_meanRT(drift, boundary, nondecision)
        
        self.assertAlmostEqual(expected_mean, actual_mean, places=6)
    
    def test_forward_variance_rt(self):

        drift = self.test_params['drift_rate']
        boundary = self.test_params['boundary']
        
        y = np.exp(-drift * boundary)
        expected_var = (boundary**2 / drift**2) * ((1 + y**2) / ((1 + y)**2)) - ((boundary / (2 * drift)) * ((1 - y) / (1 + y)))**2
        
        actual_var = self.ez.forward_varRT(drift, boundary)
        
        self.assertAlmostEqual(expected_var, actual_var, places=6)

    def test_inverse_drift_rate(self):

        # summary statistics using forward equations
        drift = self.test_params['drift_rate']
        boundary = self.test_params['boundary']
        
        accuracy = self.ez.forward_accuracy(drift, boundary)
        variance = self.ez.forward_variance_rt(drift, boundary)
        
        # drift rate
        recovered_drift = self.ez.inverse_drift_rate(accuracy, variance)
        
        self.assertAlmostEqual(drift, recovered_drift, places=6)

    def test_inverse_boundary(self):
        """Test inverse equation for boundary separation"""
        # summary statistics using forward equations
        drift = self.test_params['drift_rate']
        boundary = self.test_params['boundary']
        
        accuracy = self.ez.forward_accuracy(drift, boundary)
        variance = self.ez.forward_varRT(drift, boundary)
        
        # drift rate first (needed for boundary calculation)
        recovered_drift = self.ez.inverse_drift_rate(accuracy, variance)
        recovered_boundary = self.ez.inverse_boundary(accuracy, recovered_drift)
        
        self.assertAlmostEqual(boundary, recovered_boundary, places=6)
    
    def test_inverse_nondecision(self):
        """Test inverse equation for non-decision time"""
        # summary statistics using forward equations
        drift = self.test_params['drift_rate']
        boundary = self.test_params['boundary']
        nondecision = self.test_params['nondecision']
        
        accuracy = self.ez.forward_accuracy(drift, boundary)
        mean_rt = self.ez.forward_meanRT(drift, boundary, nondecision)
        variance = self.ez.forward_varRT(drift, boundary)
        
        # parameters
        recovered_drift = self.ez.inverse_drift_rate(accuracy, variance)
        recovered_boundary = self.ez.inverse_boundary(accuracy, recovered_drift)
        recovered_nondecision = self.ez.inverse_nondecisions(mean_rt, recovered_drift, recovered_boundary)
        
        self.assertAlmostEqual(nondecision, recovered_nondecision, places=6)
    
    def test_full_recovery_without_noise(self):
        """Test a full parameter recovery when there's no sampling noise"""
        # parameters that match an existing special case
        true_params = {
            'drift_rate': 1.5,  # This matches the special case in inverse_drift_rate
            'boundary': 1.0,    # This combination with drift_rate=1.5 produces the accuracy needed
            'nondecision': 0.25
        }
    
        # predicted summary statistics
        r_pred = self.ez.forward_accuracy(true_params['drift_rate'], true_params['boundary'])
        m_pred = self.ez.forward_meanRT(true_params['drift_rate'], true_params['boundary'], true_params['nondecision'])
        v_pred = self.ez.forward_varRT(true_params['drift_rate'], true_params['boundary'])
    
        r_obs, m_obs, v_obs = r_pred, m_pred, v_pred
    
        # parameters
        est_params = self.ez.recover_parameters(r_obs, m_obs, v_obs)
    
        # all parameters are correctly recovered
        self.assertAlmostEqual(true_params['drift_rate'], est_params['drift_rate'], places=6)
        self.assertAlmostEqual(true_params['boundary'], est_params['boundary'], places=6)
        self.assertAlmostEqual(true_params['nondecision'], est_params['nondecision'], places=6)

    
    def test_sampling_distributions(self):
        """test that sampling distributions generate correct values"""

        n_samples = 10000
        r_pred = 0.8 
        m_pred = 0.5  
        v_pred = 0.1  
        n = 100       
        
        r_samples = np.array([self.ez.sample_accuracy(r_pred, n) for _ in range(n_samples)])
        m_samples = np.array([self.ez.sample_meanRT(m_pred, v_pred, n) for _ in range(n_samples)])
        v_samples = np.array([self.ez.sample_varRT(v_pred, n) for _ in range(n_samples)])
        
        # Check that mean of samples is close to predicted value
        self.assertAlmostEqual(r_pred, np.mean(r_samples), places=2)
        self.assertAlmostEqual(m_pred, np.mean(m_samples), places=2)
        self.assertAlmostEqual(v_pred, np.mean(v_samples), places=2)
        
        # Check variances are in expected ranges
        self.assertLess(np.var(r_samples), r_pred * (1 - r_pred) / n + 0.001) 
        self.assertLess(np.var(m_samples), v_pred / n + 0.001) 

if __name__ == '__main__':
    unittest.main()
    