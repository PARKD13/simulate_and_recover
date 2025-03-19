#assisted with AI

import unittest
import numpy as np
import pandas as pd
from src.simulate import SimulationRunner
from src.ez_diffusion import EZ_diffusion

class TestSimulationRunner(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.ez = EZ_diffusion()
        self.small_runner = SimulationRunner(n_iterations=10, sample_sizes=[10])
    
    def test_simulation_runner_initialization(self):
        """Test that the SimulationRunner initializes correctly."""
        runner = SimulationRunner(n_iterations=5, sample_sizes=[10, 20])
        self.assertEqual(runner.n_iterations, 5)
        self.assertEqual(runner.sample_sizes, [10, 20])
        self.assertIsInstance(runner.ez, EZ_diffusion)
    
    def test_run_simulations_basic(self):
        """Test that run_simulations runs and returns a DataFrame."""
        # Run a very small simulation to test functionality
        results = self.small_runner.run_simulations()
        
        # Check that results is a DataFrame
        self.assertIsInstance(results, pd.DataFrame)
        
        # Check that it has the expected number of rows
        expected_rows = len(self.small_runner.sample_sizes) * self.small_runner.n_iterations
        self.assertEqual(len(results), expected_rows)
        
        # Check that it has the expected columns
        expected_columns = ['sample_size', 'iteration', 'true_drift', 'true_boundary', 
                           'true_nondecision', 'est_drift', 'est_boundary', 
                           'est_nondecision', 'drift_bias', 'boundary_bias', 
                           'nondecision_bias', 'drift_se', 'boundary_se', 'nondecision_se']
        for col in expected_columns:
            self.assertIn(col, results.columns)
    
    def test_analyze_results(self):
        """Test that analyze_results produces a summary."""
        # Create a small test DataFrame
        test_data = {
            'sample_size': [10, 10, 20, 20],
            'drift_bias': [0.1, -0.1, 0.05, -0.05],
            'boundary_bias': [0.2, -0.2, 0.1, -0.1],
            'nondecision_bias': [0.02, -0.02, 0.01, -0.01],
            'drift_se': [0.01, 0.01, 0.0025, 0.0025],
            'boundary_se': [0.04, 0.04, 0.01, 0.01],
            'nondecision_se': [0.0004, 0.0004, 0.0001, 0.0001]
        }
        df = pd.DataFrame(test_data)
        
        # Get summary
        summary = self.small_runner.analyze_results(df)
        
        # Check that summary is a DataFrame with MultiIndex
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertIsInstance(summary.columns, pd.MultiIndex)
        
        # Check that means are calculated correctly
        self.assertAlmostEqual(summary.loc[10, ('drift_bias', 'mean')], 0.0, places=6)
        self.assertAlmostEqual(summary.loc[10, ('boundary_bias', 'mean')], 0.0, places=6)
        self.assertAlmostEqual(summary.loc[20, ('drift_bias', 'mean')], 0.0, places=6)
        
        # Check that standard deviations are calculated correctly
        self.assertAlmostEqual(summary.loc[10, ('drift_bias', 'std')], 0.1414, places=4)
        
        # Check that means of squared errors are calculated correctly
        self.assertAlmostEqual(summary.loc[10, ('drift_se', 'mean')], 0.01, places=6)
        self.assertAlmostEqual(summary.loc[20, ('drift_se', 'mean')], 0.0025, places=6)

if __name__ == '__main__':
    unittest.main()
