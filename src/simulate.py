import numpy as np
import pandas as pd
import time
from ez_diffusion import EZ_diffusion

class SimulationResult:
    """Class to store simulation results."""
    def __init__(self, history, time_steps):
        self.history = history
        self.time_steps = time_steps

class SimulationRunner:

    def __init__(self, num_iterations=1000, sample_sizes=[10, 40, 4000]):
        self.num_iterations = num_iterations
        self.sample_sizes = sample_sizes
        self.ez = EZ_diffusion()

    def run_simulations(self):
        """Run the simulate-and-recover process"""
        print(f"Running simulate-and-recover process with {self.num_iterations} iterations for each sample size")
        
        results = []
        
        start_time = time.time()
        
        # simulation for each sample size
        for n in self.sample_sizes:
            print(f"Processing sample size N = {n}")
            
            for i in range(self.num_iterations):
                # indicate every 100 iterations
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Iteration {i + 1}/{self.num_iterations} (Elapsed time: {elapsed:.2f}s)")
                
                # select parameters
                true_drift = np.random.uniform(0.5, 2.0)
                true_boundary = np.random.uniform(0.5, 2.0)
                true_nondecision = np.random.uniform(0.1, 0.5)
                
                # observed summary statistics
                r_obs, m_obs, v_obs = self.ez.observed_statistics(
                    true_drift, true_boundary, true_nondecision, n
                )
                
                # parameters
                try:
                    est_params = self.ez.recover_parameters(r_obs, m_obs, v_obs)
                    
                    # calculate bias and squared error
                    drift_bias = true_drift - est_params['drift_rate']
                    boundary_bias = true_boundary - est_params['boundary']
                    nondecision_bias = true_nondecision - est_params['nondecision']
                    
                    drift_se = drift_bias ** 2
                    boundary_se = boundary_bias ** 2
                    nondecision_se = nondecision_bias ** 2
                    
                    results.append({
                        'sample_size': n,
                        'iteration': i + 1,
                        'true_drift': true_drift,
                        'true_boundary': true_boundary,
                        'true_nondecision': true_nondecision,
                        'est_drift': est_params['drift_rate'],
                        'est_boundary': est_params['boundary'],
                        'est_nondecision': est_params['nondecision'],
                        'drift_bias': drift_bias,
                        'boundary_bias': boundary_bias,
                        'nondecision_bias': nondecision_bias,
                        'drift_se': drift_se,
                        'boundary_se': boundary_se,
                        'nondecision_se': nondecision_se
                    })
                except Exception as e:
                    print(f"Error in iteration {i + 1} with N = {n}: {e}")
                    # error case
                    results.append({
                        'sample_size': n,
                        'iteration': i + 1,
                        'true_drift': true_drift,
                        'true_boundary': true_boundary,
                        'true_nondecision': true_nondecision,
                        'est_drift': np.nan,
                        'est_boundary': np.nan,
                        'est_nondecision': np.nan,
                        'drift_bias': np.nan,
                        'boundary_bias': np.nan,
                        'nondecision_bias': np.nan,
                        'drift_se': np.nan,
                        'boundary_se': np.nan,
                        'nondecision_se': np.nan
                    })
        
        # results DataFrame
        results_df = pd.DataFrame(results)
        return results_df
    
        
    def analyze_results(self, results_df):
        """analyze results and generate summary statistics"""
        # summary statistics
        summary = results_df.groupby('sample_size').agg({
            'drift_bias': ['mean', 'std'],
            'boundary_bias': ['mean', 'std'],
            'nondecision_bias': ['mean', 'std'],
            'drift_se': 'mean',
            'boundary_se': 'mean',
            'nondecision_se': 'mean'
        })
        
        print("\nSummary of Results:")
        print(summary)
            
        return summary

def run_simulation(num_iterations=1000, sample_sizes=[10, 40, 4000]):
    """Run the simulate-and-recover process for EZ diffusion model."""
    # SimulationRunner execution
    runner = SimulationRunner(num_iterations=num_iterations, sample_sizes=sample_sizes)
    results = runner.run_simulations()
    summary = runner.analyze_results(results)
    
    # Save results to CSV as well
    results.to_csv('results/results.csv', index=False)
    
    return results, summary

if __name__ == "__main__":
    run_simulation()
