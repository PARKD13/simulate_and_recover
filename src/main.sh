echo "Starting EZ Diffusion Model Simulate-and-Recover Program"
echo "--------------------------------------------------------"
echo "This program will execute 1000 iterations for each sample size 10, 40, and 4000"
echo "Total: 3000 iterations"
echo ""

# Python version
PYTHON_CMD="python3.12"

# verify Python version
$PYTHON_CMD --version
if [ $? -ne 0 ]; then
    echo "Error: Python 3.12.3 is not available as 'python3.12'. Trying different path..."
    
    # Try python3 with version check
    if python3 --version | grep -q "Python 3.12"; then
        PYTHON_CMD="python3"
        echo "Using $PYTHON_CMD instead."
    else
        echo "Error: Python 3.12.3 is not found. Please install."
        exit 1

# create a results directory if it doesn't exist
mkdir -p results

# run the Python script
$PYTHON_CMD -c "
import numpy as np
import os
from src.simulate import SimulationRunner
from src.ez_diffusion import EZ_diffusion

# Check Python version
import sys
if not (sys.version_info.major == 3 and sys.version_info.minor == 12):
    print(f'Warning: This script expects Python 3.12 but you are using {sys.version}')

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Initialize the simulation runner with 1000 iterations for each sample size
runner = SimulationRunner(num_iterations=1000, sample_sizes=[10, 40, 4000])

# run simulations
results = runner.run_simulations()

# analyze and print results
summary = runner.analyze_results(results)

# save to files
np.save('results/simulation_results.npy', results)
np.save('results/summary_results.npy', summary)

# text files for each sample size
for n in [10, 40, 4000]:
    subset = results[results['sample_size'] == n]
    
    # biases
    drift_bias = subset['drift_bias'].mean()
    boundary_bias = subset['boundary_bias'].mean()
    nondecision_bias = subset['nondecision_bias'].mean()
    
    # squared errors
    drift_se = subset['drift_se'].mean()
    boundary_se = subset['boundary_se'].mean() 
    nondecision_se = subset['nondecision_se'].mean()
    
    # write to file in the results directory
    with open(f'results/results_N{n}.txt', 'w') as main_file:
        main_file.write(f'N={n}\\n')
        main_file.write(f'Biases(v, a, t): [{drift_bias:.8f} {boundary_bias:.8f} {nondecision_bias:.8f}]\\n')
        main_files.write(f'Squared Errors(v, a, t): [{drift_se:.8f} {boundary_se:.8f} {nondecision_se:.8f}]\\n')

print('\\nResults saved to the results directory.')
"

echo ""
echo "Simulate-and-recover program completed."
echo "Results have been saved to the results directory."
