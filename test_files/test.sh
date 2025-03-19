echo "Running EZ Diffusion Model Test Suite"
echo "-------------------------------------"

# run tests using python3
python3 -m unittest test.test_ez_diffusion
python3 -m unittest test.test_simulate

# Alternative method if the above doesn't work
echo ""
echo "Trying alternative test discovery method if needed:"
cd "$(dirname "$0")/.."  # Move to the root directory
python3 -m unittest discover -s test

echo ""
echo "Test suite complete."
