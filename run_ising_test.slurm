#!/bin/bash
#SBATCH -J ising_gilt
#SBATCH -p gpu_test
#SBATCH -t 0:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -o /n/home00/mdouglas/rg/software_refs/tensorRGflow/logs/ising_test_%j.out
#SBATCH -e /n/home00/mdouglas/rg/software_refs/tensorRGflow/logs/ising_test_%j.err

# Test tensorRGflow GILT+HOTRG on 2D Ising at criticality
# Uses JAX GPU acceleration

# Create logs dir if needed
mkdir -p /n/home00/mdouglas/rg/software_refs/tensorRGflow/logs

echo "=============================================="
echo "tensorRGflow Ising Test - $(date)"
echo "=============================================="
echo "Node: $(hostname)"
echo ""

# Load required modules
module load cuda/12.4.1-fasrc01
module load cudnn/9.1.1.17_cuda12-fasrc01

# Activate JAX venv
source ~/.venvs/jax-gpu/bin/activate

# Verify GPU
echo "Checking JAX backend..."
python3 -c "import jax; print(f'Backend: {jax.default_backend()}')"
echo ""

# Check if hotrgTc.py exists and has correct syntax
cd /n/home00/mdouglas/rg/software_refs/tensorRGflow/analysisCodes
echo "Working directory: $(pwd)"
echo "Python script: $(ls hotrgTc.py)"
echo ""

# First test: just import check
echo "Testing imports..."
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    import jax
    print(f'JAX: {jax.__version__}')
    import numpy as np
    print(f'NumPy: {np.__version__}')
    import scipy
    print(f'SciPy: {scipy.__version__}')
    from ncon import ncon
    print('ncon: OK')
    print('All imports successful!')
except Exception as e:
    print(f'Import error: {e}')
    sys.exit(1)
"

echo ""
echo "=============================================="
echo "Running GILT+HOTRG at critical temperature"
echo "=============================================="

# Quick test: chi=10, few iterations
python3 hotrgTc.py --chi 10 --isGilt --isSym --Ngilt 2 --legcut 2 \
                   --gilteps 6e-5 --maxiter 10 --rootiter 5

echo ""
echo "=============================================="
echo "Test complete - $(date)"
echo "=============================================="
