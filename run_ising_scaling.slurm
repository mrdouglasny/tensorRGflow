#!/bin/bash
#SBATCH -J ising_scale
#SBATCH -p gpu_test
#SBATCH -t 0:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -o /n/home00/mdouglas/rg/software_refs/tensorRGflow/logs/ising_scaling_%j.out
#SBATCH -e /n/home00/mdouglas/rg/software_refs/tensorRGflow/logs/ising_scaling_%j.err

# Full tensorRGflow workflow: Find Tc -> Generate flow -> Extract scaling dimensions
# Validates GILT+HOTRG implementation by computing 2D Ising critical exponents

set -e

echo "=============================================="
echo "tensorRGflow Ising Scaling Dimensions"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "=============================================="

# Load modules
module load cuda/12.4.1-fasrc01
module load cudnn/9.1.1.17_cuda12-fasrc01

# Ensure CUDA libraries are in path
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Debug: show loaded modules and paths
module list
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Activate venv
source ~/.venvs/jax-gpu/bin/activate

# Verify GPU
echo ""
echo "JAX backend: $(python -c 'import jax; print(jax.default_backend())')"
echo ""

cd /n/home00/mdouglas/rg/software_refs/tensorRGflow/analysisCodes

# Parameters - reduced for faster test
CHI=12
GILTEPS=6e-5
NGILT=2
LEGCUT=2
MAXITER=20

echo "=============================================="
echo "Parameters:"
echo "  chi = $CHI"
echo "  gilteps = $GILTEPS"
echo "  Ngilt = $NGILT"
echo "  legcut = $LEGCUT"
echo "  maxiter = $MAXITER"
echo "=============================================="

# Step 1: Find critical temperature
echo ""
echo "=============================================="
echo "Step 1: Finding critical temperature (Tc)"
echo "=============================================="
python hotrgTc.py --chi $CHI --isGilt --isSym --Ngilt $NGILT --legcut $LEGCUT \
                  --gilteps $GILTEPS --maxiter $MAXITER --rootiter 15

# Step 2: Generate RG flow at Tc
echo ""
echo "=============================================="
echo "Step 2: Generating RG flow at Tc"
echo "=============================================="
python hotrgFlow.py --chi $CHI --Ngilt $NGILT --legcut $LEGCUT \
                    --gilteps $GILTEPS --maxiter $MAXITER

# Step 3: Extract scaling dimensions
echo ""
echo "=============================================="
echo "Step 3: Extracting scaling dimensions"
echo "=============================================="
python hotrgScale.py --chi $CHI --gilteps $GILTEPS --Ngilt $NGILT --legcut $LEGCUT \
                     --iRGlow 5 --iRGhi 15 --NscaleD 15

echo ""
echo "=============================================="
echo "Expected 2D Ising scaling dimensions:"
echo "  Delta = 0       (identity)"
echo "  Delta = 0.125   (spin field sigma)"
echo "  Delta = 1       (energy field epsilon)"
echo "  Delta = 2       (stress tensor)"
echo "=============================================="
echo ""
echo "Completed: $(date)"
echo "=============================================="
