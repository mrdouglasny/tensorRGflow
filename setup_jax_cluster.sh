#!/bin/bash
# Setup JAX environment on FASRC cluster for tensorRGflow
# Usage: bash setup_jax_cluster.sh

set -e

echo "=============================================="
echo "Setting up JAX environment for tensorRGflow"
echo "=============================================="

# Load CUDA and cuDNN modules (both required for JAX GPU)
echo "Loading CUDA 12.4 and cuDNN..."
module load cuda/12.4.1-fasrc01
module load cudnn/9.1.1.17_cuda12-fasrc01

# Create virtual environment
VENV_DIR="$HOME/.venvs/jax-gpu"
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "To reinstall, run: rm -rf $VENV_DIR && bash $0"
else
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Activate
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install JAX with CUDA support
echo "Installing JAX with CUDA 12 support..."
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install other dependencies
echo "Installing other dependencies..."
pip install numpy scipy ncon
pip install git+https://github.com/mhauru/tntools

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To use JAX, run:"
echo "  module load cuda/12.4.1-fasrc01"
echo "  source ~/.venvs/jax-gpu/bin/activate"
echo ""
echo "To test on a GPU node, submit an interactive job:"
echo "  salloc -p gpu -t 0:30:00 --gres=gpu:1 --mem=8G"
echo ""
echo "Then run the prereqs check again:"
echo "  python ~/rg/software_refs/tensorRGflow/check_prereqs.py"
