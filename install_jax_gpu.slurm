#!/bin/bash
#SBATCH -J install_jax
#SBATCH -p gpu_test
#SBATCH -t 0:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -o ~/rg/software_refs/tensorRGflow/install_jax_%j.out
#SBATCH -e ~/rg/software_refs/tensorRGflow/install_jax_%j.err

# Install JAX with GPU support - MUST run on GPU node per FASRC docs
# See: https://docs.rc.fas.harvard.edu/kb/gpgpu-computing-on-the-cluster/

set -e

echo "=============================================="
echo "Installing JAX on GPU node - $(date)"
echo "Node: $(hostname)"
echo "=============================================="

# Check GPU is available
nvidia-smi

# Load CUDA and cuDNN (both required for JAX GPU)
echo ""
echo "Loading CUDA and cuDNN modules..."
module load cuda/12.4.1-fasrc01
module load cudnn/9.1.1.17_cuda12-fasrc01

# Remove old venv if exists
VENV_DIR="$HOME/.venvs/jax-gpu"
if [ -d "$VENV_DIR" ]; then
    echo "Removing old venv..."
    rm -rf "$VENV_DIR"
fi

# Create fresh venv
echo "Creating virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install JAX with CUDA - on GPU node this should detect GPU
echo ""
echo "Installing JAX with CUDA 12 support..."
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install dependencies
echo ""
echo "Installing other dependencies..."
pip install numpy scipy ncon
pip install git+https://github.com/mhauru/tntools

# Test JAX GPU
echo ""
echo "=============================================="
echo "Testing JAX GPU support..."
echo "=============================================="

python3 << 'EOF'
import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

if jax.default_backend() == 'gpu':
    print("\n✅ GPU ACCELERATION ENABLED!")

    # Quick benchmark
    import time
    x = jnp.ones((4000, 4000))
    _ = jnp.dot(x, x).block_until_ready()  # warm up

    start = time.time()
    for _ in range(10):
        y = jnp.dot(x, x).block_until_ready()
    elapsed = time.time() - start

    gflops = 10 * 2 * 4000**3 / elapsed / 1e9
    print(f"  4000x4000 matmul: {elapsed:.3f}s ({gflops:.1f} GFLOPS)")
else:
    print("\n❌ WARNING: Running on CPU, not GPU!")
    print("   Check CUDA installation and jaxlib version")
EOF

echo ""
echo "=============================================="
echo "Installation complete! - $(date)"
echo "=============================================="
echo ""
echo "To use:"
echo "  module load cuda/12.4.1-fasrc01"
echo "  source ~/.venvs/jax-gpu/bin/activate"
