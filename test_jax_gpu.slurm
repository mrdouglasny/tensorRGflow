#!/bin/bash
#SBATCH -J test_jax_gpu
#SBATCH -p gpu
#SBATCH -t 0:10:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -o ~/rg/software_refs/tensorRGflow/test_jax_gpu_%j.out
#SBATCH -e ~/rg/software_refs/tensorRGflow/test_jax_gpu_%j.err

# Test JAX on GPU node

echo "=============================================="
echo "JAX GPU Test - $(date)"
echo "=============================================="
echo "Node: $(hostname)"
echo ""

# Load CUDA and cuDNN (both required for JAX GPU)
module load cuda/12.4.1-fasrc01
module load cudnn/9.1.1.17_cuda12-fasrc01

# Activate venv
source ~/.venvs/jax-gpu/bin/activate

# Run prereqs check
echo "Running prerequisites check..."
python ~/rg/software_refs/tensorRGflow/check_prereqs.py

# Quick JAX GPU test
echo ""
echo "=============================================="
echo "JAX GPU Benchmark"
echo "=============================================="

python3 << 'EOF'
import jax
import jax.numpy as jnp
import time

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
print()

# Warm up
x = jnp.ones((100, 100))
_ = jnp.dot(x, x).block_until_ready()

# Benchmark
sizes = [1000, 2000, 4000]
for n in sizes:
    x = jnp.ones((n, n))

    # Time it
    start = time.time()
    for _ in range(10):
        y = jnp.dot(x, x).block_until_ready()
    elapsed = time.time() - start

    gflops = 10 * 2 * n**3 / elapsed / 1e9
    print(f"  {n}x{n} matmul: {elapsed:.3f}s ({gflops:.1f} GFLOPS)")

print()
print("JAX GPU test completed successfully!")
EOF

echo ""
echo "Done - $(date)"
