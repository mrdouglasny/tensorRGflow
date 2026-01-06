#!/usr/bin/env python3
"""
Check prerequisites for running tensorRGflow with JAX on cluster.
Run this script to verify GPU availability and dependencies.

Usage: python check_prereqs.py
"""

import sys
import subprocess

def check_python():
    print("=" * 60)
    print("PYTHON VERSION")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    print()

def check_cuda():
    print("=" * 60)
    print("CUDA / GPU CHECK")
    print("=" * 60)

    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("nvidia-smi output:")
            # Print just the GPU info lines
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines[:20]):
                print(line)
            print()
        else:
            print("nvidia-smi failed:", result.stderr)
    except FileNotFoundError:
        print("nvidia-smi not found - no NVIDIA GPU or drivers not installed")
    print()

    # Check CUDA environment variables
    import os
    cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH']
    print("CUDA environment variables:")
    for var in cuda_vars:
        val = os.environ.get(var, '<not set>')
        if len(val) > 80:
            val = val[:80] + '...'
        print(f"  {var}: {val}")
    print()

def check_modules():
    print("=" * 60)
    print("AVAILABLE MODULES (if on cluster)")
    print("=" * 60)
    try:
        result = subprocess.run(['module', 'avail', 'cuda'],
                              capture_output=True, text=True, shell=False)
        print(result.stdout or result.stderr or "module command not available")
    except:
        try:
            # Try with shell=True for module command
            result = subprocess.run('module avail cuda 2>&1 | head -20',
                                  capture_output=True, text=True, shell=True)
            print(result.stdout or "No CUDA modules found")
        except:
            print("module command not available (not on cluster or not using modules)")
    print()

def check_jax():
    print("=" * 60)
    print("JAX STATUS")
    print("=" * 60)

    try:
        import jax
        print(f"JAX version: {jax.__version__}")

        try:
            import jaxlib
            print(f"jaxlib version: {jaxlib.__version__}")
        except ImportError:
            print("jaxlib: NOT INSTALLED")

        # Check available devices
        print(f"\nJAX devices: {jax.devices()}")
        print(f"Default backend: {jax.default_backend()}")

        # Quick computation test
        import jax.numpy as jnp
        x = jnp.ones((1000, 1000))
        y = jnp.dot(x, x)
        print(f"Test computation (1000x1000 matmul): OK, result shape = {y.shape}")

    except ImportError as e:
        print(f"JAX: NOT INSTALLED ({e})")
        print("\nTo install JAX:")
        print("  CPU only:  pip install jax jaxlib")
        print("  With CUDA: pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
    print()

def check_other_deps():
    print("=" * 60)
    print("OTHER DEPENDENCIES")
    print("=" * 60)

    deps = [
        ('numpy', 'np'),
        ('scipy', 'scipy'),
        ('ncon', 'ncon'),
    ]

    for name, import_name in deps:
        try:
            mod = __import__(import_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  {name}: {version}")
        except ImportError:
            print(f"  {name}: NOT INSTALLED")

    # Check tntools
    try:
        import tntools
        print(f"  tntools: installed")
    except ImportError:
        print(f"  tntools: NOT INSTALLED (pip install git+https://github.com/mhauru/tntools)")

    print()

def check_tensorrgflow():
    print("=" * 60)
    print("TENSORRGFLOW CODE")
    print("=" * 60)

    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    codes_dir = os.path.join(script_dir, 'analysisCodes')

    if os.path.isdir(codes_dir):
        print(f"analysisCodes directory: {codes_dir}")
        files = os.listdir(codes_dir)
        py_files = [f for f in files if f.endswith('.py')]
        print(f"Python files: {', '.join(py_files)}")
    else:
        print(f"analysisCodes directory not found at {codes_dir}")
    print()

def main():
    print("\n" + "=" * 60)
    print("  TENSORRGFLOW PREREQUISITES CHECK")
    print("=" * 60 + "\n")

    check_python()
    check_cuda()
    check_modules()
    check_jax()
    check_other_deps()
    check_tensorrgflow()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Quick summary
    try:
        import jax
        jax_ok = True
        backend = jax.default_backend()
    except:
        jax_ok = False
        backend = None

    print(f"  JAX installed: {'YES' if jax_ok else 'NO'}")
    if jax_ok:
        print(f"  JAX backend: {backend}")
        if backend == 'gpu':
            print("  GPU acceleration: ENABLED")
        else:
            print("  GPU acceleration: DISABLED (running on CPU)")

    print()
    if not jax_ok:
        print("NEXT STEPS:")
        print("  1. Load CUDA module: module load cuda/12.x")
        print("  2. Create venv: python -m venv ~/.venvs/jax")
        print("  3. Activate: source ~/.venvs/jax/bin/activate")
        print("  4. Install JAX with CUDA:")
        print("     pip install --upgrade pip")
        print("     pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
        print("  5. Install other deps: pip install numpy scipy ncon")
        print("     pip install git+https://github.com/mhauru/tntools")
    print()

if __name__ == "__main__":
    main()
