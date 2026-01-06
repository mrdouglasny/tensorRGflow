# tensorRGflow - GILT + HOTRG with JAX Autodiff

**Repository:** https://github.com/brucelyu/tensorRGflow
**Paper:** [Scaling dimensions from linearized tensor renormalization group transformations](https://arxiv.org/abs/2102.08136) (Lyu & Hauru, 2021)
**Author:** Xinliang (Bruce) Lyu (ISSP, U-Tokyo)

## Overview

This package implements GILT (Graph Independent Local Truncations) combined with HOTRG (Higher-Order Tensor Renormalization Group), using JAX for automatic differentiation to compute scaling dimensions from the linearized RG transformation.

## Key Innovation

The central idea is to use JAX's `linearize` function to compute the Jacobian of the RG map at the fixed point, then extract scaling dimensions from its eigenvalues:

```python
# Linearize RG equation at fixed point
psiAp, responseMat = jax.linearize(equationRG, psiA)

# Get eigenvalues of linearized RG
RGhyperM = LinearOperator((dimA,dimA), matvec = responseMat)
eigenvalues = eigs(RGhyperM, k=scaleN, which='LM')

# Scaling dimensions from eigenvalue ratios
scDims = -np.log2(abs(eigenvalues/eigenvalues[0]))
```

## File Structure

```
analysisCodes/
├── HOTRG.py          # Main GILT+HOTRG implementation (33KB)
│   ├── oneHOTRG()         - Single HOTRG step with GILT
│   ├── normFlowHOTRG()    - Generate full RG flow
│   ├── fixBestSign()      - Fix sign ambiguity at fixed point
│   └── diffGiltHOTRG()    - JAX autodiff for scaling dimensions
├── gilts.py          # GILT implementation (48KB)
│   └── gilt_hotrgplaq()   - GILT on HOTRG plaquette
├── Isings.py         # 2D Ising tensor construction
├── jncon.py          # JAX version of ncon for autodiff
├── hotrgTc.py        # Find critical temperature (bisection)
├── hotrgFlow.py      # Generate RG flow at Tc
├── hotrgScale.py     # Compute scaling dimensions
├── drawRGflow.py     # Plot RG flows
└── drawScD.py        # Plot scaling dimensions
```

## Workflow

### 1. Find Critical Temperature
```bash
python hotrgTc.py --chi 20 --isGilt --isSym --Ngilt 2 --legcut 2 \
                  --gilteps 6e-5 --maxiter 31 --rootiter 12
```
Uses bisection to locate Tc where RG flow is marginal.

### 2. Generate RG Flow at Tc
```bash
python hotrgFlow.py --chi 20 --Ngilt 2 --legcut 2 --gilteps 6e-5 --maxiter 31
```
Saves tensors and isometries at each RG step.

### 3. Compute Scaling Dimensions
```bash
python hotrgScale.py --chi 20 --gilteps 6e-5 --Ngilt 2 --legcut 2 \
                     --iRGlow 5 --iRGhi 21
```
Linearizes RG equation and extracts scaling dimensions.

## Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `--chi` | Bond dimension | 20-40 |
| `--gilteps` | GILT truncation threshold | 6e-5 |
| `--Ngilt` | GILT iterations per HOTRG step | 2 |
| `--legcut` | Legs to truncate in GILT | 2 or 4 |
| `--maxiter` | Total RG steps | 31 |

## Dependencies

Original paper used old JAX versions:
```
jax==0.1.66        # OLD - needs update for modern JAX 0.4.x
jaxlib==0.1.46
```

### FASRC Cluster Setup (JAX GPU) - VERIFIED WORKING

```bash
# Load required modules
module load cuda/12.4.1-fasrc01
module load cudnn/9.1.1.17_cuda12-fasrc01

# Activate venv
source ~/.venvs/jax-gpu/bin/activate

# Verify GPU backend
python -c "import jax; print(jax.default_backend())"  # Should print 'gpu'
```

**Note:** Use NumPy 1.26.x (not 2.0+) for compatibility with tntools/abeliantensors.

**Test results (Jan 2026):**
- JAX 0.6.2 with GPU backend working
- Bisection for Tc runs in ~35 seconds (chi=10, 5 iterations)
- cuDNN version warnings are benign

Other dependencies:
```
ncon
tntools (git+https://github.com/mhauru/tntools)
abeliantensors    # Included locally with modifications
```

## Comparison with Our Existing Code

| Feature | tensorRGflow | EKR (ekrgilttrnr) | Our LoopTNR |
|---------|--------------|-------------------|-------------|
| Disentangler | GILT | GILT | Loop optimization |
| Coarse-graining | HOTRG | TRG | TRG |
| Autodiff | JAX | Manual | None |
| Scaling dims | Linearized RG | Transfer matrix | Transfer matrix |
| Language | Python | Python | Julia (TNRKit) |

## Potential Uses

1. **Validate scaling dimensions**: Compare JAX autodiff approach with transfer matrix method
2. **Adapt to φ⁴**: Replace `Isings.py` with φ⁴ tensor construction
3. **Port to Julia**: The algorithm could be implemented using Enzyme.jl for autodiff
4. **Benchmark GILT vs LoopTNR**: Compare disentangling effectiveness

## Notes

- Uses Z2 symmetry for efficiency (`--isSym` flag)
- Sign fixing is crucial for stable fixed points (`fixBestSign()`)
- The `abeliantensors` package is included locally with modifications for even truncation across Z2 sectors
- Runtime at chi=20: "a few minutes" for full analysis

## References

- GILT: [arXiv:1709.07460](https://arxiv.org/abs/1709.07460) (Hauru et al., 2018)
- HOTRG: [arXiv:1201.1144](https://arxiv.org/abs/1201.1144) (Xie et al., 2012)
- This work: [arXiv:2102.08136](https://arxiv.org/abs/2102.08136) (Lyu & Hauru, 2021)
