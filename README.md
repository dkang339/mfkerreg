# Multifidelity Kernel Regression

A Python implementation of multifidelity kernel regression that combines high-fidelity and low-fidelity data to achieve variance reduction in predictions. This method applies the Multifidelity Monte Carlo (MFMC) estimator framework to kernel regression problems.

## Motivation

In many scientific and engineering applications, **high-fidelity data is expensive and scarce**. For example:
- High-resolution simulations require significant computational resources
- Physical experiments are costly and time-consuming
- Detailed measurements require expensive equipment

Standard kernel regression relies solely on high-fidelity data, leading to **high variance in predictions** when data is limited. This package addresses this limitation by incorporating cheaper low-fidelity data to improve prediction robustness.

## Theory

### Standard Kernel Regression

Kernel regression (also known as Nadaraya-Watson regression) is a non-parametric method that estimates the conditional expectation of a random variable. Given:

- X in R^d: d-dimensional input variable
- f₁: R^d → R: high-fidelity input-output map
- Y₁ = f₁(X): high-fidelity output variable
- {(xᵢ, yᵢ)}ᵢ₌₁ⁿ: n i.i.d. training samples

For an unseen point x*, kernel regression predicts the output as a **weighted average of training data**:

```
E[Y₁|X=x*] ≈ Σᵢ wᵢ(x*) yᵢ
```

where the weights are computed using a kernel function Kₕ:

```
wᵢ(x*) = Kₕ(x* - xᵢ) / Σⱼ Kₕ(x* - xⱼ)
```

The kernel function Kₕ(·) = (1/h)K(·/h) must satisfy:
1. Non-negativity for all inputs
2. Integration to 1
3. Symmetry

Note that Σᵢ wᵢ = 1, making this a proper weighted average.

### Limitations of Standard Kernel Regression

When high-fidelity data is limited:
- Predictions suffer from **high variance**
- The estimator is sensitive to the particular training samples drawn
- Accuracy degrades significantly in low-budget regimes

### Multifidelity Setup

To address these limitations, we introduce a **multifidelity framework**:

- f₂: R^d → R: low-fidelity input-output map (cheap to evaluate)
- n: number of high-fidelity samples
- m (m >> n): number of low-fidelity samples
- α: optimal weight derived from correlation between fidelities

The key insight is that both kernel regression and the MFMC estimator are **mean estimation methods**, allowing us to combine them naturally.

### Multifidelity Kernel Regression

The MFMC estimator for mean estimation is:

```
E[f₁(X)] ≈ (1/n)Σᵢⁿ f₁(xᵢ) + α((1/m)Σᵢᵐ f₂(xᵢ) - (1/n)Σᵢⁿ f₂(xᵢ))
```

Applying this framework to kernel regression:

```
E[Y₁|X=x*] ≈ Σᵢⁿ wᵢ,ₙ(x*) yᵢ⁽¹⁾ + α(Σᵢᵐ wᵢ,ₘ(x*) yᵢ⁽²⁾ - Σᵢⁿ wᵢ,ₙ(x*) yᵢ⁽²⁾)
```

where:
- wᵢ,ₙ are weights computed using the n high-fidelity samples
- wᵢ,ₘ are weights computed using the m low-fidelity samples
- α is optimized to minimize variance

The estimator remains **unbiased** since the low-fidelity correction term has zero expectation. For the detailed derivation of unbiasedness, see [mfkernel.pdf](mfkernel.pdf).

## Results

### Example 1: Exponential Function

**Setup:**
- High-fidelity: f₁(x) = eˣ
- Low-fidelity: f₂(x) = 0.9e^(0.5x)
- Input distribution: x ~ U(0, 5)
- Correlation coefficient: 0.97
- Cost ratio: [1, 0.001]

**High-fidelity and Low-fidelity Functions:**

![Exponential Functions](examples/exponential/plots/functions.png)

**Mean Squared Error Comparison:**

Multifidelity kernel regression achieves significantly lower MSE and variance, especially in low-budget regimes.

![Exponential MSE](examples/exponential/plots/mfkr_mse.png)

| Computational Budget | High-fidelity Samples (n) | Low-fidelity Samples (m) |
|---------------------|---------------------------|--------------------------|
| 10                  | 8                         | 1,126                    |
| 100                 | 88                        | 11,263                   |

### Example 2: Ackley Function (2D)

**Setup:**
- High-fidelity: Standard Ackley function
- Low-fidelity: Modified Ackley with different parameters
- Input distribution: x ~ U(-32.768, 32.768)²
- Correlation coefficient: 0.76
- Cost ratio: [1, 0.001]

**High-fidelity and Low-fidelity Functions:**

![Ackley Functions](examples/forrester/plots/functions.png)

**Prediction Comparison (Budget = 50):**

| Method | Samples | MSE |
|--------|---------|-----|
| Single-fidelity KR | n=50 | 4.21 |
| Multifidelity KR | n=48, m=1,804 | 1.77 |

![Ackley Prediction](examples/forrester/plots/mfkr_pred.png)

**Mean Squared Error Comparison:**

Multifidelity kernel regression consistently outperforms single-fidelity kernel regression across all computational budgets. The variance reduction is more significant in the low-budget regime, where high-fidelity data is most scarce.

![Ackley MSE](examples/forrester/plots/mfkr_mse.png)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mfkerreg.git
cd mfkerreg

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

## Usage

```bash
# Run exponential function example
python examples/exponential/mfkr.py

# Run Ackley function example
python examples/forrester/mfkr.py
```

## Project Structure

```
mfkerreg/
├── src/
│   ├── kr.py          # Kernel regression implementation
│   ├── kernels.py     # Kernel functions (Matern, RBF, etc.)
│   ├── mfmc.py        # MFMC sample allocation
│   └── aux.py         # Auxiliary functions
├── examples/
│   ├── exponential/   # 1D exponential function example
│   └── forrester/     # 2D Ackley function example
├── data/              # Precomputed statistics
└── mfkernel.pdf       # Theory document
```

## Key Features

- **Variance Reduction**: Leverages cheap low-fidelity data to reduce prediction variance
- **Optimal Sample Allocation**: Automatically determines the optimal number of samples at each fidelity level
- **Unbiased Estimator**: Maintains unbiasedness while reducing variance
- **Kernels**: Supports ARD Matern 3/2 and Exponential kernel functions
- **Parallel Computation**: Uses joblib for parallel cross-validation

## Dependencies

- numpy >= 2.4.1
- scipy >= 1.17.0
- matplotlib >= 3.10.8
- joblib >= 1.5.3
- h5py >= 3.15.1

## References

- Peherstorfer, B., Willcox, K., & Gunzburger, M. (2016). Optimal model management for multifidelity Monte Carlo estimation. *SIAM Journal on Scientific Computing*.
- Nadaraya, E. A. (1964). On estimating regression. *Theory of Probability & Its Applications*.
- Watson, G. S. (1964). Smooth regression analysis. *Sankhyā: The Indian Journal of Statistics*.
