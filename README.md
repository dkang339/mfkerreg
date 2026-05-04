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

- $X \in \mathbb{R}^d$: $d$-dimensional input variable
- $f^{(1)}: \mathbb{R}^d \to \mathbb{R}$: high-fidelity input-output map
- $Y^{(1)} = f^{(1)}(X)$: high-fidelity output variable
- $\{(x_i, y_i^{(1)})\}_{i=1}^n$: $n$ i.i.d. training samples

For an unseen point $x^*$, kernel regression predicts the output as a **weighted average of training data**:

$$E[Y^{(1)}|X=x^*] \approx \sum_{i=1}^{n} w_i(x^*) y_i^{(1)}$$

where the weights are computed using a kernel function $K_h$:

$$w_i(x^*) = \frac{K_h(x^* - x_i)}{\sum_{j=1}^{n} K_h(x^* - x_j)}$$

The kernel function $K_h(\cdot) = \frac{1}{h}K(\frac{\cdot}{h})$ must satisfy:
1. Non-negativity for all inputs
2. Integration to 1
3. Symmetry

Note that $\sum_{i=1}^{n} w_i = 1$, making this a proper weighted average.

### Limitations of Standard Kernel Regression

When high-fidelity data is limited:
- Predictions suffer from **high variance**
- The estimator is sensitive to the particular training samples drawn
- Accuracy degrades significantly in low-budget regimes

### Multifidelity Setup

To address these limitations, we introduce a **multifidelity framework**:

- $f^{(2)}: \mathbb{R}^d \to \mathbb{R}$: low-fidelity input-output map (cheap to evaluate)
- $n$: number of high-fidelity samples
- $m$ ($m \gg n$): number of low-fidelity samples
- $\alpha$: optimal weight derived from correlation between fidelities

The key insight is that both kernel regression and the MFMC estimator are **mean estimation methods**, allowing us to combine them naturally.

### Multifidelity Kernel Regression

The MFMC estimator for mean estimation is:

$$\mathbb{E}[f^{(1)}(X)] \approx \frac{1}{n}\sum_{i=1}^{n} f^{(1)}(x_i) + \alpha \left( \frac{1}{m}\sum_{i=1}^{m} f^{(2)}(x_i) - \frac{1}{n}\sum_{i=1}^{n} f^{(2)}(x_i) \right)$$

Applying this framework to kernel regression:

$$E[Y^{(1)}|X=x^*] \approx \sum_{i=1}^{n} w_{i,n}(x^*) y_i^{(1)} + \alpha \left( \sum_{i=1}^{m} w_{i,m}(x^*) y_i^{(2)} - \sum_{i=1}^{n} w_{i,n}(x^*) y_i^{(2)} \right)$$

where:
- $w_{i,n}$ are weights computed using the $n$ high-fidelity samples
- $w_{i,m}$ are weights computed using the $m$ low-fidelity samples
- $\alpha$ is optimized to minimize variance

The estimator remains **unbiased** since the low-fidelity correction term has zero expectation. For the detailed derivation of unbiasedness, see [mfkernel.pdf](mfkernel.pdf).

## Results

### Example 1: Exponential Function

**Setup:**
- High-fidelity: $f^{(1)}(x) = e^x$
- Low-fidelity: $f^{(2)}(x) = 0.9e^{0.5x}$
- Input distribution: $x \sim \mathcal{U}(0, 5)$
- Correlation coefficient: 0.97
- Cost ratio: [1, 0.001]

**High-fidelity and Low-fidelity Functions:**

![Exponential Functions](examples/exponential/plots/functions.png)

**Prediction Comparison:**

The plot below shows predictions from 50 replicates. Single-fidelity kernel regression (magenta) shows high variance, while multifidelity kernel regression (blue) produces more consistent predictions.

![Exponential Prediction](examples/exponential/plots/mfkr_pred.png)

**Mean Squared Error Comparison:**

Multifidelity kernel regression achieves significantly lower MSE and variance, especially in low-budget regimes.

![Exponential MSE](examples/exponential/plots/mfkr_mse.png)

| Computational Budget | High-fidelity Samples ($n$) | Low-fidelity Samples ($m$) |
|---------------------|----------------------------|---------------------------|
| 10                  | 8                          | 1,126                     |
| 100                 | 88                         | 11,263                    |

### Example 2: Ackley Function (2D)

**Setup:**
- High-fidelity: Standard Ackley function
- Low-fidelity: Modified Ackley with different parameters
- Input distribution: $x \sim \mathcal{U}(-32.768, 32.768)^2$
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

```python
from src.kr import eval_kr, eval_mfkr, find_sigma, find_mfsigma
from src.mfmc import alloc

# Define your high-fidelity and low-fidelity functions
f1 = lambda x: np.exp(x)  # high-fidelity
f2 = lambda x: 0.9 * np.sqrt(np.exp(x))  # low-fidelity

# Set cost ratio and correlation
w = np.array([1, 0.001])  # cost per fidelity
rho = np.array([1.0, 0.97, 0.0])  # correlation coefficients

# Allocate samples optimally
alpha, m, _ = alloc(std, rho, w, budget)

# Generate training data
X = np.random.uniform(0, 5, (m[-1], 1))
Y_hf = f1(X[:m[0]])  # high-fidelity outputs
Y_lf = f2(X[:m[1]])  # low-fidelity outputs

# Find optimal kernel bandwidth
sigma = find_mfsigma(X, Y, m)

# Predict at test points
y_pred = eval_mfkr(X_test, X, Y, sigma, m, alpha)
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
- **Flexible Kernels**: Supports ARD Matern and other kernel functions
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
