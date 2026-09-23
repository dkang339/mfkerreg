# Multifidelity Kernel Regression

A Python implementation of multifidelity kernel regression that combines high-fidelity and low-fidelity data to seek lower prediction error. This method adapts the nested correction used by Multifidelity Monte Carlo (MFMC) to kernel regression.

## Table of Contents

- [Motivation](#motivation)
- [Theory](#theory)
  - [Standard Kernel Regression](#standard-kernel-regression)
  - [Limitations of Standard Kernel Regression](#limitations-of-standard-kernel-regression)
  - [Multifidelity Setup](#multifidelity-setup)
  - [Multifidelity Kernel Regression](#multifidelity-kernel-regression)
  - [Asymptotic Unbiasedness](#asymptotic-unbiasedness)
  - [Population-Optimal Alpha](#population-optimal-alpha)
  - [Comparison with the Same Single-Fidelity Estimator](#comparison-with-the-same-single-fidelity-estimator)
  - [Further Theory](#further-theory)
- [Results](#results)
  - [Example 1: Exponential Function](#example-1-exponential-function)
  - [Example 2: NASA CRM Wing Stress Field](#example-2-nasa-crm-wing-stress-field)
  - [Example 3: Optimal and Naive Exoplanet Allocations](#example-3-optimal-and-naive-exoplanet-allocations)
- [Derivation Locations](#derivation-locations)
- [Installation](#installation)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Dependencies](#dependencies)
- [References](#references)

## Motivation

In many scientific and engineering applications, **high-fidelity data is expensive and scarce**. For example:
- High-resolution simulations require significant computational resources
- Physical experiments are costly and time-consuming
- Detailed measurements require expensive equipment

Standard kernel regression can have high prediction variance when high-fidelity
data is limited. This package incorporates cheaper low-fidelity data to reduce
that error when the correction is informative enough.

## Theory

### Standard Kernel Regression

Kernel regression (also known as Nadaraya-Watson regression) is a non-parametric method that estimates the conditional expectation of a random variable. Given:

- $X \in \mathbb{R}^d$: d-dimensional input variable
- $f_H: \mathbb{R}^d \to \mathbb{R}$: high-fidelity input-output map
- $Y_H=f_H(X)$: high-fidelity output variable
- $\lbrace (x_i,y_{H,i}) \rbrace_{i=1}^{n}$: $n$ independent, identically distributed training samples

For an unseen point $z$, kernel regression predicts the output as a **weighted average of training data**:

$$
\mathbb E[Y_H\mid X=z] \approx \sum_{i=1}^{n} w_i(z)y_{H,i}
$$

where the weights are computed using a kernel function $K_h$:

$$
w_i(z) = \frac{K_h(z-x_i)}{\sum_{j=1}^{n} K_h(z-x_j)}
$$

For a scalar bandwidth $h>0$ and an input displacement $u\in\mathbb R^d$,
the scaled kernel is $K_h(u)=h^{-d}K(u/h)$. In one dimension, $d=1$, so the
factor is $h^{-1}$. In $d$ dimensions, substituting $v=u/h$ gives
$du=h^d\,dv$. Hence, when $\int_{\mathbb R^d}K(v)\,dv=1$,

$$
\int_{\mathbb R^d}K_h(u)\,du
=\int_{\mathbb R^d}h^{-d}K(u/h)\,du
=\int_{\mathbb R^d}K(v)\,dv=1.
$$

The factor $h^{-d}$ cancels from the prediction weights $w_i(z)$, but this
normalization makes the expected kernel denominator approach the probability
density of the input $X$ at $z$ in the proof below.

For the standard consistency argument below, we use a nonnegative, symmetric
kernel that integrates to one. These are sufficient assumptions for that
argument, not requirements for every kernel regression estimator. The
denominator must be positive at the query for the displayed weights to exist.

When the denominator is positive, $\sum_{i=1}^{n} w_i = 1$.

### Limitations of Standard Kernel Regression

When high-fidelity data is limited:
- Predictions suffer from **high variance**
- The estimator is sensitive to the particular training samples drawn
- Accuracy degrades significantly in low-budget regimes

### Multifidelity Setup

To address these limitations, we introduce a **multifidelity framework**:

- $f_L: \mathbb{R}^d \to \mathbb{R}$: low-fidelity input-output map (cheap to evaluate)
- $n$: number of high-fidelity samples
- $m\ge n$: total number of low-fidelity samples, including the $n$ samples paired with high-fidelity observations
- $\alpha$: correction coefficient, chosen for a stated prediction-risk objective

Kernel regression estimates a conditional mean with local weights. The LF
correction follows the nested-sample structure of the MFMC mean estimator.

### Multifidelity Kernel Regression

The MFMC estimator for mean estimation is:

$$
\mathbb E[f_H(X)] \approx \frac{1}{n}\sum_{i=1}^{n} f_H(x_i) + \alpha \left( \frac{1}{m}\sum_{i=1}^{m} f_L(x_i) - \frac{1}{n}\sum_{i=1}^{n} f_L(x_i) \right)
$$

Applying this framework to kernel regression:

$$
\mathbb E[Y_H\mid X=z] \approx \sum_{i=1}^{n} w^{(H)}_{i,n}(z)y_{H,i} + \alpha \left( \sum_{i=1}^{m} w^{(L)}_{i,m}(z)y_{L,i} - \sum_{i=1}^{n} w^{(L)}_{i,n}(z)y_{L,i} \right)
$$

where:
- $w^{(H)}_{i,n}$ uses the HF bandwidth and the $n$ paired inputs
- $w^{(L)}_{i,k}$ uses the LF bandwidth and the first $k$ inputs, for $k=n,m$
- $y_{H,i}$ and $y_{L,i}$ are the HF and LF outputs at the $i$th input
- $\alpha$ is a coefficient whose optimum depends on the chosen objective

The first $n$ input locations are shared by both fidelities, and the $n$
paired LF observations are also included in the $m$-sample LF predictor.
Ordinary MFMC sample means have an exactly zero-mean LF correction. Kernel
predictors are ratios with random denominators, so the two LF predictors can
have different finite-sample expectations even when they use the same
bandwidth. Consequently, the kernel correction need not have zero expectation
at finite sample counts.

### Asymptotic Unbiasedness

Fix a query input $z\in\mathbb R^d$. Let $X$ be the random input, let
$Y_H=f_H(X)$ and $Y_L=f_L(X)$ be its HF and LF outputs. Since these
simulators are deterministic, their conditional means at $z$ are $f_H(z)$
and $f_L(z)$.

The first $n$ inputs have paired HF and LF outputs. LF outputs are available
at $m\ge n$ inputs in total. For a fixed finite coefficient $\alpha$, the MF
kernel predictor is

$$
\widehat\mu_{\mathrm{MF},n,m,\alpha}(z)
=\widehat\mu_{H,n}(z)
+\alpha\left(\widehat\mu_{L,m}(z)-\widehat\mu_{L,n}(z)\right).
$$

Here $\widehat\mu_{H,n}$ uses $n$ HF outputs, while
$\widehat\mu_{L,n}$ and $\widehat\mu_{L,m}$ use the first $n$ and first
$m$ LF outputs with the same LF bandwidth. The target is $f_H(z)$.

#### Conditions and the role of the bandwidth

Let $p_X$ be the probability density function of $X$. Assume that
$p_X(z)>0$ and that $p_X$ is continuous at $z$.
Assume $X_1,\ldots,X_m$ are independent and identically distributed; the
HF and LF outputs share each of the first $n$ inputs. For this short
deterministic-simulator proof, assume that the inputs lie in a compact set
$\mathcal X\subset\mathbb R^d$ and that $f_H$ and $f_L$ are continuous on
$\mathcal X$. A continuous function on a compact set has a finite maximum,
so there is a finite constant $M$ such that

$$
|f_H(x)|\le M,\qquad |f_L(x)|\le M,
\qquad x\in\mathcal X.
$$

Compact input support is a simple sufficient condition for the expectation
step, not a requirement of kernel regression in general. Use a positive
Gaussian kernel $K$. Let $h_H$ and
$h_L$ be the HF and LF bandwidths, and require, as $n\to\infty$ with
$m\ge n$,

$$
h_H\to0,\qquad h_L\to0,\qquad
nh_H^d\to\infty,\qquad nh_L^d\to\infty.
$$

Because $m\ge n$, the last condition also gives $mh_L^d\to\infty$.
Shrinking bandwidths remove smoothing bias at $z$. The products
$nh_H^d$ and $nh_L^d$ control sampling variation near $z$. In one input
dimension, $d=1$, so these products are $nh_H$ and $nh_L$.

#### Why the three kernel predictors converge

For $K_h(u)=h^{-d}K(u/h)$, write the three predictors directly as ratios:

$$
\widehat\mu_{H,n}(z)
=\frac{n^{-1}\sum_{i=1}^nK_{h_H}(z-X_i)Y_{H,i}}
{n^{-1}\sum_{i=1}^nK_{h_H}(z-X_i)},
$$

$$
\widehat\mu_{L,k}(z)
=\frac{k^{-1}\sum_{i=1}^kK_{h_L}(z-X_i)Y_{L,i}}
{k^{-1}\sum_{i=1}^kK_{h_L}(z-X_i)},\qquad k\in\{n,m\}.
$$

The preceding calculation established that $K_{h_H}$ integrates to one.
In the deterministic simulator considered here, $Y_H=f_H(X)$. The query
$z$, bandwidth $h_H$, and function $K$ are fixed. For each possible input
value $x$, the product has value $K_{h_H}(z-x)f_H(x)$. Averaging over $X$
with its density $p_X$ gives the first integral below. Then substitute
$u=(z-x)/h_H$, so $x=z-h_Hu$ and the volume element contributes $h_H^d$.
For each fixed $u$, $h_H\to0$ implies $z-h_Hu\to z$. The following
equalities first perform the substitution; the limit is evaluated next.

$$
\begin{aligned}
\mathbb E[K_{h_H}(z-X)Y_H]
&=\mathbb E[K_{h_H}(z-X)f_H(X)]\\
&=\int_{\mathbb R^d}K_{h_H}(z-x)f_H(x)p_X(x)\,dx\\
&=\int_{\mathbb R^d}K(u)
f_H(z-h_Hu)p_X(z-h_Hu)\,du
\end{aligned}
$$

Continuity of $f_H$ and $p_X$ at $z$ gives
$f_H(z-h_Hu)p_X(z-h_Hu)\to f_H(z)p_X(z)$. The Gaussian tail controls the
part of the integral far from $z$, while compact input support and
continuity of $f_H$ make $f_Hp_X$ integrable. These conditions justify
passing the limit through the integral:

$$
\begin{aligned}
\lim_{h_H\to0}\int K(u)f_H(z-h_Hu)p_X(z-h_Hu)\,du
&=\int K(u)\lim_{h_H\to0}
\bigl[f_H(z-h_Hu)p_X(z-h_Hu)\bigr]\,du\\
&=f_H(z)p_X(z)\underbrace{\int K(u)\,du}_{=1}\\
&=f_H(z)p_X(z).
\end{aligned}
$$

Applying the same substitution to the other three terms
gives

$$
\begin{aligned}
\mathbb E[K_{h_H}(z-X)]
&=\int K(u)p_X(z-h_Hu)\,du\to p_X(z),\\
\mathbb E[K_{h_L}(z-X)Y_L]
&=\int K(u)f_L(z-h_Lu)p_X(z-h_Lu)\,du
\to f_L(z)p_X(z),\\
\mathbb E[K_{h_L}(z-X)]
&=\int K(u)p_X(z-h_Lu)\,du\to p_X(z).
\end{aligned}
$$

These are limits of the numerator and denominator expectations, not yet a
limit for the expectation of their ratio. The sample averages must also
approach their expectations. Squaring the scaled kernel and making the same
substitution gives

$$
\mathbb E[K_h(z-X)^2]
=h^{-d}\int_{\mathbb R^d}K(u)^2p_X(z-hu)\,du
=O(h^{-d}).
$$

The factor $h^{-d}$ remains because squaring $K_h=h^{-d}K(\cdot/h)$
produces $h^{-2d}$, while the change of variables contributes $h^d$.
Independence across observations and $|Y_H|\le M$ now give

$$
\begin{aligned}
\operatorname{Var}\!\left(
\frac1n\sum_{i=1}^n K_{h_H}(z-X_i)Y_{H,i}\right)
&=\frac1n\operatorname{Var}(K_{h_H}(z-X)Y_H)\\
&\le\frac{M^2}{n}\mathbb E[K_{h_H}(z-X)^2]\\
&=O((nh_H^d)^{-1}).
\end{aligned}
$$

The HF denominator average has the same order, without the factor $M^2$.
For each LF count $k\in\{n,m\}$, the same calculation gives

$$
\operatorname{Var}\!\left(
\frac1k\sum_{i=1}^kK_{h_L}(z-X_i)Y_{L,i}\right)
\le\frac{M^2}{k}\mathbb E[K_{h_L}(z-X)^2]
=O((kh_L^d)^{-1}).
$$

The LF denominator average again has the same order without $M^2$.
The notation $O(a)$ means that a quantity is at most a constant multiple
of $a$ for sufficiently large sample counts. Thus
$nh_H^d,nh_L^d\to\infty$ makes the HF and $n$-sample LF fluctuations
vanish; $m\ge n$ gives the same result for the $m$-sample LF averages.
Since the denominator limit $p_X(z)$ is positive, the ratios satisfy

$$
\widehat\mu_{H,n}(z)\xrightarrow{p}f_H(z),\qquad
\widehat\mu_{L,n}(z)\xrightarrow{p}f_L(z),\qquad
\widehat\mu_{L,m}(z)\xrightarrow{p}f_L(z).
$$

The symbol $\xrightarrow{p}$ means convergence in probability. The
condition $nh^d\to\infty$ is sufficient for this variance argument; it is
not necessary for every possible kernel-regression setting.

#### From predictor convergence to expectation convergence

Probability convergence alone does not guarantee expectation convergence.
Here, however, every kernel prediction is a weighted average of simulator
outputs. The weights are nonnegative and sum to one. Since all HF and LF
outputs on $\mathcal X$ lie between $-M$ and $M$,

$$
|\widehat\mu_{H,n}(z)|\le M,\qquad
|\widehat\mu_{L,n}(z)|\le M,\qquad
|\widehat\mu_{L,m}(z)|\le M.
$$

For example, on the event
$|\widehat\mu_{H,n}(z)-f_H(z)|\le\varepsilon$, the HF prediction error is
at most $\varepsilon$. On the remaining event, its magnitude is at most
$2M$. Hence

$$
\mathbb E[|\widehat\mu_{H,n}(z)-f_H(z)|]
\le\varepsilon+2M\,
\mathbb P(|\widehat\mu_{H,n}(z)-f_H(z)|>\varepsilon).
$$

The probability on the right tends to zero by the preceding convergence
result. Because $\varepsilon>0$ can be arbitrarily small, the expected
absolute HF prediction error tends to zero. The same argument applies to
the two LF predictions. Thus their expectations converge:

$$
\mathbb E[\widehat\mu_{H,n}(z)]\to f_H(z),\qquad
\mathbb E[\widehat\mu_{L,n}(z)]\to f_L(z),\qquad
\mathbb E[\widehat\mu_{L,m}(z)]\to f_L(z).
$$

The two LF limits cancel visibly:

$$
\begin{aligned}
\lim_{n\to\infty}\mathbb E[
\widehat\mu_{\mathrm{MF},n,m,\alpha}(z)]
&=f_H(z)+\alpha\bigl(\cancel{f_L(z)}-\cancel{f_L(z)}\bigr)\\
&=f_H(z).
\end{aligned}
$$

Therefore the MF predictor is **asymptotically unbiased** for $f_H(z)$
under these conditions. The finite-sample LF correction need not have zero
expectation. If $\alpha$ changes with the sample counts, the product of
$\alpha$ and the LF expectation difference must still tend to zero; the
fixed-$\alpha$ proof alone does not establish that property for an
allocation-dependent coefficient.

### Population-Optimal Alpha

For fixed $n$, $m\ge n$, and $z$, consider the population MSE relative to
the HF conditional mean $f_H(z)$:

$$
R(\alpha;z)=\mathbb E\left[
\left(\widehat\mu_{H,n}(z)-f_H(z)
+\alpha\bigl(\widehat\mu_{L,m}(z)-\widehat\mu_{L,n}(z)\bigr)\right)^2
\right].
$$

The expectation averages over random training sets. Assume that the squared
HF prediction error and squared LF correction have finite expectations.
Expanding the square gives

$$
\begin{aligned}
R(\alpha;z)
&=\mathbb E[(\widehat\mu_{H,n}(z)-f_H(z))^2]\\
&\quad+2\alpha\,\mathbb E[(\widehat\mu_{H,n}(z)-f_H(z))
(\widehat\mu_{L,m}(z)-\widehat\mu_{L,n}(z))]\\
&\quad+\alpha^2\mathbb E[(\widehat\mu_{L,m}(z)
-\widehat\mu_{L,n}(z))^2].
\end{aligned}
$$

If the final expectation is positive, differentiate with respect to
$\alpha$ and set the derivative to zero:

$$
\boxed{
\alpha_{\mathrm{MSE}}^*(n,m;z)
=-\frac{
\mathbb E[(\widehat\mu_{H,n}(z)-f_H(z))
(\widehat\mu_{L,m}(z)-\widehat\mu_{L,n}(z))]
}{
\mathbb E[(\widehat\mu_{L,m}(z)-\widehat\mu_{L,n}(z))^2]
}.}
$$

The second derivative is twice the positive denominator, so this is the
unique population-MSE minimizer. Its moments are generally unknown and
must be estimated in an implementation.

### Comparison with the Same Single-Fidelity Estimator

The SF comparator is exactly $\widehat\mu_{H,n}(z)$, using the same $n$
HF observations as the MF predictor. Setting $\alpha=0$ in $R(\alpha;z)$
gives its MSE. Substituting $\alpha_{\mathrm{MSE}}^*$ gives

$$
\begin{aligned}
R(\alpha_{\mathrm{MSE}}^*;z)
&=R(0;z)\\
&\quad-\frac{
\mathbb E[(\widehat\mu_{H,n}(z)-f_H(z))
(\widehat\mu_{L,m}(z)-\widehat\mu_{L,n}(z))]^2
}{
\mathbb E[(\widehat\mu_{L,m}(z)-\widehat\mu_{L,n}(z))^2]
}\\
&\le R(0;z).
\end{aligned}
$$

The subtracted quantity is nonnegative. It is positive if the expectation
in its numerator is nonzero. If the LF correction has zero second moment,
the correction is zero almost surely and MF equals SF. This guarantee
concerns expected squared error over random training sets, not every
realized training set. An estimated coefficient does not automatically
preserve the guarantee. The [detailed proof](docs/current_scope_mf_vs_sf.md)
also covers coefficient-estimation error and field-valued outputs.

### Further Theory

The [allocation derivation](docs/future_optimal_allocation.md) studies the
first-order variance approximation and selection of sample counts. The
[previous README](docs/readme_archive/README_2026-09-23.md) retains the
longer appendix formerly included here.

## Results

### Example 1: Exponential Function

**Setup:**
- High-fidelity: $f_1(x) = e^x$
- Low-fidelity: $f_2(x) = 0.9e^{0.5x}$
- Input distribution: $x \sim \mathcal{U}(0, 5)$
- Correlation coefficient: 0.97
- Model evaluation cost (artificial): [1, 0.001]

**High-fidelity and Low-fidelity Functions:**

![Exponential Functions](examples/exponential/plots/functions.png)

**Mean Squared Error Comparison:**

The plotted experiment shows lower empirical MSE for multifidelity kernel
regression at the tested budgets. The plot is an empirical comparison, not a
general MSE guarantee.

![Exponential MSE](examples/exponential/plots/mfkr_mse.png)

| Computational Budget | High-fidelity Samples ($n$) | Low-fidelity Samples ($m$) |
|---------------------|----------------------------|---------------------------|
| 10                  | 8                          | 1,126                     |
| 100                 | 88                         | 11,263                    |

### Example 2: NASA CRM Wing Stress Field

**Setup:**
- Input: NASA Common Research Model (CRM) wing design parameters
- High-fidelity output: CRM wing von Mises stress field
- Low-fidelity output: coarse-grid wing stress field
- Number of high-fidelity training samples: 100
- Number of low-fidelity training samples: 300
- Number of test samples: 100

In this example, the multifidelity kernel regression predicts POD coefficients
for unseen input points, then reconstructs the predicted stress field with the
high-fidelity POD basis.
To train the model, we use the MAROM framework from Perron et al. The method applies POD separately
to the high- and low-fidelity datasets and retains the same number of POD modes. It then
uses manifold alignment to map the low-fidelity POD coefficients into the high-fidelity
POD coefficient space.

**Stress Prediction and Pointwise Absolute Error:**

![CRM wing stress field comparison](examples/wing/results/mfkr_field_lowfi_grid.png)

The absolute error distribution shows that the multifidelity prediction has
lower error than the single-fidelity prediction across the wing field.

For the coarse-grid low-fidelity data, multifidelity regression reduced the
mean relative L2 error from 4.04% with single-fidelity kernel regression to
2.34%. The worst-case relative L2 error also dropped from 19.55% to 8.80%.

| Method | Training Samples | Mean Relative L2 Error | Max Relative L2 Error |
|--------|------------------|------------------------|-----------------------|
| Single-fidelity KR | N_HF=100 | 4.04% | 19.55% |
| Multifidelity KR | N_HF=100, N_LF=300 | 2.34% | 8.80% |

### Example 3: Optimal and Naive UM/ExoPlaSim Allocations

This experiment uses UM as high fidelity and ExoPlaSim as low fidelity. The
data contain 197 exact UM/ExoPlaSim input pairs and 803 additional ExoPlaSim
cases. A deterministic split made with seed `20260910` assigns 100 pairs to an
independent pilot set and the remaining 97 pairs to the training candidate
pool. The 30 target-domain UM cases without generated ExoPlaSim pairs form the
fixed test set. Pilot cases are excluded from every training pool.

The pilot data fit a five-mode UM POD representation, a five-mode ExoPlaSim POD
representation, their scaled orthogonal Procrustes alignment, common LF kernel
bandwidths, and the query-dependent influence coefficient. Training draws use
bootstrap sampling with replacement because some requested allocations exceed
the 97 distinct paired training candidates. Every policy uses the same test
cases, pilot estimates, and matched bootstrap streams. The assumed costs are
$c_H=1$ for one UM observation and $c_L=0.1$ for one ExoPlaSim observation.

The derived allocation is compared with $m=2n$, $m=4n$, $m=6n$, and $m=8n$.
For a naive ratio $r$, the experiment uses

$$
n=\left\lfloor\frac{P}{c_H+rc_L}\right\rfloor,
\qquad m=rn,
$$

so every policy respects the same nominal cost bound. A same-budget
single-fidelity reference spends the complete budget on UM observations.

Three fields were evaluated over budgets 10, 15, 20, 30, 40, 50, 75, 100,
150, 200, 300, and 400 with 30 matched repetitions:

- `surface_temperature`: the pilot influence correlation is weak. The optimal
  rule selects $m/n$ close to one. Single-fidelity UM has lower observed RMSE
  than every tested MF rule.
- `cloud_fraction_8`: the pilot influence correlation is also weak. The optimal
  rule again selects $m/n$ close to one. Its observed mean RMSE is slightly
  below single-fidelity UM at budgets 40, 300, and 400. The fixed ratios are
  unstable at many smaller budgets.
- `olr_cloudy`: the pilot influence correlation is stronger. The optimal rule
  selects $m/n$ near 2.5 to 2.8, but fixed ratio 2 has slightly lower mean RMSE
  at most budgets. Single-fidelity UM still has the lowest mean RMSE at every
  tested budget.

These results do not show uniform MFKR dominance over SFKR for the present data
split and pilot-fitted coefficient. The allocation formula minimizes an
estimated leading-variance objective. The experiment measures finite-sample
field RMSE, which also includes smoothing bias, influence-moment estimation
error, POD truncation error, and behavior caused by resampling a limited paired
pool.

![Surface-temperature allocation comparison](examples/exoplanet/results/optimal_vs_naive_um_exoplasim/surface_temperature_optimal_vs_naive_rmse.png)

![Cloud-fraction-8 allocation comparison](examples/exoplanet/results/optimal_vs_naive_um_exoplasim/cloud_fraction_8_optimal_vs_naive_rmse.png)

![Cloudy-OLR allocation comparison](examples/exoplanet/results/optimal_vs_naive_um_exoplasim/olr_cloudy_optimal_vs_naive_rmse.png)

Run the following command from the repository root:

```bash
.venv/bin/python examples/exoplanet/compare_optimal_naive_allocations.py \
  --field-name surface_temperature \
  --budgets 10 15 20 30 40 50 75 100 150 200 300 400 \
  --ratios 2 4 6 8 \
  --repeats 30 \
  --pilot-samples 100 \
  --output-dir examples/exoplanet/results/optimal_vs_naive_um_exoplasim
```

Replace `surface_temperature` with `cloud_fraction_8` or `olr_cloudy` to run
the other fields. The output directory is
`examples/exoplanet/results/optimal_vs_naive_um_exoplasim`. Each field has an
HTML report, a PNG figure, a JSON summary, and an NPZ archive containing the
complete per-repeat results and all pilot, training, low-fidelity, and test IDs.
The directory-level `README.md` and `index.html` summarize all three fields.

#### Alpha estimated from all available pairs

A second experiment changes only the sample used to estimate the influence
moments and alpha. POD fitting, bandwidth fitting, the 97-pair training pool,
the 900-case LF pool, the test set, and random streams remain fixed. The first
experiment estimates alpha from the 100 independent pilot pairs. The second
experiment estimates alpha from all 197 available UM/ExoPlaSim pairs. Thus, the
second statistics set overlaps all 97 training candidates and is not an
independent alpha evaluation.

Using all 197 pairs did not uniformly improve performance. For surface
temperature, nearly zero estimated local LF influence variances produced
extreme alpha values and severe numerical instability. For cloud fraction
level 8, results changed irregularly across budgets. For cloudy OLR, all-pairs
alpha reduced optimal-allocation mean RMSE modestly at most budgets and beat
the same-budget SF mean at budget 400.

The combined comparison is in
`examples/exoplanet/results/alpha_statistics_pilot100_vs_all197/index.html`.
Run the all-pairs experiment by adding `--statistics-mode all-pairs`, then run
`examples/exoplanet/compare_alpha_statistics_modes.py` to regenerate the
comparison report.

## Derivation Locations

The mathematical material is distributed as follows:

- [`docs/current_scope_mf_vs_sf.md`](docs/current_scope_mf_vs_sf.md) is the
  canonical current-study proof file. It gives the exact same-HF-count MSE
  result, estimated-coefficient penalty, and weighted-field extension.
- [`docs/future_optimal_allocation.md`](docs/future_optimal_allocation.md) is the
  canonical future-study allocation file. It distinguishes exact full variance
  from exact first-order variance, identifies where Taylor expansion is applied,
  derives the continuous and integer allocation, and records the questions and
  answers that motivated the clarification.
- [`optimal_mfkr_experiment/README.md`](optimal_mfkr_experiment/README.md)
  gives a shorter statement of the influence formulas, assumptions, effective
  local sample diagnostic, and controlled synthetic results.
- [`optimal_mfkr_experiment/optimal_mfkr.py`](optimal_mfkr_experiment/optimal_mfkr.py)
  implements the influence values, local covariance statistics, field-level
  constants $A$ and $B$, continuous solution, and exhaustive feasible-integer
  allocation.
- [`optimal_mfkr_experiment/mse_optimal_mfkr.py`](optimal_mfkr_experiment/mse_optimal_mfkr.py)
  implements the finite-sample MSE coefficient
  $\alpha=-\mathbb E[eD]/\mathbb E[D^2]$ and candidate allocation helpers.
- [`mfkerreg_field_discussion.md`](mfkerreg_field_discussion.md) explains the
  consistency conditions, direct field prediction, field-valued risk, and why
  finite-sample kernel corrections are not generally exactly unbiased.
- [`mfkernel.pdf`](mfkernel.pdf) contains the original Nadaraya-Watson derivation,
  raw MFMC motivation, and early numerical examples. The PDF predates the
  influence-based allocation derivation.
- [`docs/readme_archive/README_2026-09-23.md`](docs/readme_archive/README_2026-09-23.md)
  preserves the README before the present revision.

## Installation

```bash
# Clone the repository
git clone https://github.com/dkang339/mfkerreg.git
cd mfkerreg

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

## Prerequisites

Download data for the wing example.

Download the wing structural stress data from [here](https://link.springer.com/article/10.1007/s00158-022-03274-1#Sec23) (3.9 GB), or by running:

```bash
wget https://static-content.springer.com/esm/art%3A10.1007%2Fs00158-022-03274-1/MediaObjects/158_2022_3274_MOESM1_ESM.zip
```

Place the following data into `data/wing`:

```text
/data/crm_baseline_4DV_N1000_slim.h5
/data/crm_coarse-grid_4DV_N1000_slim.h5
/data/crm_coarse-ribs_4DV_N1000_slim.h5
```

Preprocess the data.

For each example, run the script beginning with `preproc_` to generate or format the data.

## Usage

```bash
# Run exponential function example
python examples/exponential/mfkr.py

# Run CRM wing stress-field example
python examples/wing/preproc_wing.py
python examples/wing/mfkr_field.py --low-fidelity grid
```

## Project Structure

```
mfkerreg/
├── src/
│   ├── kr.py          # Kernel regression implementation
│   ├── kernels.py     # Kernel functions (Matern, RBF, etc.)
│   ├── mfmc.py        # MFMC sample allocation
│   └── utils.py       # Utility functions
├── examples/
│   ├── exponential/   # 1D exponential function example
│   ├── wing/          # CRM wing stress-field example
│   └── exoplanet/     # UM and ExoPlaSim field experiments
├── optimal_mfkr_experiment/ # Influence-based allocation experiments
├── docs/              # Proofs and archived README
├── data/              # Precomputed statistics
├── README.md          # Current repository overview
└── mfkernel.pdf       # Theory document
```

## Key Features

- **Variance Reduction**: Leverages cheap low-fidelity data to reduce prediction variance
- **Extended Allocation Study**: Includes an experimental first-order allocation method whose assumptions and approximation are documented separately from the current MFKR dominance claim
- **Asymptotic Unbiasedness**: The theory section gives explicit bandwidth, local-sample, density, and compact-domain conditions under which the expected prediction with a fixed finite coefficient converges to the HF conditional mean
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
- NASA Common Research Model: https://commonresearchmodel.larc.nasa.gov/
- Perron, C., Sarojini, D., Rajaram, D., Corman, J., & Mavris, D. N. (2022). Manifold alignment-based multi-fidelity reduced-order modeling applied to structural analysis. *Structural and Multidisciplinary Optimization*, 65, 236. https://doi.org/10.1007/s00158-022-03274-1
