---
type: uni_general
course: "[[Machine learning]]"
status: 🟢 Active
order: Summary
date_added: 2026-06-21
---
# Topic 4 — Parameter Estimation (MLE, MAP, & Bayesian Estimation) (Class 03A & Recitation 03)

**MOC:** [[Machine learning MOC]]
**Course:** [[Machine learning]]

# Study Guide: Topic 4 — Parameter Estimation (MLE, MAP, & Bayesian Estimation) (Class 03A & Recitation 03)

## 1. Context: Moving from BDT to Parameter Estimation

In **Bayesian Decision Theory (BDT)**, we assumed full, perfect knowledge of the true joint data distribution $P(\mathbf{x}, y)$. In real-world machine learning, we only have access to an empirical training dataset $\mathcal{D} = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n\}$ sampled independent and identically distributed (i.i.d.) from an unknown distribution.

### The Curse of Dimensionality in Density Estimation
To estimate the probability density function $P(\mathbf{x})$ directly from data without making parametric assumptions (non-parametric estimation, e.g., histograms):
1. Suppose we partition each feature dimension into $k$ bins to estimate local densities.
2. For a $d$-dimensional feature space $\mathbf{x} \in \mathbb{R}^d$, the total number of bins (regions of space) scales exponentially as:
   $$\text{Total Bins} = k^d$$
3. To estimate the density in each bin with even a single sample on average, we require $n \propto k^d$ training samples. For example, if $k = 10$ and $d = 20$, we would need $10^{20}$ samples—vastly exceeding the number of atoms in the universe. 

In high dimensions, the data becomes extremely sparse, and non-parametric density estimation breaks down entirely.

### How Parametric Statistics Resolves the Curse
To make density estimation tractable in high dimensions, we transition to **Parametric Statistics**:
* Instead of estimating arbitrary densities, we assume the data follows a known probability distribution family (e.g., a Multivariate Gaussian) parameterized by a fixed vector $\boldsymbol{\theta}$.
* This reduces the parameter space dramatically. For instance, a Multivariate Gaussian is defined by a mean vector $\boldsymbol{\mu} \in \mathbb{R}^d$ and a covariance matrix $\Sigma \in \mathbb{R}^{d \times d}$, requiring only $d + \frac{d(d+1)}{2} = \mathcal{O}(d^2)$ parameters to estimate, rather than $\mathcal{O}(k^d)$. Our goal then becomes estimating the values of $\boldsymbol{\theta}$ directly from the data sample $\mathcal{D}$.

## 2. Paradigms of Parameter Estimation

|**Dimension / View**|**Maximum Likelihood Estimation (MLE)**|**Maximum A Posteriori (MAP)**|**Bayesian Parameter Estimation**|
|---|---|---|---|
|**Philosophical View**|Frequentist|Bayesian|Bayesian|
|**Nature of Parameters $\boldsymbol{\theta}$**|Fixed, unknown constant vector.|Random variables with a prior distribution.|Random variables with a prior distribution.|
|**Objective Strategy**|Maximize data likelihood.|Maximize posterior probability density.|Compute full posterior; integrate to get expected value.|
|**Output Type**|Point estimate ($\hat{\boldsymbol{\theta}}$).|Point estimate ($\hat{\boldsymbol{\theta}}$).|Complete posterior distribution or expected value.|

## 3. Maximum Likelihood Estimation (MLE)

### The Principle

Under the frequentist view, the parameter vector $\boldsymbol{\theta}$ is a fixed but unknown physical constant. The **Likelihood Function** $L(\boldsymbol{\theta}) = P(\mathcal{D} \mid \boldsymbol{\theta})$ represents the probability of observing our specific training dataset given a parameter state $\boldsymbol{\theta}$. Since the samples are independent and identically distributed (i.i.d.), the joint likelihood equals the product of individual sample likelihoods:

$$L(\boldsymbol{\theta}) = P(\mathcal{D} \mid \boldsymbol{\theta}) = \prod_{i=1}^n P(\mathbf{x}_i \mid \boldsymbol{\theta})$$

The MLE framework selects the point estimate $\hat{\boldsymbol{\theta}}_{\text{MLE}}$ that maximizes this probability:

$$\hat{\boldsymbol{\theta}}_{\text{MLE}} = \arg\max_{\boldsymbol{\theta}} \prod_{i=1}^n P(\mathbf{x}_i \mid \boldsymbol{\theta})$$

### The Log-Likelihood Transformation

Because multiplying many probabilities can lead to numerical underflow, and because taking derivatives of products is complex, we apply a monotonic natural logarithm transformation to create the **Log-Likelihood** function $\ell(\boldsymbol{\theta})$:

$$\ell(\boldsymbol{\theta}) = \ln L(\boldsymbol{\theta}) = \sum_{i=1}^n \ln P(\mathbf{x}_i \mid \boldsymbol{\theta})$$

Since the logarithm function is strictly increasing, maximizing $\ell(\boldsymbol{\theta})$ yields the exact same point solution as maximizing $L(\boldsymbol{\theta})$. To find $\hat{\boldsymbol{\theta}}_{\text{MLE}}$, we calculate the gradient with respect to $\boldsymbol{\theta}$, set it to zero, and solve:

$$\nabla_{\boldsymbol{\theta}} \ell(\boldsymbol{\theta}) = \mathbf{0}$$

## 4. Maximum A Posteriori (MAP)

### The Principle

The Bayesian view treats $\boldsymbol{\theta}$ as a random variable governed by a **Prior Probability Distribution** $P(\boldsymbol{\theta})$, which encodes our beliefs before observing any data. Using Bayes' Rule, we construct the **Posterior Distribution** $P(\boldsymbol{\theta} \mid \mathcal{D})$:

$$P(\boldsymbol{\theta} \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \boldsymbol{\theta})P(\boldsymbol{\theta})}{P(\mathcal{D})} = \frac{P(\mathcal{D} \mid \boldsymbol{\theta})P(\boldsymbol{\theta})}{\int_{\Omega} P(\mathcal{D} \mid \boldsymbol{\theta})P(\boldsymbol{\theta}) d\boldsymbol{\theta}}$$

The MAP framework looks for a single point estimate $\hat{\boldsymbol{\theta}}_{\text{MAP}}$ that maximizes this posterior density. Since the marginal evidence $P(\mathcal{D})$ is independent of $\boldsymbol{\theta}$, it acts as a constant normalization factor and can be omitted during optimization:

$$\hat{\boldsymbol{\theta}}_{\text{MAP}} = \arg\max_{\boldsymbol{\theta}} P(\boldsymbol{\theta} \mid \mathcal{D}) = \arg\max_{\boldsymbol{\theta}} \left[ \prod_{i=1}^n P(\mathbf{x}_i \mid \boldsymbol{\theta}) \right] P(\boldsymbol{\theta})$$

### The Log-Posterior Formulation

Taking the natural logarithm transforms the optimization problem into a sum of the log-likelihood and the log-prior:

$$\hat{\boldsymbol{\theta}}_{\text{MAP}} = \arg\max_{\boldsymbol{\theta}} \left[ \sum_{i=1}^n \ln P(\mathbf{x}_i \mid \boldsymbol{\theta}) + \ln P(\boldsymbol{\theta}) \right]$$

The log-prior term $\ln P(\boldsymbol{\theta})$ mathematically acts exactly like a **regularization penalty** in machine learning, constraining the parameters from overfitting to sample noise.

## 5. Pure Bayesian Parameter Estimation

Unlike MLE and MAP, which output a single point estimate, **Bayesian Parameter Estimation** computes the complete posterior probability distribution $P(\boldsymbol{\theta} \mid \mathcal{D})$.

To make a prediction for a new, unseen data point $\mathbf{x}^*$, we integrate over the entire parameter space $\Omega$. This technique incorporates parameter uncertainty directly into the final prediction by weighting each parameter state by its posterior probability:

$$P(\mathbf{x}^* \mid \mathcal{D}) = \int_{\Omega} P(\mathbf{x}^* \mid \boldsymbol{\theta}) P(\boldsymbol{\theta} \mid \mathcal{D}) d\boldsymbol{\theta}$$

If forced to select a single representative parameter value under a squared-error loss criterion ($\lambda(\hat{\boldsymbol{\theta}}, \boldsymbol{\theta}) = \|\hat{\boldsymbol{\theta}} - \boldsymbol{\theta}\|^2$), the optimal Bayesian estimator is the expected value (mean) of the posterior distribution:

$$\hat{\boldsymbol{\theta}}_{\text{Bayes}} = \mathbb{E}[\boldsymbol{\theta} \mid \mathcal{D}] = \int_{\Omega} \boldsymbol{\theta} P(\boldsymbol{\theta} \mid \mathcal{D}) d\boldsymbol{\theta}$$

## 6. Python / NumPy Implementation Snippet

This snippet demonstrates how to compute numerical MLE and MAP estimates for data generated by an Exponential distribution ($\lambda e^{-\lambda x}$):

```python
import numpy as np

def compute_exponential_mle(data):
    """
    Data is a 1D numpy array of i.i.d. samples from Exponential(lambda)
    Analytical MLE formula: lambda_hat = n / sum(x_i)
    """
    n = len(data)
    sum_x = np.sum(data)
    return n / sum_x

def compute_exponential_map_gamma_prior(data, alpha=1.0, beta=1.0):
    """
    Assumes a Gamma prior on lambda: P(lambda) proportional to lambda^(alpha-1) * exp(-beta * lambda)
    Analytical MAP formula: lambda_hat = (n + alpha - 1) / (sum(x_i) + beta)
    For a standard prior alpha=1, beta=1 -> P(lambda) = exp(-lambda)
    """
    n = len(data)
    sum_x = np.sum(data)
    return (n + alpha - 1.0) / (sum_x + beta)
```

## 7. Solved Exam-Style Examples

### Example 1: Deriving the MLE for a 1D Normal Distribution (Mean Optimization)

**Problem Statement:** Let $\mathcal{D} = \{x_1, x_2, \dots, x_n\}$ be a set of i.i.d. samples drawn from a 1D Gaussian distribution $\mathcal{N}(\mu, \sigma^2)$, where the variance $\sigma^2$ is known and fixed. Derive the Maximum Likelihood Estimate ($\hat{\mu}_{\text{MLE}}$) for the unknown mean parameter.

#### Step-by-Step Analytical Derivation

1. Write out the product likelihood function using the Gaussian probability density function:
    
    $$L(\mu) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x_i - \mu)^2}{2\sigma^2} \right)$$
    
2. Apply the natural logarithm to construct the log-likelihood function $\ell(\mu)$:
    
    $$\ell(\mu) = \sum_{i=1}^n \left[ -\frac{1}{2}\ln(2\pi\sigma^2) - \frac{(x_i - \mu)^2}{2\sigma^2} \right]$$
    
    $$\ell(\mu) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$$
    
3. Take the first derivative of $\ell(\mu)$ with respect to $\mu$:
    
    $$\frac{\partial \ell(\mu)}{\partial \mu} = 0 - \frac{1}{2\sigma^2} \sum_{i=1}^n 2(x_i - \mu)(-1) = \frac{1}{\sigma^2} \sum_{i=1}^n (x_i - \mu)$$
    
4. Set this derivative to zero to find the maximizing threshold point $\hat{\mu}_{\text{MLE}}$:
    
    $$\frac{1}{\sigma^2} \sum_{i=1}^n (x_i - \hat{\mu}_{\text{MLE}}) = 0 \implies \sum_{i=1}^n x_i - \sum_{i=1}^n \hat{\mu}_{\text{MLE}} = 0$$
    
    $$\sum_{i=1}^n x_i - n\hat{\mu}_{\text{MLE}} = 0 \implies \hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n x_i$$
    

**Final Exam Answer:** The Maximum Likelihood Estimate for the mean is the sample average: **$\hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n x_i$**.

### Example 2: Deriving a MAP Point Estimate with an Exponential Distribution

**Problem Statement:** Let $\mathcal{D} = \{x_1, \dots, x_n\}$ be $n$ i.i.d. observations sampled from an Exponential rate distribution:

$$P(x_i \mid \theta) = \theta e^{-\theta x_i} \quad (\text{for } x_i \geq 0, \theta > 0)$$

We define our prior distribution over the rate parameter as a standard exponential decay model:

$$P(\theta) = e^{-\theta} \quad (\text{for } \theta > 0)$$

Derive the explicit analytical formula for the Maximum A Posteriori estimate $\hat{\theta}_{\text{MAP}}$.

#### Step-by-Step Analytical Derivation

1. Construct the joint likelihood formula for the i.i.d. observations:
    
    $$P(\mathcal{D} \mid \theta) = \prod_{i=1}^n \theta e^{-\theta x_i} = \theta^n \exp\left( -\theta \sum_{i=1}^n x_i \right)$$
    
2. Set up the unnormalized posterior probability expression by incorporating the prior:
    
    $$P(\theta \mid \mathcal{D}) \propto P(\mathcal{D} \mid \theta)P(\theta) = \left[ \theta^n \exp\left( -\theta \sum_{i=1}^n x_i \right) \right] \cdot e^{-\theta}$$
    
    $$P(\theta \mid \mathcal{D}) \propto \theta^n \exp\left( -\theta \left( \sum_{i=1}^n x_i + 1 \right) \right)$$
    
3. Compute the natural log-posterior transformation field $\mathcal{L}_{\text{MAP}}(\theta)$:
    
    $$\mathcal{L}_{\text{MAP}}(\theta) = \ln P(\theta \mid \mathcal{D}) = n\ln\theta - \theta\left(\sum_{i=1}^n x_i + 1\right)$$
    
4. Differentiate with respect to the parameter variable $\theta$:
    
    $$\frac{\partial \mathcal{L}_{\text{MAP}}(\theta)}{\partial \theta} = \frac{n}{\theta} - \left(\sum_{i=1}^n x_i + 1\right)$$
    
5. Set the derivative to zero to isolate the MAP optimal point solution:
    
    $$\frac{n}{\hat{\theta}_{\text{MAP}}} = \sum_{i=1}^n x_i + 1 \implies \hat{\theta}_{\text{MAP}} = \frac{n}{\sum_{i=1}^n x_i + 1}$$
    

**Final Exam Answer:** The analytical MAP estimate for the rate parameter is **$\hat{\theta}_{\text{MAP}} = \frac{n}{\sum_{i=1}^n x_i + 1}$**. Notice that the prior adds a $+1$ modifier to the denominator, regularizing the rate when the sample size $n$ is small.

---
## 🔗 Navigation
**Previous:** [[ ]] | **Next:** [[ ]]