---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Summary
date_added: 2026-06-21
---
# Topic 19 — Expectation-Maximization (EM) & Mixture Models (Syllabus Item 10)

**MOC:** [[Machine learning MOC]]
**Course:** [[English]]

# Study Guide: Topic 19 — Expectation-Maximization (EM) & Mixture Models (Syllabus Item 10)

## 1. Context: Moving Beyond Hard Latent Clustering

In Topic 12, we evaluated the $K$-means algorithm for hard prototype clustering. While $K$-means is computationally efficient, it exhibits two primary limitations:

1. **Hard Assignment:** It forces every data instance to map strictly to a single group ($z_{i,k} \in \{0,1\}$), ignoring situations where a sample lies directly on the ambiguous boundary between two distributions.
    
2. **Spherical Geometry Constraints:** It relies purely on isotropic Euclidean distances, meaning it completely fails to model clusters that exhibit varying scales, densities, or elongated ellipsoidal shapes.
    

The **Expectation-Maximization (EM)** algorithm handles these limitations by shifting from a hard geometric partition to a **soft probabilistic generative framework** (Mixture Models).

## 2. Gaussian Mixture Models (GMMs)

A Gaussian Mixture Model assumes that the overall data distribution is composed of a linear combination of $K$ distinct, underlying Gaussian subpopulations.

### The Generative Process

To generate a data point $\mathbf{x}_i \in \mathbb{R}^d$:

1. Select a hidden latent cluster component $k \in \{1, \dots, K\}$ from a multinomial prior distribution parameterized by mixing weights $\pi_k$:
    
    $$P(z_{i,k} = 1) = \pi_k \quad \text{where } \sum_{k=1}^K \pi_k = 1 \text{ and } \pi_k \geq 0$$
    
2. Given the selected component $k$, draw the continuous feature vector $\mathbf{x}_i$ from its corresponding class-conditional multivariate Gaussian distribution:
    
    $$\mathbf{x}_i \mid z_{i,k}=1 \sim \mathcal{N}(\boldsymbol{\mu}_k, \Sigma_k)$$
    

The overall marginal probability density function of an observed sample is:

$$P(\mathbf{x}_i \mid \boldsymbol{\theta}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \Sigma_k)$$

where $\boldsymbol{\theta} = \{\pi_k, \boldsymbol{\mu}_k, \Sigma_k\}_{k=1}^K$ collects all hidden parameters of the world model.

## 3. The Complete Likelihood with Latent Variables (From 2024 & 2025 Moed A)

If we try to maximize the log-likelihood of the observed data directly, the summation appears _inside_ the logarithm, making an analytical solution mathematically intractable:

$$\ell(\boldsymbol{\theta}) = \sum_{i=1}^n \ln \left( \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \Sigma_k) \right)$$

To resolve this, we introduce hidden binary indicator variables $\mathbf{z}_i = (z_{i,1}, \dots, z_{i,K})$ that specify which individual component generated sample $i$. This defines the **Complete Data Joint Likelihood**:

$$P(\mathcal{D}, Z \mid \boldsymbol{\theta}) = \prod_{i=1}^n \prod_{k=1}^K \Big[ \pi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \Sigma_k) \Big]^{z_{i,k}}$$

> 💡 **Intuitive Step-by-Step Breakdown of How This Works:**
> 
> 1. **Why standard log-likelihood is hard:** When we only observe $\mathbf{x}_i$, the density is $P(\mathbf{x}_i) = \sum_{k=1}^K \pi_k \mathcal{N}_k$. Taking $\ln\left(\sum_{k=1}^K \dots\right)$ traps the sum *inside* the logarithm, preventing us from breaking the expression into simple linear terms.
> 2. **The One-Hot Binary Indicator ($\mathbf{z}_i$):** If we knew the true generating cluster $k$, $\mathbf{z}_i$ would be a one-hot vector where $z_{i,k} = 1$ for the true cluster and $z_{i,j} = 0$ for all other $j \neq k$.
> 3. **The Product Exponentiation Trick ($\prod [A_k]^{z_{i,k}}$):**
>    * For the single active cluster ($z_{i,k} = 1$), $[A_k]^1 = A_k$.
>    * For all inactive clusters ($z_{i,j} = 0$), $[A_j]^0 = 1$.
>    * Multiplying them together isolates the active component without any addition ($\sum$) inside!
> 4. **Pulling the Sum OUTSIDE the Logarithm:** Taking the logarithm converts the product into a clean linear sum:
>    $$\ln P(\mathcal{D}, Z \mid \boldsymbol{\theta}) = \sum_{i=1}^n \sum_{k=1}^K z_{i,k} \Big[ \ln \pi_k + \ln \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \Sigma_k) \Big]$$
>    Since $\ln \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \Sigma_k)$ cancels out the exponential $e^{(\dots)}$, differentiating with respect to $\boldsymbol{\mu}_k$ or $\Sigma_k$ becomes simple and linear!

Taking the natural logarithm transforms the product into a manageable linear summation:

$$\ln P(\mathcal{D}, Z \mid \boldsymbol{\theta}) = \sum_{i=1}^n \sum_{k=1}^K z_{i,k} \Big[ \ln \pi_k + \ln \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \Sigma_k) \Big]$$

## 4. The General Expectation-Maximization (EM) Framework

The EM algorithm (Dempster, Laird, & Rubin, 1977) optimizes models with latent variables by constructing and maximizing an auxiliary lower-bound function called the **$Q$-Function**.

---

### 4.1 The $Q$-Function (Expected Complete Log-Likelihood)

At iteration step $t$, given current parameter estimates $\boldsymbol{\theta}^{(t)}$, the **$Q$-function** is defined as the expected value of the complete log-likelihood taken with respect to the conditional posterior distribution of the latent variables $Z$ given observed data $\mathcal{D}$:

$$Q(\boldsymbol{\theta}, \boldsymbol{\theta}^{(t)}) = \mathbb{E}_{Z \mid \mathcal{D}, \boldsymbol{\theta}^{(t)}} \left[ \ln P(\mathcal{D}, Z \mid \boldsymbol{\theta}) \right]$$

Expanding for a mixture model with responsibilities $\gamma_{i,k}^{(t)} = \mathbb{E}[z_{i,k} \mid \mathbf{x}_i, \boldsymbol{\theta}^{(t)}]$:

$$Q(\boldsymbol{\theta}, \boldsymbol{\theta}^{(t)}) = \sum_{i=1}^n \sum_{k=1}^K \gamma_{i,k}^{(t)} \left[ \ln \pi_k + \ln P(\mathbf{x}_i \mid \boldsymbol{\theta}_k) \right]$$

---

### 4.2 Step 1: The Expectation Step (E-Step)

Compute the **responsibilities** $\gamma_{i,k}^{(t)}$—the posterior probability that latent component $k$ generated observed data point $\mathbf{x}_i$:

$$\gamma_{i,k}^{(t)} = P(z_{i,k} = 1 \mid \mathbf{x}_i, \boldsymbol{\theta}^{(t)}) = \frac{\pi_k^{(t)} \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k^{(t)}, \Sigma_k^{(t)})}{\sum_{j=1}^K \pi_j^{(t)} \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j^{(t)}, \Sigma_j^{(t)})}$$

Define the **effective number of data points** assigned to cluster $k$:

$$N_k = \sum_{i=1}^n \gamma_{i,k}^{(t)}$$

---

### 4.3 Step 2: The Maximization Step (M-Step) Analytical Derivations

Find new parameters $\boldsymbol{\theta}^{(t+1)} = \arg\max_{\boldsymbol{\theta}} Q(\boldsymbol{\theta}, \boldsymbol{\theta}^{(t)})$.

#### 1. Derivation of Mixture Weights $\pi_k$ (using Lagrange Multipliers)
We maximize $\sum_{k=1}^K N_k \ln \pi_k$ subject to the normalization constraint $\sum_{k=1}^K \pi_k = 1$:

$$\mathcal{L}_{\text{Lagrange}}(\boldsymbol{\pi}, \lambda) = \sum_{k=1}^K N_k \ln \pi_k + \lambda \left( 1 - \sum_{k=1}^K \pi_k \right)$$

Taking the partial derivative with respect to $\pi_k$ and setting to zero:

$$\frac{\partial \mathcal{L}_{\text{Lagrange}}}{\partial \pi_k} = \frac{N_k}{\pi_k} - \lambda = 0 \implies N_k = \lambda \pi_k$$

Summing both sides over all $K$ components:

$$\sum_{k=1}^K N_k = \lambda \sum_{k=1}^K \pi_k \implies n = \lambda \cdot 1 \implies \lambda = n$$

Substituting $\lambda = n$ back into the weight expression yields the exact M-step update:

$$\pi_k^{(t+1)} = \frac{N_k}{n} = \frac{1}{n} \sum_{i=1}^n \gamma_{i,k}^{(t)}$$

#### 2. Derivation of Means $\boldsymbol{\mu}_k$
Differentiating $Q(\boldsymbol{\theta}, \boldsymbol{\theta}^{(t)})$ with respect to $\boldsymbol{\mu}_k$:

$$\frac{\partial Q}{\partial \boldsymbol{\mu}_k} = \sum_{i=1}^n \gamma_{i,k}^{(t)} \nabla_{\boldsymbol{\mu}_k} \left[ -\frac{1}{2}(\mathbf{x}_i - \boldsymbol{\mu}_k)^T \Sigma_k^{-1} (\mathbf{x}_i - \boldsymbol{\mu}_k) \right] = \sum_{i=1}^n \gamma_{i,k}^{(t)} \Sigma_k^{-1} (\mathbf{x}_i - \boldsymbol{\mu}_k) = \mathbf{0}$$

Multiplying by $\Sigma_k$ and expanding:

$$\sum_{i=1}^n \gamma_{i,k}^{(t)} \mathbf{x}_i - \left(\sum_{i=1}^n \gamma_{i,k}^{(t)}\right) \boldsymbol{\mu}_k = \mathbf{0} \implies \boldsymbol{\mu}_k^{(t+1)} = \frac{\sum_{i=1}^n \gamma_{i,k}^{(t)} \mathbf{x}_i}{\sum_{i=1}^n \gamma_{i,k}^{(t)}} = \frac{1}{N_k} \sum_{i=1}^n \gamma_{i,k}^{(t)} \mathbf{x}_i$$

#### 3. Updated Covariance Matrix $\Sigma_k^{(t+1)}$:

$$\Sigma_k^{(t+1)} = \frac{1}{N_k} \sum_{i=1}^n \gamma_{i,k}^{(t)} (\mathbf{x}_i - \boldsymbol{\mu}_k^{(t+1)})(\mathbf{x}_i - \boldsymbol{\mu}_k^{(t+1)})^T$$

---

### 4.4 The Evidence Lower Bound (ELBO) & Jensen's Inequality Interpretation

The EM algorithm can be interpreted as coordinate ascent on the **Evidence Lower Bound (ELBO)** $F(q, \boldsymbol{\theta})$ using Jensen's Inequality:

$$\ln P(\mathcal{D} \mid \boldsymbol{\theta}) = \ln \sum_{Z} P(\mathcal{D}, Z \mid \boldsymbol{\theta}) = \ln \sum_{Z} q(Z) \frac{P(\mathcal{D}, Z \mid \boldsymbol{\theta})}{q(Z)}$$

Applying **Jensen's Inequality** ($\ln \mathbb{E}[U] \ge \mathbb{E}[\ln U]$):

$$\ln P(\mathcal{D} \mid \boldsymbol{\theta}) \ge \sum_{Z} q(Z) \ln \frac{P(\mathcal{D}, Z \mid \boldsymbol{\theta})}{q(Z)} = \mathbb{E}_q[\ln P(\mathcal{D}, Z \mid \boldsymbol{\theta})] + \mathcal{H}(q) = F(q, \boldsymbol{\theta})$$

*   **E-Step:** Fix $\boldsymbol{\theta}^{(t)}$ and maximize $F(q, \boldsymbol{\theta}^{(t)})$ with respect to distribution $q(Z)$. The maximum occurs when $q(Z) = P(Z \mid \mathcal{D}, \boldsymbol{\theta}^{(t)})$, closing the gap ($\text{KL}(q \parallel P) = 0$).
*   **M-Step:** Fix $q(Z)$ and maximize $F(q, \boldsymbol{\theta})$ with respect to parameters $\boldsymbol{\theta}$. This guarantees that the true data log-likelihood never decreases ($\ln P(\mathcal{D} \mid \boldsymbol{\theta}^{(t+1)}) \ge \ln P(\mathcal{D} \mid \boldsymbol{\theta}^{(t)})$).
    

## 5. Python / NumPy Implementation Snippet

```python
import numpy as np
from scipy.stats import multivariate_normal

def gmm_em_step(X, pi, means, covariances):
    """
    Executes a single complete E-step and M-step iteration for a GMM.
    X: shape (n, d)
    pi: shape (K,) - mixture weights
    means: shape (K, d) - centroid positions
    covariances: shape (K, d, d) - cluster shape structures
    """
    n, d = X.shape
    K = len(pi)
    
    # --- E-Step: Compute Responsibilities ---
    responsibilities = np.zeros((n, K))
    for k in range(K):
        # Evaluate multivariate normal pdf
        responsibilities[:, k] = pi[k] * multivariate_normal.pdf(X, mean=means[k], cov=covariances[k])
        
    # Normalize across components per row
    responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
    
    # --- M-Step: Update Model Parameters ---
    N_k = np.sum(responsibilities, axis=0) # Effective number of points assigned to cluster k
    
    for k in range(K):
        # 1. Update Mean
        means[k] = np.sum(responsibilities[:, k, np.newaxis] * X, axis=0) / N_k[k]
        
        # 2. Update Covariance
        diff = X - means[k]
        weighted_diff = responsibilities[:, k, np.newaxis] * diff
        covariances[k] = np.dot(weighted_diff.T, diff) / N_k[k]
        
    # 3. Update Prior Weights
    pi = N_k / n
    
    return pi, means, covariances
```

## 6. Solved Exam Question: Mixture of Bernoullis (From 2024 Moed A)

**Problem Statement:** Let $X$ be a binary random variable drawn from a mixture of $K$ different Bernoulli distributions, each with its own probability parameter $p_k$. Its probability mass function is $P(X=x \mid p_k) = p_k^x (1-p_k)^{1-x}$ for $x \in \{0,1\}$. Given samples $\mathcal{D} = \{x_1, \dots, x_n\}$, derive the explicit E-step posterior probability equation for the latent indicator variables $P(Z_{i,j} = 1 \mid x_i)$.

### Step-by-Step Analytical Derivation

1. Identify the prior probability of selecting the $j$-th Bernoulli component:
    
    $$P(Z_{i,j} = 1) = \pi_j$$
    
2. State the conditional probability of observing data point $x_i$ given that it was generated by component $j$:
    
    $$P(x_i \mid Z_{i,j} = 1) = p_j^{x_i} (1 - p_j)^{1 - x_i}$$
    
3. Set up the posterior conditional formulation using Bayes' Theorem:
    
    $$P(Z_{i,j} = 1 \mid x_i) = \frac{P(x_i \mid Z_{i,j} = 1) P(Z_{i,j} = 1)}{P(x_i)}$$
    
4. Expand the denominator marginal evidence $P(x_i)$ using the law of total probability across all $K$ mixture components:
    
    $$P(x_i) = \sum_{l=1}^K P(x_i \mid Z_{i,l} = 1) P(Z_{i,l} = 1) = \sum_{l=1}^K \pi_l p_l^{x_i} (1 - p_l)^{1 - x_i}$$
    
5. Substitute this back into the numerator configuration to find the final E-step equation:
    
    $$P(Z_{i,j} = 1 \mid x_i) = \frac{\pi_j p_j^{x_i} (1 - p_j)^{1 - x_i}}{\sum_{l=1}^K \pi_l p_l^{x_i} (1 - p_l)^{1 - x_i}}$$
    

**Final Exam Answer:** The analytical E-step update for a mixture of Bernoullis is **$\gamma_{i,j} = \frac{\pi_j p_j^{x_i} (1 - p_j)^{1 - x_i}}{\sum_{l=1}^K \pi_l p_l^{x_i} (1 - p_l)^{1 - x_i}}$**.



---
## 🔗 Navigation
**Previous:** [[ ]] | **Next:** [[ ]]