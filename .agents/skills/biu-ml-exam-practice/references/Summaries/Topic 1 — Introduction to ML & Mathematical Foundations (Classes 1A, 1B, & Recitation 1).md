---
type: uni_general
course: "[[Machine learning]]"
status: 🟢 Active
order: Summary
date_added: 2026-06-21
---
# Study Guide - Topic 1 — Introduction to ML & Mathematical Foundations (Classes 1A, 1B, & Recitation 1)

**MOC:** [[Machine learning MOC]]
**Course:** [[Machine learning]]

# Study Guide: Topic 1 — Introduction to ML & Mathematical Foundations (Classes 1A, 1B, & Recitation 1)

## 1. Core Machine Learning Concepts

### Paradigms of Machine Learning

- **Supervised Learning:** Given a training dataset of paired inputs and labels $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ sampled from an unknown joint distribution $P(\mathbf{x}, y)$, the objective is to learn a mapping $f: \mathcal{X} \rightarrow \mathcal{Y}$ that minimizes the expected loss on unseen data.
    
- **Unsupervised Learning:** Given an unlabeled dataset $\mathcal{D} = \{\mathbf{x}_i\}_{i=1}^n$, the objective is to uncover latent structures, patterns, dense representations, or underlying probability density functions without explicit target labels.
    
- **Reinforcement Learning:** An interactive paradigm where an agent learns to map states to actions within an environment to maximize a long-term cumulative scalar reward signal.
    

### The Data Distribution Spectrum

Real-world data exhibits a highly non-uniform, long-tailed distribution:

- **Head (High Frequency):** A small fraction of distinct queries or instances repeat extremely often. The optimal strategy here is **memorization** or knowledge retrieval.
    
- **Body (Moderate Frequency):** Instances that are similar but not identical to training observations. The optimal strategy is **generalization**.
    
- **Tail (Low Frequency / Outliers):** Unique, rare, or completely novel combinations of features. Handling the tail effectively requires robust algorithmic optimization.
    

## 2. Linear Algebra Fundamentals

### Matrices as Linear Transformations

An operational matrix $A \in \mathbb{R}^{d \times d}$ applied to a vector $\mathbf{v} \in \mathbb{R}^d$ acts as a linear transformation consisting of geometric rotation and scaling.

### Orthogonal Matrices

A square matrix $A \in \mathbb{R}^{n \times n}$ is strictly orthogonal if and only if:

$$A^T A = A A^T = I$$

This fundamental property implies:

1. The matrix transpose is identically its inverse: $A^T = A^{-1}$.
    
2. Every distinct row and column vector pair within $A$ forms an orthonormal basis (mutually orthogonal with a unit norm of 1).
    

### Eigenvalues & Eigenvectors

For a square linear operator matrix $A \in \mathbb{R}^{n \times n}$, a non-zero vector $\mathbf{v} \neq \mathbf{0}$ is an eigenvector of $A$ if it satisfies:

$$A\mathbf{v} = \lambda\mathbf{v}$$

where $\lambda \in \mathbb{R}$ (or $\mathbb{C}$) is the corresponding scalar eigenvalue. Geometrically, the transformation of $\mathbf{v}$ by $A$ alters only its scale along its span, not its directional orientation.

### Quadratic Forms

Given a symmetric matrix $A \in \mathbb{R}^{d \times d}$ and a vector $\mathbf{x} \in \mathbb{R}^d$, the quadratic form maps the vector to a scalar value via the expression:

$$f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} = \sum_{i=1}^d \sum_{j=1}^d A_{ij} x_i x_j$$

- **Positive Definite ($A \succ 0$):** $\mathbf{x}^T A \mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$.
    
- **Positive Semi-Definite ($A \succeq 0$):** $\mathbf{x}^T A \mathbf{x} \geq 0$ for all $\mathbf{x}$.
    

## 3. Probability & Statistics

### Bayes' Rule

Bayes' rule computes the posterior probability of an event or world state $A$ conditioned on observed evidence $B$:

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}$$

Using the law of total probability, if the sample space is partitioned by mutually exclusive and exhaustive states $\{\omega_i\}_{i=1}^k$, the formula expands to:

$$P(\omega_j \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid \omega_j) P(\omega_j)}{\sum_{i=1}^k P(\mathbf{x} \mid \omega_i) P(\omega_i)}$$

### The Multivariate Gaussian Distribution

The probability density function of a $d$-dimensional multivariate Gaussian random variable $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$ is given analytically by:

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

where $\boldsymbol{\mu} \in \mathbb{R}^d$ is the mean vector, $\Sigma \in \mathbb{R}^{d \times d}$ is the symmetric, positive-definite covariance matrix, and $|\Sigma|$ represents the matrix determinant.

#### Critical Special Covariance Conditions

1. **Spherical / Identity Covariance ($\Sigma = \sigma^2 I$):** All dimensions are statistically independent and share identical variance. The probability density isolines form concentric hyperspheres centered at $\boldsymbol{\mu}$.
    
2. **Diagonal Covariance ($\Sigma = \text{diag}(\sigma_1^2, \sigma_2^2, \dots, \sigma_d^2)$):** Features are statistically uncorrelated, but possess distinct variances along each standard coordinate axis. Isolines form axis-aligned hyperellipsoids.
    
3. **Full Covariance Matrix:** Features exhibit non-zero linear correlation. Isolines form rotated hyperellipsoids whose principal axes correspond directly to the eigenvectors of $\Sigma$.
    

## 4. Information Theory & Uncertainty

### Entropy

Entropy quantifies the expected internal uncertainty or average information content inherent to a discrete random variable $X$ distributed according to $P(x)$:

$$H(X) = -\sum_{x \in \mathcal{X}} P(x) \log_2 P(x)$$

### Cross Entropy

Given a true distribution $P(x)$ and an empirical model prediction or approximating distribution $Q(x)$, the cross entropy evaluates the average bits needed to encode outcomes under the mismatch:

$$H(P, Q) = -\sum_{x \in \mathcal{X}} P(x) \log_2 Q(x)$$

## 5. Multivariate Calculus

### The Gradient

For a scalar field function $f: \mathbb{R}^d \rightarrow \mathbb{R}$, the gradient vector $\nabla_\mathbf{x} f(\mathbf{x}) \in \mathbb{R}^d$ pools all first-order partial derivatives, indicating the direction of steepest local ascent:

$$\nabla_\mathbf{x} f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_d} \end{bmatrix}^T$$

#### Essential Core Gradient Identities

- **Linear Form:** $\nabla_\mathbf{x} (\mathbf{a}^T \mathbf{x}) = \mathbf{a}$
    
- **Symmetric Quadratic Form:** $\nabla_\mathbf{x} (\mathbf{x}^T A \mathbf{x}) = 2A\mathbf{x} \quad (\text{for symmetric } A)$
    

### The Hessian Matrix

The Hessian matrix $\nabla^2_\mathbf{x} f(\mathbf{x}) \in \mathbb{R}^{d \times d}$ structures all second-order partial derivatives, capturing the local geometric curvature of the function space:

$$\left( \nabla^2_\mathbf{x} f(\mathbf{x}) \right)_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

- **Identity for Quadratic Form:** $\nabla^2_\mathbf{x} (\mathbf{x}^T A \mathbf{x}) = 2A \quad (\text{for symmetric } A)$
    

## 6. Python / NumPy Implementation Snippets

These functional blocks illustrate how to compute core concepts for exam-style software validations:

```python
import numpy as np

# 1. Verification of Matrix Orthogonality
def is_orthogonal(A, tol=1e-9):
    identity = np.eye(A.shape[0])
    return np.allclose(np.dot(A.T, A), identity, atol=tol)

# 2. Eigen-Decomposition Extraction
def get_eigen_properties(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    return eigenvalues, eigenvectors

# 3. Multivariate Gaussian PDF Computation
def multivariate_gaussian_pdf(x, mu, sigma):
    d = len(mu)
    det_sigma = np.linalg.det(sigma)
    inv_sigma = np.linalg.inv(sigma)
    
    # Compute the quadratic term inside the exponential
    diff = x - mu
    quad_term = np.dot(diff.T, np.dot(inv_sigma, diff))
    
    normalization = 1.0 / (((2 * np.pi) ** (d / 2.0)) * (det_sigma ** 0.5))
    return normalization * np.exp(-0.5 * quad_term)
    

# Query point
x = np.array([0.5, -0.2])

# Mean vector
mu = np.array([0.0, 0.0])

# Covariance matrix (must be symmetric and positive-definite)
sigma = np.array([
    [1.0, 0.5],
    [0.5, 1.0]
])

density = multivariate_gaussian_pdf(x, mu, sigma)
print("Gaussian PDF density value:", density)  # Expected output: ~0.138

```

## 7. Solved Exam-Style Examples

### Example 1: Disease Screening Diagnostic (Bayes' Rule)

**Problem Statement:** A rare medical condition affects exactly $1\%$ of a designated population ($P(\text{Disease}) = 0.01$). A diagnostic screening protocol provides the following technical parameters:

- **Sensitivity (True Positive Rate):** $P(+\mid \text{Disease}) = 0.95$
    
- **Specificity (True Negative Rate):** $P(-\mid \text{Health}) = 0.90$
    

Calculate the exact posterior probability that an individual chosen at random who tests positive actually has the disease ($P(\text{Disease} \mid +)$).

#### Step-by-Step Analytical Derivation

1. Identify the complementary prior probability of being healthy:
    
    $$P(\text{Health}) = 1 - P(\text{Disease}) = 1 - 0.01 = 0.99$$
    
2. Identify the false positive rate using the specificity bound:
    
    $$P(+\mid \text{Health}) = 1 - P(-\mid \text{Health}) = 1 - 0.90 = 0.10$$
    
3. Formulate the full expansion using Bayes' theorem:
    
    $$P(\text{Disease} \mid +) = \frac{P(+\mid \text{Disease})P(\text{Disease})}{P(+\mid \text{Disease})P(\text{Disease}) + P(+\mid \text{Health})P(\text{Health})}$$
    
4. Substitute the numerical parameters into the setup:
    
    $$P(\text{Disease} \mid +) = \frac{(0.95)(0.01)}{(0.95)(0.01) + (0.10)(0.99)}$$
    
    $$P(\text{Disease} \mid +) = \frac{0.0095}{0.0095 + 0.0990} = \frac{0.0095}{0.1085} \approx 0.08756$$
    

**Final Exam Answer:** The posterior probability that the patient has the disease given a positive test is approximately **$8.76\%$**.

### Example 2: Extremum Evaluation via Vector Calculus

**Problem Statement:** Find the critical point $\mathbf{x}^*$ that minimizes the multivariate scalar objective function defined by:

$$f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T A \mathbf{x} - \mathbf{b}^T \mathbf{x}$$

where $A \in \mathbb{R}^{d \times d}$ is a symmetric positive-definite matrix and $\mathbf{b} \in \mathbb{R}^d$.

#### Step-by-Step Analytical Derivation

1. Compute the complete gradient vector with respect to $\mathbf{x}$:
    
    $$\nabla_\mathbf{x} f(\mathbf{x}) = \nabla_\mathbf{x} \left( \frac{1}{2}\mathbf{x}^T A \mathbf{x} \right) - \nabla_\mathbf{x} (\mathbf{b}^T \mathbf{x})$$
    
2. Apply standard vector calculus derivative identities:
    
    $$\nabla_\mathbf{x} f(\mathbf{x}) = A\mathbf{x} - \mathbf{b}$$
    
3. Set the gradient vector to zero to locate the stationary critical point:
    
    $$A\mathbf{x}^* - \mathbf{b} = \mathbf{0} \implies A\mathbf{x}^* = \mathbf{b}$$
    
4. Since $A$ is positive-definite, it is invertible. Isolate $\mathbf{x}^*$:
    
    $$\mathbf{x}^* = A^{-1}\mathbf{b}$$
    
5. Verify the optimality type by inspecting the Hessian matrix:
    
    $$\nabla^2_\mathbf{x} f(\mathbf{x}) = A$$
    
    Since $A \succ 0$ (positive-definite), the curvature guarantees that $\mathbf{x}^*$ represents a unique global minimum.
    

**Final Exam Answer:** The optimal critical point minimizing the objective function is **$\mathbf{x}^* = A^{-1}\mathbf{b}$**.

---
## 🔗 Navigation
**Previous:** [[ ]] | **Next:** [[ ]]