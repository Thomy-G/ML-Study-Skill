---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Summary
date_added: 2026-06-21
---
# Topic 13 — Dimensionality Reduction & Principal Component Analysis (PCA) (Class 09A, Past Exams)

**MOC:** [[Machine learning MOC]]
**Course:** [[English]]

# Study Guide: Topic 13 — Dimensionality Reduction & Principal Component Analysis (PCA) (Class 09A, Past Exams)

## 1. Context: The Curse of Dimensionality & Feature Redundancy

In Unsupervised Learning, real-world data points often reside in exceptionally high-dimensional feature spaces ($\mathbf{x}_i \in \mathbb{R}^d$, where $d$ is very large). However, high-dimensional spaces introduce severe algorithmic challenges:

1. **The Curse of Dimensionality:** As $d$ grows, the volume of the space expands exponentially, causing data points to become extremely sparse. Geometric distances (like Euclidean distance) begin to lose their discriminative power because the distance between any two random points converges to a nearly uniform value.
    
2. **Statistical Redundancy:** Many features are highly correlated or represent purely noisy variations. The true underlying degrees of freedom—the **intrinsic dimensionality** of the data—is often much smaller than $d$.
    

**Dimensionality Reduction** seeks to learn an unsupervised mapping function $f: \mathbb{R}^d \rightarrow \mathbb{R}^r$ (where $r \ll d$) that compresses the data into a low-dimensional representation while preserving its essential geometric, structural, or statistical variance.

## 2. Principal Component Analysis (PCA) Foundations

**Principal Component Analysis (PCA)** is the classic linear framework for dimensionality reduction. It constructs an orthogonal coordinate system where the new axes—called **Principal Components**—are ordered by the amount of data variance they capture.

### Structural Requirements

Let $X \in \mathbb{R}^{n \times d}$ be our data matrix containing $n$ samples.

- **Zero-Mean Centering:** Before applying PCA, the dataset _must_ be centered by subtracting the empirical sample mean vector $\boldsymbol{\mu} = \frac{1}{n}\sum_{i=1}^n \mathbf{x}_i$ from every observation row:
    
    $$\mathbf{x}_i \leftarrow \mathbf{x}_i - \boldsymbol{\mu}$$
    
    Centering ensures that the sample covariance matrix can be calculated purely via matrix multiplication.
    

### The Sample Covariance Matrix

For zero-mean data, the symmetric, positive semi-definite sample covariance matrix $\Sigma \in \mathbb{R}^{d \times d}$ is given by:

$$\Sigma = \frac{1}{n} X^T X = \frac{1}{n} \sum_{i=1}^n \mathbf{x}_i \mathbf{x}_i^T$$

## 3. Dual Formulations of PCA

PCA can be derived from two completely different geometric perspectives, both leading to the exact same optimal mathematical solution.

### Formulation A: Maximizing Projected Variance

We want to find a unit projection direction vector $\mathbf{v} \in \mathbb{R}^d$ ($\|\mathbf{v}\|^2 = \mathbf{v}^T\mathbf{v} = 1$) such that projecting our zero-mean data points onto $\mathbf{v}$ maximizes the variance of the projected coordinates.

The projected scalar coordinate of a point $\mathbf{x}_i$ onto direction $\mathbf{v}$ is $\mathbf{v}^T \mathbf{x}_i$. The sample variance of these projected coordinates is:

$$\sigma_{\text{projected}}^2 = \frac{1}{n}\sum_{i=1}^n \left(\mathbf{v}^T \mathbf{x}_i\right)^2 = \frac{1}{n}\sum_{i=1}^n \mathbf{v}^T \mathbf{x}_i \mathbf{x}_i^T \mathbf{v} = \mathbf{v}^T \left( \frac{1}{n} X^T X \right) \mathbf{v} = \mathbf{v}^T \Sigma \mathbf{v}$$

To maximize this objective under the constraint $\mathbf{v}^T\mathbf{v} = 1$, we set up the **Lagrangian function**:

$$\mathcal{L}(\mathbf{v}, \lambda) = \mathbf{v}^T \Sigma \mathbf{v} - \lambda \left(\mathbf{v}^T \mathbf{v} - 1\right)$$

Taking the vector derivative with respect to $\mathbf{v}$ and setting it to $\mathbf{0}$ yields:

$$\nabla_{\mathbf{v}} \mathcal{L} = 2\Sigma \mathbf{v} - 2\lambda \mathbf{v} = \mathbf{0} \implies \Sigma \mathbf{v} = \lambda \mathbf{v}$$

This is the standard **Eigenvalue Equation**. The optimal projection direction $\mathbf{v}$ must be an **eigenvector** of the covariance matrix $\Sigma$, and the resulting projected variance equals its corresponding eigenvalue: $\mathbf{v}^T \Sigma \mathbf{v} = \mathbf{v}^T (\lambda \mathbf{v}) = \lambda$.

### Formulation B: Minimizing Reconstruction Error (From 2023 Moed B)

Alternatively, we can frame PCA as finding a low-dimensional linear subspace such that the squared reconstruction distance between the original high-dimensional points $\mathbf{x}_i$ and their low-dimensional projections is minimized. Minimizing reconstruction error is mathematically equivalent to maximizing projected variance due to the Pythagorean theorem.

## 4. The Complete PCA Algorithmic Pseudocode

To compress a zero-mean dataset from $d$ dimensions down to a lower target dimension $r$:

1. Compute the sample covariance matrix: $\Sigma = \frac{1}{n} X^T X$.
    
2. Compute the eigen-decomposition of $\Sigma$: extract all eigenvectors and their corresponding scalar eigenvalues $\lambda_j$.
    
3. Sort the eigenvalues in descending order: $\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_d \geq 0$.
    
4. Select the top $r$ eigenvectors corresponding to the $r$ largest eigenvalues. Stack these vectors horizontally as columns to form the orthogonal projection matrix $V_r \in \mathbb{R}^{d \times r}$:
    
    $$V_r = \begin{bmatrix} \mid & \mid & & \mid \\ \mathbf{v}_1 & \mathbf{v}_2 & \dots & \mathbf{v}_r \\ \mid & \mid & & \mid \end{bmatrix}$$
    
5. Project the training samples into the compressed $r$-dimensional space:
    
    $$Z = X V_r \quad (Z \in \mathbb{R}^{n \times r})$$
    

## 5. Python / NumPy Implementation Snippet

```python
import numpy as np

def principal_component_analysis(X, r):
    """
    Performs PCA dimensionality reduction to compress data down to r dimensions.
    X: Data matrix of shape (n, d)
    r: Target lower dimension (r < d)
    """
    # 1. Zero-mean centering transformation
    mean_vector = np.mean(X, axis=0)
    X_centered = X - mean_vector
    
    # 2. Compute the sample covariance matrix (d x d)
    n = X.shape[0]
    covariance_matrix = (1.0 / n) * np.dot(X_centered.T, X_centered)
    
    # 3. Compute eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # np.linalg.eigh returns values in ascending order; reverse for descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 4. Extract the top r components
    V_r = eigenvectors[:, :r]
    
    # 5. Project data into the low-dimensional subspace
    X_reduced = np.dot(X_centered, V_r)
    
    return X_reduced, V_r, eigenvalues[:r]
```

## 6. Solved Exam-Style Examples

### Example 1: Calculating the Total Variance Preserved

**Problem Statement:** You compute the complete eigen-decomposition of a 3D dataset's covariance matrix, obtaining three sorted eigenvalues: $\lambda_1 = 6.0$, $\lambda_2 = 3.0$, and $\lambda_3 = 1.0$. If you compress this dataset down to $r=2$ dimensions using PCA, calculate the exact proportion of total data variance preserved in the low-dimensional subspace.

#### Step-by-Step Analytical Derivation

1. The total variance of a dataset equals the sum of all eigenvalues of its covariance matrix (which matches the trace of the matrix):
    
    $$\text{Total Variance} = \sum_{j=1}^d \lambda_j = \lambda_1 + \lambda_2 + \lambda_3 = 6.0 + 3.0 + 1.0 = 10.0$$
    
2. Calculate the variance preserved by retaining the top $r=2$ components:
    
    $$\text{Preserved Variance} = \sum_{j=1}^r \lambda_j = \lambda_1 + \lambda_2 = 6.0 + 3.0 = 9.0$$
    
3. Compute the ratio of preserved variance to total variance:
    
    $$\text{Proportion Preserved} = \frac{\text{Preserved Variance}}{\text{Total Variance}} = \frac{9.0}{10.0} = 0.90 = 90\%$$
    

**Final Exam Answer:** The 2D PCA projection preserves exactly **$90\%$** of the original dataset's total variance.

### Example 2: Mathematical Tradeoffs of Subspace Projection (From 2023 Moed B)

**Problem Statement:** A student wants to train a linear binary classifier on a high-dimensional dataset ($d=1000$). Because $d \gg n$, the model suffers from severe overfitting. The student proposes compressing the data down to $r=50$ dimensions using PCA before fitting the classifier.

Explain in what scenario this preprocessing pipeline is expected to improve classification generalizability, and in what specific scenario it could severely hurt accuracy.

#### Analytical Evaluation

- **When it helps (Overfitting Mitigation):** If the true underlying features that distinguish the two classes align directly along the directions of maximum global variance, PCA will successfully capture them. By filtering out the remaining 950 low-variance dimensions, PCA removes random noise and irrelevant variations, significantly lowering model variance and reducing the risk of overfitting.
    
- **When it hurts (Information Loss Failure Mode):** PCA is a completely **unsupervised** technique; it selects projection axes based entirely on feature spread without ever looking at class labels. If the critical features needed to separate the classes happen to reside along a low-variance direction, PCA will completely discard them. For example, if the data forms two parallel, tightly squeezed lines, the axis of maximum variance runs parallel to the lines, while the axis that separates the classes runs perpendicular to them (capturing very low variance). In this scenario, compressing the data with PCA will project the classes directly on top of each other, making them linearly inseparable and severely hurting classification performance.
    



---
## 🔗 Navigation
**Previous:** [[ ]] | **Next:** [[ ]]