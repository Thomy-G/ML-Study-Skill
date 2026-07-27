---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Summary
date_added: 2026-06-21
---
# Topic 7 — Linear Models & Optimization (Linear & Ridge Regression) (Class 04A, Recitation 04)

**MOC:** [[Machine learning MOC]]
**Course:** [[English]]

# Study Guide: Topic 7 — Linear Models & Optimization (Linear & Ridge Regression) (Class 04A, Recitation 04)

## 1. Context: Empirical Risk Minimization for Continuous Variables

We now transition to parametric algorithms that do not assume prior distribution knowledge. Instead, they use a training dataset to explicitly learn a parametric function by optimizing a concrete geometric loss surface.

For **Linear Regression**, the target space is continuous ($\mathcal{Y} = \mathbb{R}$), and our hypothesis class $\mathcal{H}$ consists of all hyperplanes mapping a $d$-dimensional input vector $\mathbf{x} = [x_1, \dots, x_d]^T$ to a scalar prediction $\hat{y}$:

$$h_{\mathbf{w}, b}(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^d w_j x_j + b$$

### The Design Matrix Vectorization Trick

To eliminate the bias term scalar $b$ from our derivations, we append a constant feature of $1$ to every input vector: $\mathbf{x} \leftarrow [1, x_1, \dots, x_d]^T \in \mathbb{R}^{d+1}$ and bundle the bias into our weights: $\mathbf{w} \leftarrow [b, w_1, \dots, w_d]^T \in \mathbb{R}^{d+1}$.

For a complete dataset of $n$ samples, we arrange the observations horizontally as rows into the **Design Matrix** $X \in \mathbb{R}^{n \times (d+1)}$ and stack targets into vector $\mathbf{y} \in \mathbb{R}^n$:

$$X = \begin{bmatrix} \rule[0.5ex]{2em}{0.4pt} & (\mathbf{x}_1)^T & \rule[0.5ex]{2em}{0.4pt} \\ \rule[0.5ex]{2em}{0.4pt} & (\mathbf{x}_2)^T & \rule[0.5ex]{2em}{0.4pt} \\ & \vdots & \\ \rule[0.5ex]{2em}{0.4pt} & (\mathbf{x}_n)^T & \rule[0.5ex]{2em}{0.4pt} \end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}$$

This allows us to write the predictions for the entire dataset simultaneously as a single matrix-vector multiplication:

$$\hat{\mathbf{y}} = X\mathbf{w}$$

## 2. Ordinary Least Squares (OLS)

### The Objective Function

Under Ordinary Least Squares (OLS), we measure model failure using the Mean Squared Error (MSE) loss. The total empirical risk function $L(\mathbf{w})$ is defined as:

$$L(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n \left(\mathbf{w}^T \mathbf{x}_i - y_i\right)^2 = \frac{1}{n}\|X\mathbf{w} - \mathbf{y}\|^2$$

### Analytical Derivation of the Normal Equations

Because the loss function is quadratic and strictly convex, we can compute its unique global minimum analytically.

1. Expand the squared vector norm using matrix transpose properties:
    
    $$\|X\mathbf{w} - \mathbf{y}\|^2 = (X\mathbf{w} - \mathbf{y})^T (X\mathbf{w} - \mathbf{y})$$
    
    $$= \left(\mathbf{w}^T X^T - \mathbf{y}^T\right)(X\mathbf{w} - \mathbf{y}) = \mathbf{w}^T X^T X \mathbf{w} - \mathbf{w}^T X^T \mathbf{y} - \mathbf{y}^T X \mathbf{w} + \mathbf{y}^T \mathbf{y}$$
    
2. Since $\mathbf{w}^T X^T \mathbf{y}$ is a scalar, it equals its own transpose ($\mathbf{w}^T X^T \mathbf{y} = \mathbf{y}^T X \mathbf{w}$). We simplify the expression to:
    
    $$L(\mathbf{w}) = \frac{1}{n}\left( \mathbf{w}^T X^T X \mathbf{w} - 2\mathbf{w}^T X^T \mathbf{y} + \mathbf{y}^T \mathbf{y} \right)$$
    
3. Compute the gradient vector with respect to $\mathbf{w}$ using matrix calculus identities ($\nabla_{\mathbf{w}}(\mathbf{a}^T\mathbf{w}) = \mathbf{a}$ and $\nabla_{\mathbf{w}}(\mathbf{w}^T A \mathbf{w}) = 2A\mathbf{w}$):
    
    $$\nabla_{\mathbf{w}} L(\mathbf{w}) = \frac{1}{n}\left( 2X^T X \mathbf{w} - 2X^T \mathbf{y} \right)$$
    
4. Set the gradient vector to zero to isolate the critical point $\hat{\mathbf{w}}_{\text{OLS}}$:
    
    $$2X^T X \hat{\mathbf{w}}_{\text{OLS}} - 2X^T \mathbf{y} = \mathbf{0} \implies X^T X \hat{\mathbf{w}}_{\text{OLS}} = X^T \mathbf{y}$$
    

This system of linear equations is known as the **Normal Equations**. Assuming the matrix $X^T X$ is full-rank and invertible, the analytical solution is:

$$\hat{\mathbf{w}}_{\text{OLS}} = \left(X^T X\right)^{-1} X^T \mathbf{y}$$

## 3. Ridge Regression ($L_2$ Regularization)

When features are highly correlated (multi-collinearity) or the feature dimension exceeds the sample size ($d > n$), the matrix $X^T X$ becomes singular or poorly conditioned, leading to numerical instability and severe overfitting.

### The Objective Function

**Ridge Regression** resolves this by adding an $L_2$-norm regularization term to the objective function, penalizing large weight values:

$$L_{\text{Ridge}}(\mathbf{w}) = \frac{1}{n}\|X\mathbf{w} - \mathbf{y}\|^2 + \lambda \|\mathbf{w}\|^2$$

where $\lambda > 0$ is a tuning hyperparameter that controls the tradeoff between data fitting and model complexity.

### The Analytical Solution

Taking the gradient of the regularized objective function yields:

$$\nabla_{\mathbf{w}} L_{\text{Ridge}}(\mathbf{w}) = \frac{2}{n}\left( X^T X \mathbf{w} - X^T \mathbf{y} \right) + 2\lambda \mathbf{w} = \mathbf{0}$$

$$X^T X \mathbf{w} + n\lambda \mathbf{w} = X^T \mathbf{y} \implies \left( X^T X + n\lambda I \right)\mathbf{w} = X^T \mathbf{y}$$

Where $I$ is the identity matrix. Solving for $\mathbf{w}$ gives the analytical solution for Ridge Regression:

$$\hat{\mathbf{w}}_{\text{Ridge}} = \left( X^T X + n\lambda I \right)^{-1} X^T \mathbf{y}$$

Adding $n\lambda I$ shifts all eigenvalues upward, guaranteeing that the matrix is strictly positive-definite and invertible, even if $X^T X$ is singular.

## 4. Optimization via Gradient Descent

When the matrix inversion step $\mathcal{O}(d^3)$ is too computationally expensive for high-dimensional feature spaces, we use iterative optimization via **Gradient Descent**.

### Gradient Descent Update Step

We start with an initial weight vector $\mathbf{w}^{(0)}$ and update it iteratively by moving in the direction of steepest local descent (opposite the gradient):

$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla_{\mathbf{w}} L\left(\mathbf{w}^{(t)}\right)$$

where $\eta > 0$ is the learning rate.

### 4.1 Gradient Descent Variants & Update Mechanics

1. **Batch Gradient Descent (BGD):** Computes the exact gradient over the *entire* dataset of size $n$ simultaneously:
   $$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \cdot \frac{1}{n}\sum_{i=1}^n \nabla \mathcal{L}_i(\mathbf{w}^{(t)})$$
   *Pros:* Guaranteed smooth monotonic convergence on convex functions.  
   *Cons:* Computationally prohibitive for large datasets ($\mathcal{O}(nd)$ per step).

2. **Stochastic Gradient Descent (SGD):** Approximates the gradient using a single randomly sampled observation $(\mathbf{x}_i, y_i)$ sampled uniformly $i \sim U(1, n)$ at each step:
   $$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta^{(t)} \nabla \mathcal{L}_i(\mathbf{w}^{(t)})$$
   *For Mean Squared Error (MSE):*
   $$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta^{(t)} \cdot 2\left( \left(\mathbf{w}^{(t)}\right)^T \mathbf{x}_i - y_i \right)\mathbf{x}_i$$
   *Pros:* Extremely fast update cost ($\mathcal{O}(d)$ per step - **2025 Moed A**); stochastic noise helps escape shallow local minima.

3. **Mini-Batch SGD:** Compromise that samples a small subset $\mathcal{B} \subset \{1, \dots, n\}$ of size $b = |\mathcal{B}|$ (typically $b \in [32, 256]$):
   $$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \cdot \frac{1}{b}\sum_{i \in \mathcal{B}} \nabla \mathcal{L}_i(\mathbf{w}^{(t)})$$
   Leverages matrix GPU acceleration while reducing single-sample variance.

---

### 4.2 Theoretical Properties of SGD

*   **Unbiased Estimator Property:**  
    The stochastic gradient derived from a single uniformly sampled instance is an **unbiased estimator** of the true full-batch gradient:
    $$\mathbb{E}_{i \sim U(1, n)}\left[ \nabla \mathcal{L}_i(\mathbf{w}) \right] = \sum_{i=1}^n P(i) \nabla \mathcal{L}_i(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^n \nabla \mathcal{L}_i(\mathbf{w}) = \nabla \mathcal{L}(\mathbf{w})$$

*   **Variance & The Robbins-Monro Convergence Conditions:**  
    Because single-sample updates introduce variance ($\text{Var}(\nabla \mathcal{L}_i) > 0$), SGD with a constant learning rate $\eta$ does *not* converge to a stationary point; instead, it oscillates in a noise ball around the minimum. To guarantee asymptotic convergence to the exact global minimum, the learning rate sequence $\eta^{(t)}$ must satisfy the **Robbins-Monro Conditions**:
    
    $$\sum_{t=1}^\infty \eta^{(t)} = \infty \quad \text{(Learning rate sum must be infinite to reach any point in space)}$$
    $$\sum_{t=1}^\infty \left(\eta^{(t)}\right)^2 < \infty \quad \text{(Square sum must be finite to dampen gradient variance to zero)}$$
    
    *Common Schedules:* Inverse time decay $\eta^{(t)} = \frac{\eta_0}{1 + \alpha t}$ or step decay $\eta^{(t)} = \eta_0 \cdot \gamma^{\lfloor t / k \rfloor}$.
    

## 5. Python / NumPy Implementation Snippet

```python
import numpy as np

class LinearRegressionModels:
    @staticmethod
    def fit_ols(X, y):
        """
        Computes analytical Ordinary Least Squares. X includes a column of ones.
        """
        # w = (X^T * X)^-1 * X^T * y
        return np.linalg.inv(X.T @ X) @ X.T @ y

    @staticmethod
    def fit_ridge(X, y, lmbda):
        """
        Computes analytical Ridge Regression.
        """
        n, d = X.shape
        identity = np.eye(d)
        # w = (X^T * X + n * lambda * I)^-1 * X^T * y
        return np.linalg.inv(X.T @ X + n * lmbda * identity) @ X.T @ y
```

## 6. Solved Exam-Style Examples

### Example 1: Equivalency of Weight Decay and $L_2$ Regularization (From 2025 Moed A)

**Problem Statement:** Prove that applying Stochastic Gradient Descent (SGD) to a loss function with an added $L_2$ regularization term $\frac{\lambda}{2}\|\mathbf{w}\|^2$ is mathematically equivalent to multiplying the weight vector by a shrinking factor $(1 - \eta\lambda)$ at each step prior to adding the standard empirical gradient update (**Weight Decay**).

#### Step-by-Step Analytical Derivation

1. Define the regularized loss function for a single random data sample $i$:
    
    $$\mathcal{L}_{\text{total}}(\mathbf{w}) = \ell_i(\mathbf{w}) + \frac{\lambda}{2}\|\mathbf{w}\|^2$$
    
2. Compute the gradient of this total loss with respect to the weight vector $\mathbf{w}$:
    
    $$\nabla_{\mathbf{w}} \mathcal{L}_{\text{total}}(\mathbf{w}) = \nabla_{\mathbf{w}} \ell_i(\mathbf{w}) + \nabla_{\mathbf{w}}\left(\frac{\lambda}{2}\mathbf{w}^T\mathbf{w}\right) = \nabla_{\mathbf{w}} \ell_i(\mathbf{w}) + \lambda\mathbf{w}$$
    
3. Write out the standard Gradient Descent update step ($\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla_{\mathbf{w}} \mathcal{L}_{\text{total}}$):
    
    $$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \left( \nabla_{\mathbf{w}} \ell_i\left(\mathbf{w}^{(t)}\right) + \lambda\mathbf{w}^{(t)} \right)$$
    
4. Distribute the learning rate parameter $\eta$ across both terms:
    
    $$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla_{\mathbf{w}} \ell_i\left(\mathbf{w}^{(t)}\right) - \eta\lambda\mathbf{w}^{(t)}$$
    
5. Group the terms involving the weight vector $\mathbf{w}^{(t)}$:
    
    $$\mathbf{w}^{(t+1)} = (1 - \eta\lambda)\mathbf{w}^{(t)} - \eta \nabla_{\mathbf{w}} \ell_i\left(\mathbf{w}^{(t)}\right)$$
    

**Final Exam Answer:** This algebraic rearrangement proves that the regularized gradient step is exactly equivalent to shrinking the weights by a factor of $(1 - \eta\lambda)$ before applying the empirical data gradient step, demonstrating that **Weight Decay and $L_2$ Regularization are mathematically identical under gradient descent**.

### Example 2: Probabilistic Derivation of Least Squares (From 2024 Moed A)

**Problem Statement:** Assume that target labels are generated by a true linear function plus added zero-mean Gaussian noise: $y_i = \mathbf{w}^T \mathbf{x}_i + \epsilon_i$, where $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ are independent and identically distributed. Show that finding the Maximum Likelihood Estimate (MLE) for $\mathbf{w}$ is mathematically equivalent to minimizing the Ordinary Least Squares objective function.

#### Step-by-Step Analytical Derivation

1. Because $\mathbf{w}^T \mathbf{x}_i$ is a deterministic constant given $\mathbf{x}_i$, the conditional distribution of the target variable follows a Gaussian distribution:
    
    $$y_i \mid \mathbf{x}_i; \mathbf{w} \sim \mathcal{N}\left(\mathbf{w}^T \mathbf{x}_i, \sigma^2\right)$$
    
2. Write out the conditional probability density function for a single observation:
    
    $$P(y_i \mid \mathbf{x}_i; \mathbf{w}) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{\left(y_i - \mathbf{w}^T \mathbf{x}_i\right)^2}{2\sigma^2} \right)$$
    
3. Construct the log-likelihood function $\ell(\mathbf{w})$ for $n$ i.i.d. observations:
    
    $$\ell(\mathbf{w}) = \sum_{i=1}^n \ln P(y_i \mid \mathbf{x}_i; \mathbf{w}) = \sum_{i=1}^n \left[ -\frac{1}{2}\ln\left(2\pi\sigma^2\right) - \frac{\left(y_i - \mathbf{w}^T \mathbf{x}_i\right)^2}{2\sigma^2} \right]$$
    
    $$\ell(\mathbf{w}) = -\frac{n}{2}\ln\left(2\pi\sigma^2\right) - \frac{1}{2\sigma^2}\sum_{i=1}^n \left(y_i - \mathbf{w}^T \mathbf{x}_i\right)^2$$
    
4. To maximize $\ell(\mathbf{w})$ with respect to $\mathbf{w}$, we can ignore the constant terms that do not depend on $\mathbf{w}$:
    
    $$\arg\max_{\mathbf{w}} \ell(\mathbf{w}) = \arg\max_{\mathbf{w}} \left[ -\frac{1}{2\sigma^2}\sum_{i=1}^n \left(y_i - \mathbf{w}^T \mathbf{x}_i\right)^2 \right]$$
    
5. Since multiplying by a positive constant or changing the sign flips the optimization direction, this is equivalent to:
    
    $$\arg\max_{\mathbf{w}} \ell(\mathbf{w}) = \arg\min_{\mathbf{w}} \sum_{i=1}^n \left(\mathbf{w}^T \mathbf{x}_i - y_i\right)^2$$
    

**Final Exam Answer:** This matches the Ordinary Least Squares minimization objective, proving that **minimizing mean squared error is equivalent to maximizing the data likelihood under the assumption of additive Gaussian noise**.


---
## 🔗 Navigation
**Previous:** [[ ]] | **Next:** [[ ]]