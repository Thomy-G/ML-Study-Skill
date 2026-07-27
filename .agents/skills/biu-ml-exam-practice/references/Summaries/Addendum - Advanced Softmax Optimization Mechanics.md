---
type: uni_general
course: "[[English]]"
status: 🟢 Active
date_added: 2026-06-21
order: Summary
---
# Addendum - Advanced Softmax Optimization Mechanics

**MOC:** [[Machine learning MOC]]
**Course:** [[English]]

# Study Guide Addendum: Advanced Softmax Optimization Mechanics

## 1. The Matrix Form of the Gradient (Full Vectorization)

In Topic 9, we derived the gradient with respect to a single class weight vector $\mathbf{w}_m$. In the exam, you may be asked to provide the fully vectorized gradient of the cross-entropy loss with respect to the entire weight matrix $W \in \mathbb{R}^{d \times K}$ for a single sample.

Let:

- $\mathbf{x} \in \mathbb{R}^d$ be the input column vector.
    
- $\mathbf{y} \in \mathbb{R}^K$ be the **one-hot encoded** true label vector (where $y_k = 1$ if class $k$ is the true class, and $0$ otherwise).
    
- $\mathbf{p} \in \mathbb{R}^K$ be the model's predicted probability vector, where $p_k = \text{Softmax}(W^T \mathbf{x})_k$.
    

The complete gradient matrix $\nabla_W \mathcal{L} \in \mathbb{R}^{d \times K}$ is computed via the outer product of the feature vector and the prediction error vector:

$$\nabla_W \mathcal{L} = \mathbf{x} (\mathbf{p} - \mathbf{y})^T$$

### Matrix Structure Breakdown

$$\nabla_W \mathcal{L} = \begin{bmatrix} \mid & \mid & & \mid \\ \nabla_{\mathbf{w}_1}\mathcal{L} & \nabla_{\mathbf{w}_2}\mathcal{L} & \dots & \nabla_{\mathbf{w}_K}\mathcal{L} \\ \mid & \mid & & \mid \end{bmatrix} = \begin{bmatrix} x_1(p_1 - y_1) & x_1(p_2 - y_2) & \dots & x_1(p_K - y_K) \\ x_2(p_1 - y_1) & x_2(p_2 - y_2) & \dots & x_2(p_K - y_K) \\ \vdots & \vdots & \ddots & \vdots \\ x_d(p_1 - y_1) & x_d(p_2 - y_2) & \dots & x_d(p_K - y_K) \end{bmatrix}$$

## 2. Over-Parameterization & Redundancy (The Score Shifting Proof)

A critical theoretical question often asked is: _Is the weight matrix representation of Softmax unique?_ The answer is **no**. Softmax regression is inherently over-parameterized. If you subtract an arbitrary constant vector $\boldsymbol{\psi} \in \mathbb{R}^d$ from every single class weight vector $\mathbf{w}_k$, the predicted class probabilities remain **completely unchanged**.

### Proof of Parameter Redundancy

1. Let $\tilde{\mathbf{w}}_k = \mathbf{w}_k - \boldsymbol{\psi}$ be the shifted weight vectors.
    
2. Compute the new class probability mapping:
    
    $$P(y = k \mid \mathbf{x}; \tilde{W}) = \frac{e^{\tilde{\mathbf{w}}_k^T \mathbf{x}}}{\sum_{j=1}^K e^{\tilde{\mathbf{w}}_j^T \mathbf{x}}} = \frac{e^{(\mathbf{w}_k - \boldsymbol{\psi})^T \mathbf{x}}}{\sum_{j=1}^K e^{(\mathbf{w}_j - \boldsymbol{\psi})^T \mathbf{x}}}$$
    
3. Distribute the matrix transpose and split the exponent:
    
    $$\frac{e^{\mathbf{w}_k^T \mathbf{x} - \boldsymbol{\psi}^T \mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x} - \boldsymbol{\psi}^T \mathbf{x}}} = \frac{e^{\mathbf{w}_k^T \mathbf{x}} \cdot e^{-\boldsymbol{\psi}^T \mathbf{x}}}{\sum_{j=1}^K \left( e^{\mathbf{w}_j^T \mathbf{x}} \cdot e^{-\boldsymbol{\psi}^T \mathbf{x}} \right)}$$
    
4. Factor out $e^{-\boldsymbol{\psi}^T \mathbf{x}}$ from the summation denominator:
    
    $$\frac{e^{\mathbf{w}_k^T \mathbf{x}} \cdot e^{-\boldsymbol{\psi}^T \mathbf{x}}}{e^{-\boldsymbol{\psi}^T \mathbf{x}} \cdot \sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}} = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}} = P(y = k \mid \mathbf{x}; W)$$
    

**Exam Takeaway:** Because of this redundancy, we can technically set one of the class weight vectors (e.g., $\mathbf{w}_K$) completely to $\mathbf{0}$ and vary only the remaining $K-1$ parameters. In practice, instead of setting it to zero, we add **$L_2$ weight regularization** ($\lambda \|W\|_F^2$), which uniquely resolves this scaling freedom by forcing the optimizer to pick the matrix with the smallest Frobenius norm.

## 3. Python Validation: Vectorized Multi-Class Optimization Step

This code piece shows how the vectorized matrix gradient updates parameters during an SGD loop:

```python
import numpy as np

def softmax_gradient_descent_step(X_batch, Y_batch_onehot, W, lr=0.01):
    """
    Performs a mini-batch gradient descent update step over the entire weight matrix.
    X_batch: shape (B, d)
    Y_batch_onehot: shape (B, K) - one-hot matrix representation of ground truth labels
    W: shape (d, K)
    """
    B = X_batch.shape[0]
    
    # Forward Pass: Compute raw logits (B, K) and stabilized probabilities
    logits = np.dot(X_batch, W)
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    P = exp_logits / np.sum(exp_logits, axis=1, keepdims=True) # shape (B, K)
    
    # Compute error matrix: (B, K)
    error = P - Y_batch_onehot
    
    # Full vectorized gradient matrix: (d, B) @ (B, K) -> (d, K)
    gradient_matrix = np.dot(X_batch.T, error) / B
    
    # Update complete weight space simultaneously
    W_updated = W - lr * gradient_matrix
    return W_updated
```

## 4. Solved Exam Problem: Geometric Decision Boundaries of Multi-Class Spaces

**Problem Statement:** Prove that the decision boundaries separating any two classes $m$ and $j$ in a Multinomial Softmax Regression model are always strictly **linear hyperplanes**.

#### Step-by-Step Analytical Derivation

1. The decision boundary line separating class $m$ from class $j$ is defined precisely by the geometric location where their predicted probabilities are identical:
    
    $$P(y = m \mid \mathbf{x}) = P(y = j \mid \mathbf{x})$$
    
2. Substitute the explicit analytical definitions using the Softmax equation:
    
    $$\frac{e^{\mathbf{w}_m^T \mathbf{x} + b_m}}{\sum_{k=1}^K e^{\mathbf{w}_k^T \mathbf{x} + b_k}} = \frac{e^{\mathbf{w}_j^T \mathbf{x} + b_j}}{\sum_{k=1}^K e^{\mathbf{w}_k^T \mathbf{x} + b_k}}$$
    
3. Cancel out the common denominator sum from both sides:
    
    $$e^{\mathbf{w}_m^T \mathbf{x} + b_m} = e^{\mathbf{w}_j^T \mathbf{x} + b_j}$$
    
4. Apply the natural logarithm transformation to eliminate the exponentials:
    
    $$\mathbf{w}_m^T \mathbf{x} + b_m = \mathbf{w}_j^T \mathbf{x} + b_j$$
    
5. Rearrange all variables onto one side to isolate the equation relative to the vector space $\mathbf{x}$:
    
    $$\left( \mathbf{w}_m - \mathbf{w}_j \right)^T \mathbf{x} + \left( b_m - b_j \right) = 0$$
    

**Final Exam Answer:** This matches the classic algebraic standard linear hyperplane formula ($\mathbf{w}_{\text{eff}}^T \mathbf{x} + b_{\text{eff}} = 0$), where $\mathbf{w}_{\text{eff}} = \mathbf{w}_m - \mathbf{w}_j$ and $b_{\text{eff}} = b_m - b_j$. This proves that **even though the Softmax model uses non-linear exponential activations internally, the resulting geometric decision splits between classes remain strictly linear**.

With these additions, your notes cover every variant of Multiclass/Softmax optimization found in past papers.

Should we move forward to **Topic 10: Decision Trees & Information Gain (Class 06B)**?

---
## 🔗 Navigation
**Previous:** [[ ]] | **Next:** [[ ]]