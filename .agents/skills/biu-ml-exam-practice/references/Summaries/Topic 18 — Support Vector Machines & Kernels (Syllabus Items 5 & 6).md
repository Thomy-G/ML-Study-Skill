---
type: uni_general
course: "[[Machine learning]]"
status: 🟢 Active
order: Summary
date_added: 2026-06-21
---
# Topic 18 — Support Vector Machines & Kernels (Syllabus Items 5 & 6)

**MOC:** [[Machine learning MOC]]
**Course:** [[Machine learning]]

# Study Guide: Topic 18 — Support Vector Machines & Kernels (Syllabus Items 5 & 6)

## 1. Context: Finding the Maximum-Margin Hyperplane

For a linearly separable binary classification dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ with $y_i \in \{-1, +1\}$, there are infinitely many valid separating hyperplanes ($\mathbf{w}^T\mathbf{x} + b = 0$) that achieve zero training error. The Perceptron algorithm randomly stops at whichever solution it encounters first.

**Support Vector Machines (SVM)** solve this ambiguity by selecting the unique geometric hyperplane that maximizes the **margin**—the physical distance separating the hyperplane from the closest training data points of either class. Maximizing this margin structurally lowers model variance and offers the tightest generalization bounds under PAC theory.

## 2. Mathematical Formalization of Hard-Margin SVM (From 2024 Moed B)

The geometric distance from any point $\mathbf{x}_i$ to a hyperplane parameter space $(\mathbf{w}, b)$ is $\frac{y_i(\mathbf{w}^T\mathbf{x}_i + b)}{\|\mathbf{w}\|}$. We can scale the weights $(\mathbf{w}, b)$ arbitrarily without altering the underlying geometric plane. We fix this scaling freedom by forcing the functional margin of the absolute closest support vectors to equal exactly 1:

$$\min_{i=1,\dots,n} y_i(\mathbf{w}^T\mathbf{x}_i + b) = 1$$

Under this normalization condition, the geometric width of the complete margin channel simplifies elegantly to $\frac{2}{\|\mathbf{w}\|}$. To maximize this physical width, we must minimize its denominator $\|\mathbf{w}\|$.

This defines the primal **Hard-Margin SVM constrained optimization problem**:

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{subject to } y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 \quad \forall i=1,\dots,n$$

---

### 2.1 Soft-Margin SVM (Handling Non-Linearly Separable Data)

When data contains noise or overlap, forcing strict linear separability causes the Hard-Margin optimization problem to become infeasible. **Soft-Margin SVM** (Cortes & Vapnik, 1995) introduces non-negative **slack variables** $\xi_i \ge 0$ that allow instances to violate the margin:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i \quad \text{subject to } y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i \quad \text{and} \quad \xi_i \ge 0 \quad \forall i$$

*   **Slack Variable Meaning ($\xi_i$):**
    *   $\xi_i = 0$: Instance is correctly classified and lies on or outside the margin boundary.
    *   $0 < \xi_i \le 1$: Instance is correctly classified but lies inside the margin channel.
    *   $\xi_i > 1$: Instance is misclassified on the wrong side of the decision boundary.
*   **Trade-off Hyperparameter $C > 0$:**
    *   *Large $C \rightarrow \infty$:* Heavy penalty for margin violations; forces a hard margin (susceptible to overfitting / high variance).
    *   *Small $C \rightarrow 0$:* Permits margin violations for wider channels (higher bias / lower variance).

---

## 3. The Dual SVM Formulation & Constrained Optimization

To handle non-linear extensions, we convert the primal problem into its mathematical **Wolfe Dual** representation using Lagrange multipliers $\alpha_i \geq 0$:

$$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\mathbf{w}^T\mathbf{w} - \sum_{i=1}^n \alpha_i \Big[ y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1 \Big]$$

Taking partial derivatives with respect to the primal variables $\mathbf{w}$ and $b$ and setting them to zero reveals the structural properties of the solution:

1. $\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \mathbf{w} - \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i = \mathbf{0} \implies \mathbf{w}^* = \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i$
    
2. $\frac{\partial \mathcal{L}}{\partial b} = \sum_{i=1}^n \alpha_i y_i = 0$
    

Substituting these relations back into the Lagrangian eliminates the primal parameters, isolating the **Dual SVM Optimization Problem**:

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_i y_j \left( \mathbf{x}_i^T \mathbf{x}_j \right)$$

$$\text{subject to } 0 \leq \alpha_i \leq C \quad \forall i \quad \text{and} \quad \sum_{i=1}^n \alpha_i y_i = 0$$

*(Note: For Hard-Margin SVM, the upper bound is $C = \infty$, giving $\alpha_i \ge 0$).*

### Support Vectors & KKT Complementary Slackness

The Karush-Kuhn-Tucker (KKT) complementary slackness conditions for Soft-Margin SVM state that at the optimal solution:

$$\alpha_i \Big[ y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1 + \xi_i \Big] = 0 \quad \text{and} \quad (C - \alpha_i)\xi_i = 0$$

*   **Non-Support Vectors ($\alpha_i = 0$):** Instance lies strictly outside the margin channel ($y_i f(\mathbf{x}_i) > 1, \xi_i = 0$). It has zero influence on the decision hyperplane.
*   **Margin Support Vectors ($0 < \alpha_i < C$):** Instance lies **exactly on the margin boundary** ($y_i f(\mathbf{x}_i) = 1, \xi_i = 0$). These points are used to compute the bias term $b$:
    $$b = y_i - \mathbf{w}^{*T} \mathbf{x}_i = y_i - \sum_{j=1}^n \alpha_j y_j K(\mathbf{x}_j, \mathbf{x}_i)$$
*   **Non-Margin Support Vectors ($\alpha_i = C$):** Instance violates the margin ($y_i f(\mathbf{x}_i) < 1, \xi_i > 0$). It lies inside the margin channel or on the wrong side of the decision boundary.
    

## 4. Non-Linear Spaces: The Kernel Trick & Mercer's Theorem

If a dataset cannot be separated linearly, we map input vectors into a higher-dimensional feature space using a non-linear mapping function $\phi(\mathbf{x}): \mathbb{R}^d \rightarrow \mathbb{R}^D$. Inspecting the Dual SVM formulation reveals that data features appear *exclusively* inside a standard vector inner product: $\mathbf{x}_i^T \mathbf{x}_j$.

Instead of explicitly computing the high-dimensional mapping $\phi(\mathbf{x})$—which is computationally prohibitive or infinite-dimensional—we define a **Kernel Function** $K(\mathbf{x}_i, \mathbf{x}_j)$ that computes the inner product directly in the original space:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)$$

---

### 4.0 Inputs to Kernel Functions (Training vs. Inference)

A kernel function $K(\mathbf{x}, \mathbf{z})$ takes **two feature vectors from the original input space** ($\mathbf{x}, \mathbf{z} \in \mathbb{R}^d$) and outputs a real scalar measuring their inner product in high-dimensional space:

1. **During Model Training (The Gram Matrix $G_{ij}$):**  
   * **Input 1 ($\mathbf{x}_i$):** Training sample $i \in \mathbb{R}^d$.  
   * **Input 2 ($\mathbf{x}_j$):** Training sample $j \in \mathbb{R}^d$.  
   * The kernel evaluates all pairwise similarities $G_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$ to build the $n \times n$ Gram Matrix used in the Wolfe Dual optimization problem.

2. **During Inference / Prediction (Classifying a Test Point):**  
   * **Input 1 ($\mathbf{x}_{\text{test}}$):** A new, unseen query test instance $\mathbf{x}_{\text{test}} \in \mathbb{R}^d$.  
   * **Input 2 ($\mathbf{x}_i$):** A **Support Vector** ($\mathbf{x}_i \in \text{SV}$, where $\alpha_i > 0$).  
   * The decision rule evaluates the kernel between the test instance and each support vector:  
     $$f(\mathbf{x}_{\text{test}}) = \text{sign}\left( \sum_{i \in \text{SV}} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}_{\text{test}}) + b \right)$$

*Key Advantage:* The inputs $\mathbf{x}$ and $\mathbf{z}$ remain in the original low-dimensional space $\mathbb{R}^d$. You **never** need to construct or evaluate high-dimensional vectors $\phi(\mathbf{x})$ directly.

---

### 4.1 Mercer's Condition for Valid Kernels
A continuous symmetric function $K(\mathbf{x}, \mathbf{z})$ is a valid kernel (i.e. corresponds to an inner product in some Hilbert space) if and only if for any finite set of points $\{\mathbf{x}_1, \dots, \mathbf{x}_n\}$, the **Gram Matrix** $G \in \mathbb{R}^{n \times n}$ defined by $G_{i,j} = K(\mathbf{x}_i, \mathbf{x}_j)$ is **Positive Semi-Definite (PSD)**:

$$\mathbf{c}^T G \mathbf{c} = \sum_{i=1}^n \sum_{j=1}^n c_i c_j K(\mathbf{x}_i, \mathbf{x}_j) \ge 0 \quad \forall \mathbf{c} \in \mathbb{R}^n$$

---

### 4.2 Standard Kernel Functions

1. **Linear Kernel:** (Standard dot product, no feature space expansion)
   $$K_{\text{Linear}}(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T \mathbf{z}$$

2. **Polynomial Kernel:** (Maps to all polynomial feature combinations up to degree $d$)
   $$K_{\text{Poly}}(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T \mathbf{z} + c)^d, \quad c \ge 0$$

3. **Gaussian Radial Basis Function (RBF) Kernel:** (Measures similarity based on Euclidean distance)
   $$K_{\text{RBF}}(\mathbf{x}, \mathbf{z}) = \exp\left(-\frac{\|\mathbf{x} - \mathbf{z}\|^2}{2\sigma^2}\right) = \exp\left(-\gamma \|\mathbf{x} - \mathbf{z}\|^2\right), \quad \gamma > 0$$

---

### 4.3 Proof: Why the RBF Kernel Has Infinite Dimensions

Expanding the squared norm in the RBF kernel gives:
$$K_{\text{RBF}}(\mathbf{x}, \mathbf{z}) = \exp\left(-\frac{\|\mathbf{x}\|^2 - 2\mathbf{x}^T\mathbf{z} + \|\mathbf{z}\|^2}{2\sigma^2}\right) = \exp\left(-\frac{\|\mathbf{x}\|^2}{2\sigma^2}\right) \exp\left(-\frac{\|\mathbf{z}\|^2}{2\sigma^2}\right) \exp\left(\frac{\mathbf{x}^T\mathbf{z}}{\sigma^2}\right)$$

Applying the Taylor series expansion of the exponential function $e^u = \sum_{k=0}^\infty \frac{u^k}{k!}$ to the cross term:
$$\exp\left(\frac{\mathbf{x}^T\mathbf{z}}{\sigma^2}\right) = \sum_{k=0}^\infty \frac{(\mathbf{x}^T\mathbf{z})^k}{k! \, \sigma^{2k}}$$

Since the sum extends to $k=\infty$, this term represents a sum of polynomial kernels of *every integer degree* up to infinity. Thus, the RBF kernel implicitly maps the input data into an **infinite-dimensional Hilbert feature space**, allowing it to draw arbitrarily complex decision boundaries.

## 5. Solved Exam Problem: Proving a Kernel Function (From 2025 Moed B)

**Problem Statement:** Let the input space be 2D vectors: $\mathbf{x}, \mathbf{z} \in \mathbb{R}^2$. Prove that the polynomial function $K(\mathbf{x}, \mathbf{z}) = (1 + \mathbf{x}^T\mathbf{z})^2$ is a valid kernel function by deriving its explicit feature mapping function $\phi(\mathbf{x})$.

### Step-by-Step Analytical Derivation

1. Write down the 2D input vectors explicitly: $\mathbf{x} = [x_1, x_2]^T$ and $\mathbf{z} = [z_1, z_2]^T$.
    
2. Expand the internal vector dot product term:
    
    $$\mathbf{x}^T\mathbf{z} = x_1 z_1 + x_2 z_2$$
    
3. Substitute this expansion into the kernel expression and expand the outer square:
    
    $$K(\mathbf{x}, \mathbf{z}) = \left(1 + x_1 z_1 + x_2 z_2\right)^2$$
    
    $$= (1 + x_1 z_1 + x_2 z_2)(1 + x_1 z_1 + x_2 z_2)$$
    
    $$= 1 + 2x_1 z_1 + 2x_2 z_2 + x_1^2 z_1^2 + 2x_1 x_2 z_1 z_2 + x_2^2 z_2^2$$
    
4. Rearrange this scalar summation to match the structure of a vector inner product ($\phi(\mathbf{x})^T \phi(\mathbf{z}) = \sum_m \phi_m(\mathbf{x})\phi_m(\mathbf{z})$):
    
    $$K(\mathbf{x}, \mathbf{z}) = (1)(1) + (\sqrt{2}x_1)(\sqrt{2}z_1) + (\sqrt{2}x_2)(\sqrt{2}z_2) + (x_1^2)(z_1^2) + (\sqrt{2}x_1 x_2)(\sqrt{2}z_1 z_2) + (x_2^2)(z_2^2)$$
    
5. Isolate the corresponding components to find the explicit feature mapping function $\phi(\mathbf{x}) \in \mathbb{R}^6$:
    
    $$\phi(\mathbf{x}) = \begin{bmatrix} 1, & \sqrt{2}x_1, & \sqrt{2}x_2, & x_1^2, & \sqrt{2}x_1 x_2, & x_2^2 \end{bmatrix}^T$$
    

**Final Exam Answer:** Since we have successfully constructed a feature mapping function $\phi(\mathbf{x})$ such that $K(\mathbf{x}, \mathbf{z}) = \phi(\mathbf{x})^T \phi(\mathbf{z})$, **the function is a valid kernel**. Computing this kernel requires only $\mathcal{O}(d)$ operations, completely avoiding the $\mathcal{O}(d^2)$ operations needed to compute the inner product in the 6D feature space explicitly.

## 6. Python Simulation Snippet: Kernel Evaluation Mapping

```python
import numpy as np

def polynomial_kernel_matrix(X, degree=2, constant=1.0):
    """
    Computes the complete kernel Gram matrix for a dataset X.
    X: Data matrix of shape (n, d)
    Returns: Kernel matrix K of shape (n, n)
    """
    # Compute all pairwise dot products: X @ X.T -> shape (n, n)
    dot_products = np.dot(X, X.T)
    
    # Compute the polynomial kernel element-wise: (constant + x_i^T * x_j)^degree
    K_matrix = (constant + dot_products) ** degree
    return K_matrix
```



---
## 🔗 Navigation
**Previous:** [[ ]] | **Next:** [[ ]]