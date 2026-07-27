# 💡 2025 Moed B — Official Solutions | פתרונות רשמיים — מועד ב׳ 2025

---

## 🔹 Part A: True/False Questions / חלק א'

### Question 1
* **a. Correct (True)**. Dropout deactivates a random subset of neurons during each training step with probability $p$, preventing co-adaptation of features and reducing overfitting.
* **b. Incorrect (False)**. Vanishing gradients affect gradient flow during **training** backpropagation, not the forward pass during inference.
* **c. Correct (True)**. Reusing convolution weights across all spatial locations enforces translational equivariance, improving robustness to object position shifts.
* **d. Correct (True)** (in Double Descent / modern deep learning regime). Beyond the interpolation threshold ($p \gg n$), overparameterization acts as implicit regularization, causing test error to decrease (the second descent).

### Question 2
* **a. Correct (True)**. $F_1 = 2 \frac{P \cdot R}{P + R} = \frac{2}{\frac{1}{P} + \frac{1}{R}}$, which is the harmonic mean of precision and recall.
* **b. Incorrect (False)**. DAE objective functions are not restricted to Gaussian noise; they can use masking noise, salt-and-pepper noise, or drop noise paired with binary cross-entropy loss.
* **c. Correct (True)**. A hypothesis class containing two constant functions (e.g. $H = \{h_{+1}, h_{-1}\}$ where $h_{+1}(x) = +1$ and $h_{-1}(x) = -1$) cannot shatter any single point $x$ because for $n=1$, $H$ provides only 2 labelings out of $2^1=2$ without dataset-dependent adaptability. Thus $\text{VCdim}(H) = 0$ while $|H| = 2 > 1$.

---

## 🔹 Part B: Open Questions / חלק ב'

### Question 3: Linear Regression Models
* **a. Unconstrained MSE Linear Regression:**
  Objective $\min_{\mathbf{w}} \frac{1}{n} \|X\mathbf{w} - \mathbf{y}\|^2$. Unique solution exists if $X^T X$ is invertible ($X$ has full column rank $\text{rank}(X) = d$). Closed form: $\mathbf{w}^* = (X^T X)^{-1} X^T \mathbf{y}$.

* **b. Optimal $w_1$ for Fixed $w_2$ in $\hat{y}_i = w_1^2 x_{i,1} + w_2^2 x_{i,2}$:**
  Let $\tilde{y}_i = y_i - w_2^2 x_{i,2}$. Let $A = w_1^2 \ge 0$.
  Objective: $\min_{A \ge 0} J(A) = \frac{1}{n} \sum_{i=1}^n (A x_{i,1} - \tilde{y}_i)^2$.
  Differentiate w.r.t. $A$: $\frac{\partial J}{\partial A} = \frac{2}{n} \sum_{i=1}^n (A x_{i,1} - \tilde{y}_i) x_{i,1} = 0 \implies A^* = \frac{\sum_{i=1}^n \tilde{y}_i x_{i,1}}{\sum_{i=1}^n x_{i,1}^2}$.
  * If $A^* \ge 0$, then $w_1 = \pm \sqrt{A^*}$.
  * If $A^* < 0$, since $w_1^2 \ge 0$, the constrained minimizer is $w_1 = 0$.

* **c. Model A ($y = w^2 x$) vs. Model B ($y = w x$):**
  * Model A is preferred when prior domain knowledge dictates that output predictions must strictly scale with positive slope ($w^2 \ge 0$).
  * Model B is preferred when the true physical relationship has a negative slope (e.g. $y \approx -2x$), which Model A cannot represent because $w^2 \ge 0$ for all $w \in \mathbb{R}$.

* **d. Parameter Multiplicity in Model C ($y = w_1^2 x + w_2 x$):**
  * Model B has **1 unique optimal parameter** $w^*$.
  * Model C prediction equation is $\hat{y} = (w_1^2 + w_2) x$. Any pair $(w_1, w_2)$ satisfying $w_1^2 + w_2 = w^* \implies w_2 = w^* - w_1^2$ achieves identical minimal loss. Since $w_1 \in \mathbb{R}$ can be chosen freely, there are **infinitely many optimal parameter pairs** $(w_1, w_2)$ for Model C.

* **e. Generalization Error Comparison:**
  Model B has 1 parameter (lower model complexity, lower variance). Model C has 2 parameters and parameter redundancy. Gradient descent on Model C will converge to one of the infinitely many global minima depending on initialization. Because Model C has higher variance and unconstrained capacity without regularizing $w_1, w_2$, Model B is expected to achieve equal or superior generalization error (lower test variance).

---

### Question 4: Binary Classification & Soft-SVM
* **a. Soft-SVM Formulation:**
  $$\min_{\mathbf{w}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i \quad \text{s.t. } y_i (\mathbf{w}^T \mathbf{x}_i) \ge 1 - \xi_i, \quad \xi_i \ge 0 \quad \forall i=1,\dots,n$$
  * $\mathbf{w}$: Normal vector defining the decision hyper-plane $\mathbf{w}^T \mathbf{x} = 0$.
  * $\xi_i \ge 0$: Slack variables measuring the margin violation distance of sample $i$.
  * $C > 0$: Tradeoff hyperparameter balancing margin maximization ($\frac{1}{2}\|\mathbf{w}\|^2$) against classification error penalty ($C \sum \xi_i$).

* **b. Orthonormal Points Separability Proof:**
  Let $\mathbf{w} = \sum_{j: y_j=+1} \mathbf{x}_j - \sum_{j: y_j=-1} \mathbf{x}_j$. For any sample $\mathbf{x}_k$ ($1 \le k \le n$):
  $$\mathbf{w}^T \mathbf{x}_k = \left( \sum_{j: y_j=+1} \mathbf{x}_j^T \mathbf{x}_k - \sum_{j: y_j=-1} \mathbf{x}_j^T \mathbf{x}_k \right)$$
  By orthonormality $\mathbf{x}_j^T \mathbf{x}_k = \delta_{jk}$:
  * If $y_k = +1$, $\mathbf{w}^T \mathbf{x}_k = 1 - 0 = +1 \implies y_k (\mathbf{w}^T \mathbf{x}_k) = (+1)(+1) = 1 \ge 1$.
  * If $y_k = -1$, $\mathbf{w}^T \mathbf{x}_k = 0 - 1 = -1 \implies y_k (\mathbf{w}^T \mathbf{x}_k) = (-1)(-1) = 1 \ge 1$.
  Thus $y_k (\mathbf{w}^T \mathbf{x}_k) = 1 \ge 1$ for all $k$, proving linear separability with margin 1!

* **c. Soft-SVM Zero Error Proof for $C = \|\mathbf{w}^*\|^2$:**
  Candidate $(\mathbf{w}^*, \mathbf{0})$ satisfies all constraints with $\boldsymbol{\xi}^* = \mathbf{0}$ because $\mathbf{w}^*$ is the Hard-SVM solution. Its Soft-SVM objective value is:
  $$J_{\text{Soft}}(\mathbf{w}^*, \mathbf{0}) = \frac{1}{2} \|\mathbf{w}^*\|^2 + C \sum_{i=1}^n 0 = \frac{1}{2} \|\mathbf{w}^*\|^2$$
  Let $(\mathbf{w}_{\text{Soft}}, \boldsymbol{\xi}_{\text{Soft}})$ be the optimal Soft-SVM solution. By minimality:
  $$\frac{1}{2} \|\mathbf{w}_{\text{Soft}}\|^2 + C \sum_{i=1}^n \xi_{\text{Soft}, i} \le \frac{1}{2} \|\mathbf{w}^*\|^2$$
  If any training misclassification occurred ($\sum \xi_i \ge 1$), then since $C = \|\mathbf{w}^*\|^2$:
  $$C \sum \xi_{\text{Soft}, i} \ge \|\mathbf{w}^*\|^2 > \frac{1}{2} \|\mathbf{w}^*\|^2$$
  which violates the objective inequality! Thus $\boldsymbol{\xi}_{\text{Soft}} = \mathbf{0}$, proving 0 training error.

---

### Question 5: Non-linear Perceptrons & Kernel Trick
* **a. Explicit Feature Mapping & FLOP Savings:**
  Let $\mathbf{x} = [x_1, x_2]^T, \mathbf{z} = [z_1, z_2]^T$.
  $$K(\mathbf{x}, \mathbf{z}) = (1 + x_1 z_1 + x_2 z_2)^2 = 1 + 2 x_1 z_1 + 2 x_2 z_2 + x_1^2 z_1^2 + 2 x_1 x_2 z_1 z_2 + x_2^2 z_2^2$$
  Matching term-by-term with $\phi(\mathbf{x})^T \phi(\mathbf{z})$:
  $$\phi(\mathbf{x}) = \begin{bmatrix} 1, & \sqrt{2}x_1, & \sqrt{2}x_2, & x_1^2, & \sqrt{2}x_1 x_2, & x_2^2 \end{bmatrix}^T \in \mathbb{R}^6$$
  * Evaluating $K(\mathbf{x}, \mathbf{z})$ in 2D requires **4 FLOPs**.
  * Computing explicit dot product $\phi(\mathbf{x})^T \phi(\mathbf{z})$ in 6D requires **11 FLOPs**.
  * **Saved FLOPs:** $11 - 4 = \mathbf{7 \text{ operations saved per evaluation}}$.

* **b. Standard Perceptron & Hinge Loss SGD Pseudocode:**

```python
import numpy as np

# Standard Perceptron
def train_perceptron(X, y, max_epochs=100):
    w = np.zeros(X.shape[1])
    for epoch in range(max_epochs):
        for i in range(len(y)):
            if y[i] * np.dot(w, X[i]) <= 0:
                w += y[i] * X[i]
    return w

# Perceptron as SGD for Hinge Loss L(w) = max(0, -y_i * w^T x_i) with lr=1
def train_perceptron_sgd(X, y, max_epochs=100):
    w = np.zeros(X.shape[1])
    for epoch in range(max_epochs):
        for i in range(len(y)):
            margin = y[i] * np.dot(w, X[i])
            if margin <= 0:  # Subgradient is -y_i * x_i
                grad = -y[i] * X[i]
                w = w - 1.0 * grad  # w += y_i * x_i
    return w
```

* **c. High-Dimensional Feature Space & Dual Representation:**
  In feature space $\phi(\mathbf{x}) \in \mathbb{R}^D$, starting from $\mathbf{w}_0 = \mathbf{0}$, every update adds $y_i \phi(\mathbf{x}_i)$. Thus by induction:
  $$\mathbf{w} = \sum_{i=1}^n \alpha_i y_i \phi(\mathbf{x}_i)$$
  where $\alpha_i \in \mathbb{N}_0$ counts the number of times instance $i$ was misclassified.

* **d. Kernel Perceptron Pseudocode:**

```python
def train_kernel_perceptron(X, y, kernel_fn, max_epochs=100):
    n = len(y)
    alpha = np.zeros(n)  # Dual misclassification counters
    
    for epoch in range(max_epochs):
        for i in range(n):
            # Compute score using kernel trick: f(x_i) = sum_j alpha_j * y_j * K(x_j, x_i)
            score = sum(alpha[j] * y[j] * kernel_fn(X[j], X[i]) for j in range(n))
            if y[i] * score <= 0:
                alpha[i] += 1
                
    return alpha
```
