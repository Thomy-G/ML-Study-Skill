# 💡 2025 Moed A — Official Solutions | פתרונות רשמיים — מועד א׳ 2025

---

## 🔹 Part A: True/False Questions / חלק א'

### Question 1
* **a. Correct (True)**. SGD updates parameters using a single sample or mini-batch of size $m \ll n$, requiring $O(m \cdot d)$ operations per step compared to $O(n \cdot d)$ for full batch gradient descent.
* **b. Incorrect (False)**. ADAM is an adaptive learning rate optimizer designed to accelerate convergence. It does not act as a regularizer to prevent overfitting.
* **c. Correct (True)**. BatchNorm normalizes internal activations, reducing internal covariate shift, smoothing loss gradients, and enabling faster network training.
* **d. Correct (True)**. Under standard SGD, weight decay $w \leftarrow (1 - \eta\lambda)w - \eta \nabla L$ is mathematically equivalent to optimizing $L(w) + \frac{\lambda}{2}\|w\|^2$.

### Question 2
* **a. Incorrect (False)**. $\text{VCdim}(H) = d < n$ means $H$ cannot shatter *all* $2^n$ possible labelings of *every* dataset of size $n$. However, there can certainly exist specific datasets $S_n$ and target hypotheses $h \in H$ that classify $S_n$ with 0 error.
* **b. Correct (True)**. For $K=2$, the cluster boundary is the set of points equidistant to $\mu_1, \mu_2$: $\|\mathbf{x} - \mu_1\|^2 = \|\mathbf{x} - \mu_2\|^2 \implies 2(\mu_2 - \mu_1)^T \mathbf{x} + \|\mu_1\|^2 - \|\mu_2\|^2 = 0$, defining a linear hyperplane.
* **c. Correct (True)**. Each feature in a BatchNorm layer has two learnable parameters: scale $\gamma$ and shift $\beta$.

---

## 🔹 Part B: Open Questions / חלק ב'

### Question 3: Logistic Regression
* **a. NLL Loss Derivation:**
  $$P(y_i^{(1)} \mid x_i^{(1)}; w) = \sigma(w x_i^{(1)})^{y_i^{(1)}} \left(1 - \sigma(w x_i^{(1)})\right)^{1 - y_i^{(1)}}$$
  $$\mathcal{L}_{D_1}(w) = -\sum_{i=1}^{n_1} \left[ y_i^{(1)} \ln \sigma(w x_i^{(1)}) + (1 - y_i^{(1)}) \ln (1 - \sigma(w x_i^{(1)})) \right]$$

* **b. Gradient via Chain Rule:**
  Using $\sigma'(z) = \sigma(z)(1-\sigma(z))$:
  $$\frac{\partial}{\partial w} \ln \sigma(w x_i^{(1)}) = \frac{\sigma'(w x_i^{(1)}) x_i^{(1)}}{\sigma(w x_i^{(1)})} = (1 - \sigma(w x_i^{(1)})) x_i^{(1)}$$
  $$\frac{\partial}{\partial w} \ln (1 - \sigma(w x_i^{(1)})) = -\frac{\sigma'(w x_i^{(1)}) x_i^{(1)}}{1 - \sigma(w x_i^{(1)})} = -\sigma(w x_i^{(1)}) x_i^{(1)}$$
  $$\nabla \text{Err}_{D_1}(w) = -\sum_{i=1}^{n_1} \left[ y_i^{(1)} (1 - \sigma) x_i^{(1)} - (1 - y_i^{(1)}) \sigma x_i^{(1)} \right] = -\sum_{i=1}^{n_1} x_i^{(1)} \left(y_i^{(1)} - \sigma(w x_i^{(1)})\right)$$

* **c. Monotonicity Proof:**
  $$\frac{\partial}{\partial w} \nabla \text{Err}_{D_1}(w) = \frac{\partial}{\partial w} \left[ -\sum_{i=1}^{n_1} x_i^{(1)} y_i^{(1)} + \sum_{i=1}^{n_1} x_i^{(1)} \sigma(w x_i^{(1)}) \right] = \sum_{i=1}^{n_1} (x_i^{(1)})^2 \sigma'(w x_i^{(1)})$$
  Since $(x_i^{(1)})^2 \ge 0$ and $\sigma'(z) = \sigma(z)(1-\sigma(z)) > 0$, the derivative is strictly positive, proving $\nabla \text{Err}_{D_1}(w)$ is strictly monotonically increasing.

* **d. Interval Bounds:**
  * **(i)** By definition of sum loss over disjoint sets $D_1 \cup D_2$:
    $$\nabla \text{Err}_{D_1 \cup D_2}(w^*) = \nabla \text{Err}_{D_1}(w^*) + \nabla \text{Err}_{D_2}(w^*)$$
  * **(ii)** At optimal points $w^{(1)}$ and $w^{(2)}$, $\nabla \text{Err}_{D_1}(w^{(1)}) = 0$ and $\nabla \text{Err}_{D_2}(w^{(2)}) = 0$. If $w^* < \min(w^{(1)}, w^{(2)})$, strict monotonicity implies $\nabla \text{Err}_{D_1}(w^*) < 0$ and $\nabla \text{Err}_{D_2}(w^*) < 0 \implies \nabla \text{Err}_{D_1 \cup D_2}(w^*) < 0$. But $w^*$ minimizes $D_1 \cup D_2 \implies \nabla \text{Err}_{D_1 \cup D_2}(w^*) = 0$, contradiction!
  * **(iii)** If $w^* > \max(w^{(1)}, w^{(2)})$, then $\nabla \text{Err}_{D_1 \cup D_2}(w^*) > 0$, contradiction. Thus $\min(w^{(1)}, w^{(2)}) \le w^* \le \max(w^{(1)}, w^{(2)})$.

---

### Question 4: Naïve Bayes
* **a. Conditional Log-Likelihood:**
  $$\ln P(\mathbf{x} \mid Y=c) = \sum_{j=1}^d \ln \mathcal{N}(x_j; \mu_{c,j}, \sigma_j^2) = \sum_{j=1}^d \left[ -\frac{1}{2}\ln(2\pi\sigma_j^2) - \frac{(x_j - \mu_{c,j})^2}{2\sigma_j^2} \right]$$

* **b. Log-Posterior Ratio:**
  $$\ln \frac{P(Y=1 \mid \mathbf{x})}{P(Y=0 \mid \mathbf{x})} = \ln \frac{\pi_1}{\pi_0} + \ln \frac{P(\mathbf{x} \mid Y=1)}{P(\mathbf{x} \mid Y=0)} = \ln \frac{\pi_1}{\pi_0} + \sum_{j=1}^d \left[ -\frac{(x_j - \mu_{1,j})^2}{2\sigma_j^2} + \frac{(x_j - \mu_{0,j})^2}{2\sigma_j^2} \right]$$
  $$-\frac{(x_j - \mu_{1,j})^2 - (x_j - \mu_{0,j})^2}{2\sigma_j^2} = \frac{2x_j(\mu_{1,j} - \mu_{0,j}) + \mu_{0,j}^2 - \mu_{1,j}^2}{2\sigma_j^2}$$
  $$\ln \frac{P(Y=1 \mid \mathbf{x})}{P(Y=0 \mid \mathbf{x})} = \ln \frac{\pi_1}{\pi_0} + \sum_{j=1}^d \frac{\mu_{1,j} - \mu_{0,j}}{\sigma_j^2} x_j + \sum_{j=1}^d \frac{\mu_{0,j}^2 - \mu_{1,j}^2}{2\sigma_j^2}$$

* **c. Linear Decision Boundary:**
  Let $w_j = \frac{\mu_{1,j} - \mu_{0,j}}{\sigma_j^2}$ and $b = \ln \frac{\pi_1}{\pi_0} + \sum_{j=1}^d \frac{\mu_{0,j}^2 - \mu_{1,j}^2}{2\sigma_j^2}$. The log-posterior ratio simplifies to $\mathbf{w}^T \mathbf{x} + b$.
  Decision rule: Classify $Y=1$ if $\mathbf{w}^T \mathbf{x} + b > 0$, else $Y=0$. This is a linear classifier with decision boundary hyper-plane $\mathbf{w}^T \mathbf{x} + b = 0$.

---

### Question 5: $K$-Means & EM
* **a. Joint Likelihood:**
  $$p_\theta(X, Z) = \prod_{i=1}^n \prod_{k=1}^K \left[ \pi_k \mathcal{N}(x_i; \mu_k, \Sigma_k) \right]^{z_{i,k}}$$

* **b. Posterior Responsibilities & Limit $\sigma \to 0$:**
  $$q(z_{i,k}=1) = \frac{P(z_{i,k}=1) P(x_i \mid z_{i,k}=1)}{P(x_i)} = \frac{\pi_k \mathcal{N}(x_i; \mu_k, \sigma^2 I)}{\sum_l \pi_l \mathcal{N}(x_i; \mu_l, \sigma^2 I)}$$
  For $\pi_k = \frac{1}{K}$:
  $$q(z_{i,k}=1) = \frac{\exp\left(-\frac{\|x_i - \mu_k\|^2}{2\sigma^2}\right)}{\sum_{l=1}^K \exp\left(-\frac{\|x_i - \mu_l\|^2}{2\sigma^2}\right)}$$
  As $\sigma \to 0$, the exponential term with the smallest distance $\|x_i - \mu_{k_i^*}\|^2$ dominates all others, yielding $q(z_{i, k_i^*}=1) \to 1$ and $q(z_{i, k}=1) \to 0$ for $k \neq k_i^*$.

* **c. $K$-Means Algorithm & E-Step Equivalence:**
  $K$-Means Step 1 assigns sample $x_i$ to cluster $c_i = \arg\min_k \|x_i - \mu_k\|^2$. In the limit $\sigma \to 0$, the E-step responsibility $q(z_{i,k}=1)$ becomes a binary indicator $\mathbb{I}\{k = c_i\}$, identical to $K$-Means hard assignment.

* **d. M-Step Equivalence:**
  $$\mu_k^{\text{new}} = \frac{\sum_{i=1}^n q(z_{i,k}=1) x_i}{\sum_{i=1}^n q(z_{i,k}=1)} = \frac{\sum_{i \in C_k} x_i}{|C_k|}$$
  which re-estimates cluster means as the arithmetic mean of assigned samples, matching Step 2 of $K$-Means. Thus EM under spherical equal variance as $\sigma \to 0$ executes the exact same steps as $K$-Means.
