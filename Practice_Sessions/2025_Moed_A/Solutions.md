# 💡 2025 Moed A — Official Solutions | פתרונות רשמיים — מועד א׳ 2025

---

## 🔹 Part A: True/False Questions / חלק א'

### Question 1
* **a. Correct (True)**. SGD updates parameters using a single sample or mini-batch of size $m \ll n$, requiring $O(m \cdot d)$ operations per step compared to $O(n \cdot d)$ for full batch gradient descent.
* **b. Incorrect (False)**. ADAM is an adaptive learning rate optimizer designed to accelerate convergence. It does not act as a regularizer to prevent overfitting.
* **c. Correct (True)**. BatchNorm normalizes internal activations, reducing internal covariate shift, smoothing loss gradients, and enabling faster network training.
* **d. Correct (True)**. Under standard SGD, weight decay is algebraically identical to adding an $L_2$ regularization term to the loss function.
  
  #### Mathematical Proof:
  1. **SGD on $L_2$ Regularized Loss**:
     Let $\mathcal{L}_0(\mathbf{w})$ be the unregularized loss. Define regularized loss $\mathcal{L}_{\text{reg}}(\mathbf{w}) = \mathcal{L}_0(\mathbf{w}) + \frac{\lambda}{2} \|\mathbf{w}\|^2$.  
     Gradient: $\nabla_{\mathbf{w}} \mathcal{L}_{\text{reg}}(\mathbf{w}) = \nabla_{\mathbf{w}} \mathcal{L}_0(\mathbf{w}) + \lambda \mathbf{w}$.  
     SGD update step with learning rate $\eta > 0$:
     $$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla_{\mathbf{w}} \mathcal{L}_{\text{reg}}(\mathbf{w}_t) = \mathbf{w}_t - \eta \left( \nabla_{\mathbf{w}} \mathcal{L}_0(\mathbf{w}_t) + \lambda \mathbf{w}_t \right)$$
     $$\mathbf{w}_{t+1} = (1 - \eta \lambda) \mathbf{w}_t - \eta \nabla_{\mathbf{w}} \mathcal{L}_0(\mathbf{w}_t)$$

  2. **Weight Decay Update**:
     Weight decay directly scales weights by factor $(1 - \gamma)$ before applying the unregularized gradient step:
     $$\mathbf{w}_{t+1} = (1 - \gamma) \mathbf{w}_t - \eta \nabla_{\mathbf{w}} \mathcal{L}_0(\mathbf{w}_t)$$

  Setting $\gamma = \eta \lambda$ proves **exact algebraic equivalence** for standard SGD!

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

* **d. Interval Bounds Proofs:**
  * **(i) Additivity of Gradients**:  
    Since total loss on dataset union $D_1 \cup D_2$ is the sum of losses $\text{Err}_{D_1 \cup D_2}(w) = \text{Err}_{D_1}(w) + \text{Err}_{D_2}(w)$, taking the derivative yields:
    $$\nabla \text{Err}_{D_1 \cup D_2}(w^*) = \nabla \text{Err}_{D_1}(w^*) + \nabla \text{Err}_{D_2}(w^*)$$

  * **(ii) Proof that $w^* < \min(w^{(1)}, w^{(2)}) \implies \nabla \text{Err}_{D_1 \cup D_2}(w^*) < 0$**:  
    By First-Order Optimality Conditions, $w^{(1)}$ and $w^{(2)}$ minimize $\text{Err}_{D_1}$ and $\text{Err}_{D_2}$ respectively:
    $$\nabla \text{Err}_{D_1}(w^{(1)}) = 0 \quad \text{and} \quad \nabla \text{Err}_{D_2}(w^{(2)}) = 0$$
    If $w^* < \min(w^{(1)}, w^{(2)})$, then $w^* < w^{(1)}$ and $w^* < w^{(2)}$.  
    By strict monotonicity of $\nabla \text{Err}$ (proved in Part 3c):
    $$\nabla \text{Err}_{D_1}(w^*) < \nabla \text{Err}_{D_1}(w^{(1)}) = 0 \implies \nabla \text{Err}_{D_1}(w^*) < 0$$
    $$\nabla \text{Err}_{D_2}(w^*) < \nabla \text{Err}_{D_2}(w^{(2)}) = 0 \implies \nabla \text{Err}_{D_2}(w^*) < 0$$
    Summing both negative terms:
    $$\nabla \text{Err}_{D_1 \cup D_2}(w^*) = \nabla \text{Err}_{D_1}(w^*) + \nabla \text{Err}_{D_2}(w^*) < 0 + 0 = 0 \quad \blacksquare$$

  * **(iii) Proof that $w^* > \max(w^{(1)}, w^{(2)}) \implies \nabla \text{Err}_{D_1 \cup D_2}(w^*) > 0$ and Interval Conclusion**:  
    If $w^* > \max(w^{(1)}, w^{(2)})$, then $w^* > w^{(1)}$ and $w^* > w^{(2)}$.  
    By strict monotonicity:
    $$\nabla \text{Err}_{D_1}(w^*) > \nabla \text{Err}_{D_1}(w^{(1)}) = 0 \implies \nabla \text{Err}_{D_1}(w^*) > 0$$
    $$\nabla \text{Err}_{D_2}(w^*) > \nabla \text{Err}_{D_2}(w^{(2)}) = 0 \implies \nabla \text{Err}_{D_2}(w^*) > 0$$
    Summing both positive terms:
    $$\nabla \text{Err}_{D_1 \cup D_2}(w^*) = \nabla \text{Err}_{D_1}(w^*) + \nabla \text{Err}_{D_2}(w^*) > 0 + 0 = 0 \quad \blacksquare$$

    **Conclusion**: Because $w^*$ is the optimal coefficient on $D_1 \cup D_2$, by First-Order Optimality Condition it must satisfy $\nabla \text{Err}_{D_1 \cup D_2}(w^*) = 0$.  
    From (ii), if $w^* < \min(w^{(1)}, w^{(2)})$, then $\nabla \text{Err}_{D_1 \cup D_2}(w^*) < 0 \neq 0$ (contradiction!).  
    From (iii), if $w^* > \max(w^{(1)}, w^{(2)})$, then $\nabla \text{Err}_{D_1 \cup D_2}(w^*) > 0 \neq 0$ (contradiction!).  
    Therefore, $w^*$ must lie bounded inside the interval:
    $$\min(w^{(1)}, w^{(2)}) \le w^* \le \max(w^{(1)}, w^{(2)}) \quad \blacksquare$$

---

### Question 4: Naïve Bayes
* **a. Conditional Log-Likelihood:**
  $$\ln P(\mathbf{x} \mid Y=c) = \sum_{j=1}^d \ln \mathcal{N}(x_j; \mu_{c,j}, \sigma_j^2) = \sum_{j=1}^d \left[ -\frac{1}{2}\ln(2\pi\sigma_j^2) - \frac{(x_j - \mu_{c,j})^2}{2\sigma_j^2} \right]$$

* **b. Log-Posterior Ratio (Step-by-Step Derivation):**

  #### Step 1: Apply Bayes' Rule to Posterior Ratio
  $$\frac{P(Y=1 \mid \mathbf{x})}{P(Y=0 \mid \mathbf{x})} = \frac{P(Y=1)}{P(Y=0)} \cdot \frac{P(\mathbf{x} \mid Y=1)}{P(\mathbf{x} \mid Y=0)} = \frac{\pi_1}{\pi_0} \cdot \frac{\prod_{j=1}^d P(x_j \mid Y=1)}{\prod_{j=1}^d P(x_j \mid Y=0)}$$

  #### Step 2: Take Natural Logarithm ($\ln$)
  $$\ln \left( \frac{P(Y=1 \mid \mathbf{x})}{P(Y=0 \mid \mathbf{x})} \right) = \ln \left( \frac{\pi_1}{\pi_0} \right) + \sum_{j=1}^d \left[ \ln P(x_j \mid Y=1) - \ln P(x_j \mid Y=0) \right]$$

  #### Step 3: Substitute 1D Gaussian Log-Likelihoods & Cancel Shared Constants
  Since $P(x_j \mid Y=c) = \frac{1}{\sqrt{2\pi\sigma_j^2}} \exp\left(-\frac{(x_j - \mu_{c,j})^2}{2\sigma_j^2}\right)$, taking the log and subtracting:
  $$\ln P(x_j \mid Y=1) - \ln P(x_j \mid Y=0) = \left[ -\frac{1}{2}\ln(2\pi\sigma_j^2) - \frac{(x_j - \mu_{1,j})^2}{2\sigma_j^2} \right] - \left[ -\frac{1}{2}\ln(2\pi\sigma_j^2) - \frac{(x_j - \mu_{0,j})^2}{2\sigma_j^2} \right]$$
  The normalization term $-\frac{1}{2}\ln(2\pi\sigma_j^2)$ cancels out completely because the variance $\sigma_j^2$ is shared:
  $$= \frac{(x_j - \mu_{0,j})^2 - (x_j - \mu_{1,j})^2}{2\sigma_j^2}$$

  #### Step 4: Expand Difference of Squares & Cancel $x_j^2$
  $$(x_j - \mu_{0,j})^2 - (x_j - \mu_{1,j})^2 = (x_j^2 - 2x_j\mu_{0,j} + \mu_{0,j}^2) - (x_j^2 - 2x_j\mu_{1,j} + \mu_{1,j}^2) = 2x_j(\mu_{1,j} - \mu_{0,j}) + \mu_{0,j}^2 - \mu_{1,j}^2$$
  Dividing by $2\sigma_j^2$:
  $$\frac{2x_j(\mu_{1,j} - \mu_{0,j}) + \mu_{0,j}^2 - \mu_{1,j}^2}{2\sigma_j^2} = \frac{\mu_{1,j} - \mu_{0,j}}{\sigma_j^2} x_j + \frac{\mu_{0,j}^2 - \mu_{1,j}^2}{2\sigma_j^2}$$

  #### Step 5: Final Log-Posterior Ratio Formula
  $$\ln \left( \frac{P(Y=1 \mid \mathbf{x})}{P(Y=0 \mid \mathbf{x})} \right) = \ln \left( \frac{\pi_1}{\pi_0} \right) + \sum_{j=1}^d \frac{\mu_{1,j} - \mu_{0,j}}{\sigma_j^2} x_j + \sum_{j=1}^d \frac{\mu_{0,j}^2 - \mu_{1,j}^2}{2\sigma_j^2} \quad \blacksquare$$

* **c. Linear Decision Boundary & Parameter Selection Explanation:**

  #### 1. What Defines a Linear Classifier?
  A binary classifier $f: \mathbb{R}^d \to \{0, 1\}$ is a **Linear Classifier** if its decision boundary separating class 1 from class 0 is a flat hyper-plane in feature space defined by a linear equation:
  $$\mathbf{w}^T \mathbf{x} + b = 0 \quad \iff \quad \sum_{j=1}^d w_j x_j + b = 0$$
  The decision rule classifies $Y=1$ if $\mathbf{w}^T \mathbf{x} + b > 0$, and $Y=0$ otherwise.  
  *Key Property*: The decision score depends **linearly** on input features $\mathbf{x} = (x_1, \dots, x_d)$. Quadratic terms ($x_j^2$), cross-feature terms ($x_i x_j$), or non-linear activation transformations are completely absent from the decision boundary function.

  #### 2. Why Select $w_j$ and $b$ as We Did?
  By Bayes' Optimal Decision Rule, we predict $Y=1$ whenever $P(Y=1 \mid \mathbf{x}) > P(Y=0 \mid \mathbf{x})$, which is equivalent to:
  $$\frac{P(Y=1 \mid \mathbf{x})}{P(Y=0 \mid \mathbf{x})} > 1 \iff \ln \left( \frac{P(Y=1 \mid \mathbf{x})}{P(Y=0 \mid \mathbf{x})} \right) > 0$$

  Equating the log-posterior ratio derived in Part 4b to the standard linear classifier form $\mathbf{w}^T \mathbf{x} + b > 0$:
  $$\underbrace{\sum_{j=1}^d \left( \frac{\mu_{1,j} - \mu_{0,j}}{\sigma_j^2} \right) x_j}_{\text{Linear feature terms } \mathbf{w}^T \mathbf{x}} + \underbrace{\left[ \ln \left( \frac{\pi_1}{\pi_0} \right) + \sum_{j=1}^d \frac{\mu_{0,j}^2 - \mu_{1,j}^2}{2\sigma_j^2} \right]}_{\text{Constant bias } b} > 0$$

  By matching corresponding coefficients:
  1. **Weight Vector Component $w_j = \frac{\mu_{1,j} - \mu_{0,j}}{\sigma_j^2}$**:
     This is the coefficient multiplying input feature $x_j$. It measures the distance between class means $(\mu_{1,j} - \mu_{0,j})$ normalized by feature variance $\sigma_j^2$. If class 1 has a higher mean than class 0 ($\mu_{1,j} > \mu_{0,j}$), $w_j > 0$, meaning a larger $x_j$ increases the probability of class 1.
  2. **Bias Scalar $b = \ln \left( \frac{\pi_1}{\pi_0} \right) + \sum_{j=1}^d \frac{\mu_{0,j}^2 - \mu_{1,j}^2}{2\sigma_j^2}$**:
     This collects all constant terms independent of $\mathbf{x}$. It combines the log-prior ratio $\ln(\pi_1 / \pi_0)$ (which shifts the boundary toward the less probable class) and the sum of normalized squared class means.

  Because $x_j^2$ terms canceled out in Part 4b (due to shared variance $\sigma_j^2$), the log-posterior ratio is strictly linear in $\mathbf{x}$, proving that Gaussian Naïve Bayes with shared variances is **exactly a Linear Classifier**! $\quad \blacksquare$

---

### Question 5: $K$-Means & Expectation-Maximization (EM)

* **a. Joint Likelihood with Latent Variables:**
  Let $\mathbf{z}_i = (z_{i,1}, \dots, z_{i,K})$ be a 1-of-$K$ binary latent indicator vector for sample $x_i$, where $z_{i,k} = 1$ if $x_i$ belongs to Gaussian component $k$, and $0$ otherwise ($\sum_{k=1}^K z_{i,k} = 1$).  
  For a single sample:
  $$p_\theta(x_i, \mathbf{z}_i) = \prod_{k=1}^K \left[ P(z_{i,k}=1) P(x_i \mid z_{i,k}=1) \right]^{z_{i,k}} = \prod_{k=1}^K \left[ \pi_k \mathcal{N}(x_i; \mu_k, \Sigma_k) \right]^{z_{i,k}}$$
  For $n$ i.i.d. samples, the full joint likelihood is:
  $$p_\theta(X, Z) = \prod_{i=1}^n \prod_{k=1}^K \left[ \pi_k \mathcal{N}(x_i; \mu_k, \Sigma_k) \right]^{z_{i,k}}$$

---

* **b. Posterior Responsibilities & Limit $\sigma \to 0$ Proof:**
  Using Bayes' Rule, the posterior responsibility $q(z_{i,k}=1) \equiv P(z_{i,k}=1 \mid x_i; \theta)$ is:
  $$q(z_{i,k}=1) = \frac{P(z_{i,k}=1) P(x_i \mid z_{i,k}=1)}{P(x_i)} = \frac{\pi_k \mathcal{N}(x_i; \mu_k, \Sigma_k)}{\sum_{l=1}^K \pi_l \mathcal{N}(x_i; \mu_l, \Sigma_l)}$$

  Assuming equal priors $\pi_k = \frac{1}{K}$ and spherical covariance $\Sigma_k = \sigma^2 I$:
  $$\mathcal{N}(x_i; \mu_k, \sigma^2 I) = \frac{1}{(2\pi \sigma^2)^{d/2}} \exp\left(-\frac{\|x_i - \mu_k\|^2}{2\sigma^2}\right)$$
  The prior terms $\pi_k$ and normalization factors $\frac{1}{(2\pi \sigma^2)^{d/2}}$ cancel out from numerator and denominator:
  $$q(z_{i,k}=1) = \frac{\exp\left(-\frac{\|x_i - \mu_k\|^2}{2\sigma^2}\right)}{\sum_{l=1}^K \exp\left(-\frac{\|x_i - \mu_l\|^2}{2\sigma^2}\right)}$$

  Let $\mu_{k^*}$ be the unique closest centroid to sample $x_i$ ($k^* = \arg\min_l \|x_i - \mu_l\|^2$).  
  Divide numerator and denominator by $\exp\left(-\frac{\|x_i - \mu_{k^*}\|^2}{2\sigma^2}\right)$:
  $$q(z_{i,k}=1) = \frac{\exp\left( -\frac{\|x_i - \mu_k\|^2 - \|x_i - \mu_{k^*}\|^2}{2\sigma^2} \right)}{1 + \sum_{l \neq k^*} \exp\left( -\frac{\|x_i - \mu_l\|^2 - \|x_i - \mu_{k^*}\|^2}{2\sigma^2} \right)}$$

  Taking the limit as variance $\sigma \to 0$:
  1. For $k = k^*$: The numerator exponent is $0 \implies e^0 = 1$. The denominator sum terms have $\|x_i - \mu_l\|^2 - \|x_i - \mu_{k^*}\|^2 > 0$, so $e^{-\infty} \to 0$. Thus:
     $$\lim_{\sigma \to 0} q(z_{i, k^*}=1) = \frac{1}{1 + 0} = 1$$
  2. For $k \neq k^*$: The numerator exponent has $\|x_i - \mu_k\|^2 - \|x_i - \mu_{k^*}\|^2 > 0$, so $e^{-\infty} \to 0$. Thus:
     $$\lim_{\sigma \to 0} q(z_{i, k}=1) = 0$$

  Therefore, as $\sigma \to 0$, the soft responsibilities become binary hard indicators $q(z_{i,k}=1) \in \{0, 1\}$. $\quad \blacksquare$

---

* **c. $K$-Means Algorithm & E-Step Equivalence:**
  * **$K$-Means Algorithm**:
    - **Step 1 (Hard Assignment)**: Assign each point $x_i$ to nearest cluster centroid $c_i = \arg\min_k \|x_i - \mu_k\|^2$.
    - **Step 2 (Centroid Update)**: Re-estimate $\mu_k = \frac{1}{|S_k|} \sum_{i \in S_k} x_i$, where $S_k = \{i \mid c_i = k\}$.
  * **E-Step Equivalence**:
    From Part (b), as $\sigma \to 0$, the E-step responsibilities satisfy:
    $$q(z_{i,k}=1) = \mathbb{I}\left\{ k = \arg\min_l \|x_i - \mu_l\|^2 \right\}$$
    This indicator function is mathematically identical to Step 1 of $K$-Means (assigning point $x_i$ to the nearest centroid).

---

* **d. M-Step Equivalence & Full Algorithmic Identity:**
  The GMM M-step update formula for cluster centroid $\mu_k$ is:
  $$\mu_k^{\text{new}} = \frac{\sum_{i=1}^n q(z_{i,k}=1) x_i}{\sum_{i=1}^n q(z_{i,k}=1)}$$

  Substituting the limiting binary responsibilities $q(z_{i,k}=1) = \mathbb{I}\{i \in S_k\}$ as $\sigma \to 0$:
  - Numerator: $\sum_{i=1}^n q(z_{i,k}=1) x_i = \sum_{i \in S_k} x_i$
  - Denominator: $\sum_{i=1}^n q(z_{i,k}=1) = \sum_{i \in S_k} 1 = |S_k|$

  Thus, the M-step becomes:
  $$\mu_k^{\text{new}} = \frac{1}{|S_k|} \sum_{i \in S_k} x_i$$
  which is **exact Step 2 of $K$-Means** (computing arithmetic mean of assigned cluster samples).

  **Conclusion**: Because the E-step matches Step 1 of $K$-Means and the M-step matches Step 2 of $K$-Means in the limit $\sigma \to 0$ (under equal priors and isotropic variance), **EM on a GMM is mathematically identical to $K$-Means clustering**! $\quad \blacksquare$
