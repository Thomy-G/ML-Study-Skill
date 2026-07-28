---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Resource
date_added: 2026-07-27
---
# 2025 Machine Learning Past Exams (Moed A & B)

**Course:** [[English]] (Bar-Ilan University)  
**Instructors:** Prof. Joseph Keshet & Prof. Gal Chechik  
**Master Index:** [[Past Exams Master Index]]

---

## 📝 2025 Moed A — Full Exam Paper & Official Solutions

### 🔹 Part A: True/False Questions (21 Points Total)

#### Question 1 (12 Points)
Mark whether each statement below is **Correct (True)** or **Incorrect (False)**:
* **a.** Stochastic gradient descent has fewer computations per update than full gradient descent.  
  * **Answer:** **Correct (True)**. SGD computes gradients using a single sample (or mini-batch) of size $m \ll n$, requiring $O(m \cdot d)$ FLOPs compared to $O(n \cdot d)$ for full batch gradient descent.
* **b.** Using ADAM instead of using SGD reduces overfitting.  
  * **Answer:** **Incorrect (False)**. Adam is an adaptive learning rate optimizer designed to accelerate convergence. It does not inherently regularize the model to reduce overfitting; in fact, Adam without weight decay can overfit more than SGD.
* **c.** BatchNorm makes training of deep networks converge faster.  
  * **Answer:** **Correct (True)**. Batch Normalization stabilizes layer activation distributions, smoothes the optimization loss landscape, and permits higher learning rates.
* **d.** With SGD, Weight Decay has the equivalent effect of adding an $L_2$ regularization term to the loss function.  
  * **Answer:** **Correct (True)**. Under standard SGD, weight decay is algebraically identical to optimizing an $L_2$ regularized loss function $\mathcal{L}_{\text{reg}}(\mathbf{w}) = \mathcal{L}_0(\mathbf{w}) + \frac{\lambda}{2}\|\mathbf{w}\|^2$.  
    *Proof*:
    1. SGD on $L_2$ regularized loss: $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla \mathcal{L}_{\text{reg}} = \mathbf{w}_t - \eta \left( \nabla \mathcal{L}_0 + \lambda \mathbf{w}_t \right) = (1 - \eta \lambda)\mathbf{w}_t - \eta \nabla \mathcal{L}_0$.
    2. Weight decay update: $\mathbf{w}_{t+1} = (1 - \gamma)\mathbf{w}_t - \eta \nabla \mathcal{L}_0$.
    Setting $\gamma = \eta \lambda$ yields **exact algebraic identity**.

#### Question 2 (9 Points)
Mark whether each statement below is **Correct (True)** or **Incorrect (False)**:
* **a.** Let $S_n = \{(x_i, y_i)\}_{i=1}^n$ be a set of $n$ labeled samples and $H$ be a hypothesis class with $\text{VCdim}(H) = d < n$. Then, there cannot be $h \in H$ such that $h$ classifies the samples without errors.  
  * **Answer:** **Incorrect (False)**. $\text{VCdim}(H) = d < n$ means $H$ cannot shatter *every* arbitrary dataset of size $n$. However, there can easily exist specific datasets $S_n$ and target hypotheses $h \in H$ that classify $S_n$ with 0 training error.
* **b.** $K$-Means clustering partitions the input space with a linear decision boundary when $K=2$.  
  * **Answer:** **Correct (True)**. For $K=2$, the cluster boundary is the set of points equidistant from cluster centers $\mu_1, \mu_2$: $\|\mathbf{x} - \mu_1\|^2 = \|\mathbf{x} - \mu_2\|^2 \implies 2(\mu_2 - \mu_1)^T \mathbf{x} + \|\mu_1\|^2 - \|\mu_2\|^2 = 0$, which defines a linear hyperplane.
* **c.** A BatchNorm layer has two learnable parameters per feature, for shift and scale.  
  * **Answer:** **Correct (True)**. The learnable parameters are the scale parameter $\gamma_j$ and shift parameter $\beta_j$ in $y_j = \gamma_j \hat{x}_j + \beta_j$.

---

### 🔹 Part B: Open Questions (Choose 2 out of 3, 40 Points Each)

#### Question 3: Logistic Regression
Consider training logistic regression models for two datasets $D_1, D_2$ with binary labels $D_1 = \{(x_i^{(1)}, y_i^{(1)})\}_{i=1}^{n_1}, D_2 = \{(x_i^{(2)}, y_i^{(2)})\}_{i=1}^{n_2}$ where $x_i^{(1)}, x_i^{(2)} \ge 0$ and $y_i^{(1)}, y_i^{(2)} \in \{0, 1\}$. Assume datasets are distinct.

* **a. Derive NLL Loss for $D_1$:**
  $$\mathcal{L}_{D_1}(w) = -\sum_{i=1}^{n_1} \left[ y_i^{(1)} \ln \sigma(w x_i^{(1)}) + (1 - y_i^{(1)}) \ln (1 - \sigma(w x_i^{(1)})) \right]$$
* **b. Gradient Derivation:**
  Using $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ and the chain rule:
  $$\nabla \text{Err}_{D_1}(w) = -\sum_{i=1}^{n_1} x_i^{(1)} \left( y_i^{(1)} - \sigma(w x_i^{(1)}) \right)$$
* **c. Monotonicity of Gradient:**
  Differentiating $\nabla \text{Err}_{D_1}(w)$ w.r.t. $w$:
  $$\frac{\partial}{\partial w} \nabla \text{Err}_{D_1}(w) = \sum_{i=1}^{n_1} (x_i^{(1)})^2 \sigma'(w x_i^{(1)}) > 0$$
  Since $(x_i^{(1)})^2 \ge 0$ and $\sigma'(z) > 0$, the gradient is strictly monotonically increasing in $w$.
* **d. Optimal Weight Interval Bounds Proofs:**
  * **(i)** Additivity of gradients: $\text{Err}_{D_1 \cup D_2}(w) = \text{Err}_{D_1}(w) + \text{Err}_{D_2}(w) \implies \nabla \text{Err}_{D_1 \cup D_2}(w^*) = \nabla \text{Err}_{D_1}(w^*) + \nabla \text{Err}_{D_2}(w^*)$.
  * **(ii)** By FOC, $\nabla \text{Err}_{D_1}(w^{(1)}) = 0$ and $\nabla \text{Err}_{D_2}(w^{(2)}) = 0$. If $w^* < \min(w^{(1)}, w^{(2)})$, strict monotonicity implies $\nabla \text{Err}_{D_1}(w^*) < \nabla \text{Err}_{D_1}(w^{(1)}) = 0$ and $\nabla \text{Err}_{D_2}(w^*) < \nabla \text{Err}_{D_2}(w^{(2)}) = 0 \implies \nabla \text{Err}_{D_1 \cup D_2}(w^*) = \nabla \text{Err}_{D_1}(w^*) + \nabla \text{Err}_{D_2}(w^*) < 0$.
  * **(iii)** If $w^* > \max(w^{(1)}, w^{(2)})$, strict monotonicity implies $\nabla \text{Err}_{D_1}(w^*) > 0$ and $\nabla \text{Err}_{D_2}(w^*) > 0 \implies \nabla \text{Err}_{D_1 \cup D_2}(w^*) > 0$.
  * **Conclusion**: Since $w^*$ minimizes $D_1 \cup D_2$, by FOC $\nabla \text{Err}_{D_1 \cup D_2}(w^*) = 0$. Since $w^* < \min$ yields negative gradient and $w^* > \max$ yields positive gradient, $w^*$ must lie bounded within $\min(w^{(1)}, w^{(2)}) \le w^* \le \max(w^{(1)}, w^{(2)})$.

#### Question 4: Naïve Bayes
Let $\mathbf{x} = (x_1, \dots, x_d) \in \mathbb{R}^d$ and $Y \in \{0, 1\}$ with prior $P(Y=1) = \pi_1, P(Y=0) = \pi_0$. Feature distributions are $P(x_j \mid Y=c) = \mathcal{N}(\mu_{c,j}, \sigma_j^2)$ with shared variance $\sigma_j^2$.
* **a. Conditional Log-Likelihood:**
  $$\ln P(\mathbf{x} \mid Y=c) = \sum_{j=1}^d \left[ -\frac{1}{2}\ln(2\pi\sigma_j^2) - \frac{(x_j - \mu_{c,j})^2}{2\sigma_j^2} \right]$$
* **b. Log-Posterior Ratio (Step-by-Step Derivation):**
  * Step 1 (Bayes' Rule): $\frac{P(Y=1 \mid \mathbf{x})}{P(Y=0 \mid \mathbf{x})} = \frac{\pi_1}{\pi_0} \cdot \frac{\prod_{j=1}^d P(x_j \mid Y=1)}{\prod_{j=1}^d P(x_j \mid Y=0)}$.
  * Step 2 (Taking Log): $\ln \frac{P(Y=1 \mid \mathbf{x})}{P(Y=0 \mid \mathbf{x})} = \ln \frac{\pi_1}{\pi_0} + \sum_{j=1}^d \left[ \ln P(x_j \mid Y=1) - \ln P(x_j \mid Y=0) \right]$.
  * Step 3 (Gaussian Log & Shared $\sigma_j^2$ Cancellation): $\ln P(x_j \mid Y=1) - \ln P(x_j \mid Y=0) = \frac{(x_j - \mu_{0,j})^2 - (x_j - \mu_{1,j})^2}{2\sigma_j^2}$.
  * Step 4 (Difference of Squares & $x_j^2$ Cancellation): $(x_j - \mu_{0,j})^2 - (x_j - \mu_{1,j})^2 = 2x_j(\mu_{1,j} - \mu_{0,j}) + \mu_{0,j}^2 - \mu_{1,j}^2$.
  * Step 5 (Final Formula):
    $$\ln \frac{P(Y=1 \mid \mathbf{x})}{P(Y=0 \mid \mathbf{x})} = \ln \frac{\pi_1}{\pi_0} + \sum_{j=1}^d \frac{\mu_{1,j} - \mu_{0,j}}{\sigma_j^2} x_j + \sum_{j=1}^d \frac{\mu_{0,j}^2 - \mu_{1,j}^2}{2\sigma_j^2}$$
* **c. Linear Classifier Equivalence & Parameter Selection Rationale:**
  * **Linear Classifier Definition**: A classifier is linear if its decision score depends purely on a linear combination of input features $\mathbf{w}^T \mathbf{x} + b > 0$, without quadratic ($x_j^2$) or cross-feature terms.
  * **Parameter Selection**: Equating Bayes optimal decision rule $\ln\left(\frac{P(Y=1 \mid \mathbf{x})}{P(Y=0 \mid \mathbf{x})}\right) > 0$ to $\mathbf{w}^T \mathbf{x} + b > 0$:
    1. $w_j = \frac{\mu_{1,j} - \mu_{0,j}}{\sigma_j^2}$ (coefficient of feature $x_j$, measuring mean separation scaled by feature variance).
    2. $b = \ln \frac{\pi_1}{\pi_0} + \sum_{j=1}^d \frac{\mu_{0,j}^2 - \mu_{1,j}^2}{2\sigma_j^2}$ (constant bias combining prior odds and squared mean shifts).
  * **Decision Rule**: Classify $Y=1$ if $\mathbf{w}^T \mathbf{x} + b > 0$, else $Y=0$. Decision boundary is hyper-plane $\mathbf{w}^T \mathbf{x} + b = 0$.

#### Question 5: $K$-Means & Expectation-Maximization (EM)
* **a. Full Likelihood with Latent $z_{i,k}$:**
  For 1-of-$K$ binary indicator vector $\mathbf{z}_i$:
  $$p_\theta(X, Z) = \prod_{i=1}^n \prod_{k=1}^K \left[ \pi_k \mathcal{N}(x_i; \mu_k, \Sigma_k) \right]^{z_{i,k}}$$
* **b. Posterior Responsibilities & Limit $\sigma \to 0$ Proof:**
  For $\pi_k = 1/K$ and $\Sigma_k = \sigma^2 I$:
  $$q(z_{i,k}=1) = \frac{\exp\left(-\frac{\|x_i - \mu_k\|^2}{2\sigma^2}\right)}{\sum_{l=1}^K \exp\left(-\frac{\|x_i - \mu_l\|^2}{2\sigma^2}\right)}$$
  Dividing by $\exp\left(-\frac{\|x_i - \mu_{k^*}\|^2}{2\sigma^2}\right)$ where $k^* = \arg\min_l \|x_i - \mu_l\|^2$:
  As $\sigma \to 0$, $e^{-\frac{\|x_i - \mu_l\|^2 - \|x_i - \mu_{k^*}\|^2}{2\sigma^2}} \to e^{-\infty} = 0$ for all $l \neq k^*$. Thus $q(z_{i,k^*}=1) \to 1$ and $q(z_{i,k}=1) \to 0$ for $k \neq k^*$.
* **c. E-Step Equivalence to $K$-Means Step 1:**
  As $\sigma \to 0$, $q(z_{i,k}=1) = \mathbb{I}\{k = \arg\min_l \|x_i - \mu_l\|^2\}$, which is identical to Step 1 of $K$-Means (hard nearest-centroid assignment).
* **d. M-Step Equivalence & Algorithmic Identity:**
  $$\mu_k^{\text{new}} = \frac{\sum_{i=1}^n q(z_{i,k}=1) x_i}{\sum_{i=1}^n q(z_{i,k}=1)} = \frac{\sum_{i \in S_k} x_i}{|S_k|}$$
  which is exact Step 2 of $K$-Means (re-estimating centroid as arithmetic mean of assigned points). Thus EM for GMM in the limit $\sigma \to 0$ is mathematically identical to $K$-Means clustering! $\quad \blacksquare$

---

## 📝 2025 Moed B — Full Exam Paper & Official Solutions

### 🔹 Part A: True/False Questions (21 Points Total)

#### Question 1 (12 Points)
* **a.** Dropout reduces overfitting during neural network training.  
  * **Answer:** **Correct (True)**. Dropout randomly deactivates neurons during training with probability $p$, preventing co-adaptation of feature detectors.
* **b.** Residual connections in ResNets improve the inference stage in very deep networks due to the vanishing gradient problem.  
  * **Answer:** **Incorrect (False)**. Vanishing gradients affect gradient backpropagation during **training**, not the forward pass during inference.
* **c.** Parameter sharing in convolutional networks makes them more robust to translations of objects in the image.  
  * **Answer:** **Correct (True)**. Reusing the same convolution kernel across all spatial locations enforces translational equivariance.
* **d.** In cases where $p \gg n$ (many more parameters than data), increasing model complexity will usually lead to a smaller test error.  
  * **Answer:** **Correct (True)** (in modern deep learning / Double Descent context). Past the interpolation threshold ($p > n$), overparameterization acts as implicit regularization, causing test error to decrease (the second descent of Double Descent).

#### Question 2 (9 Points)
* **a.** F1-score is the harmonic mean of recall and precision.  
  * **Answer:** **Correct (True)**. $F_1 = 2 \frac{P \cdot R}{P + R} = \frac{2}{\frac{1}{P} + \frac{1}{R}}$.
* **b.** The reconstruction objective function in a Denoising Autoencoder (DAE) assumes the corruption process is governed by Gaussian noise.  
  * **Answer:** **Incorrect (False)**. DAEs can use masking noise, salt-and-pepper noise, or drop noise with cross-entropy or binary loss functions, not solely Gaussian noise.
* **c.** A hypothesis class with a VC-dimension of 0 can contain more than one hypothesis.  
  * **Answer:** **Correct (True)**. A class $H = \{h_1, h_2\}$ containing two constant functions (e.g. $h_1(x)=+1, h_2(x)=-1$) cannot shatter any single point $x$ because for $n=1$, it produces only 2 labelings out of $2^1=2$ without arbitrary flexibility, or if both output $+1$. Thus $\text{VCdim}(H) = 0$ while $|H| > 1$.

---

### 🔹 Part B: Open Questions (Choose 2 out of 3, 40 Points Each)

#### Question 3: Linear Regression Models
* **a. Unconstrained MSE Linear Regression:**
  Objective $\min_{\mathbf{w}} \frac{1}{n} \|X\mathbf{w} - \mathbf{y}\|^2$. Unique solution exists if $X^T X$ is invertible ($X$ full column rank). Closed form: $\mathbf{w}^* = (X^T X)^{-1} X^T \mathbf{y}$.
* **b. Optimal $w_1$ given fixed $w_2$ in $\hat{y}_i = w_1^2 x_{i,1} + w_2^2 x_{i,2}$:**
  Let $\tilde{y}_i = y_i - w_2^2 x_{i,2}$. Let $A = w_1^2$. Objective $\min_{A \ge 0} \sum (A x_{i,1} - \tilde{y}_i)^2$.
  Analytical solution: $A^* = \frac{\sum \tilde{y}_i x_{i,1}}{\sum x_{i,1}^2}$.  
  If $A^* \ge 0 \implies w_1 = \pm \sqrt{A^*}$. If $A^* < 0 \implies w_1 = 0$.
* **c. Model A ($y = w^2 x$) vs. Model B ($y = w x$):**
  Model A is preferred when true data requires a positive slope ($w^2 \ge 0$) as non-negative slope regularization. Model B is preferred when true slope is negative ($y \approx -2x$), which Model A cannot represent.
* **d. Parameter Multiplicity in Model C ($y = w_1^2 x + w_2 x$):**
  Model B has 1 unique optimal parameter $w^*$. Model C has **infinitely many optimal parameter pairs** $(w_1, w_2)$ satisfying $w_1^2 + w_2 = w^*$.
* **e. Generalization Error Comparison:**
  Model B has 1 parameter (lower variance/capacity). Model C has 2 parameters with parameter redundancy. Model B is expected to achieve equal or better generalization error due to lower variance.

#### Question 4: Binary Classification & Soft-SVM
* **a. Soft-SVM Formulation:**
  $$\min_{\mathbf{w}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i \quad \text{s.t. } y_i (\mathbf{w}^T \mathbf{x}_i) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
* **b. Orthonormal Points Separability Proof:**
  Let $\mathbf{w} = \sum_{j: y_j=+1} \mathbf{x}_j - \sum_{j: y_j=-1} \mathbf{x}_j$. For any sample $\mathbf{x}_k$:
  $$\mathbf{w}^T \mathbf{x}_k = y_k (\mathbf{x}_k^T \mathbf{x}_k) = y_k (1) = y_k$$
  Hence $y_k (\mathbf{w}^T \mathbf{x}_k) = y_k^2 = 1 \ge 1$, proving perfect linear separability with margin 1!
* **c. Soft-SVM Zero Error Proof for $C = \|\mathbf{w}^*\|^2$:**
  Candidate $(\mathbf{w}^*, \mathbf{0})$ gives Soft-SVM objective $J(\mathbf{w}^*, \mathbf{0}) = \frac{1}{2}\|\mathbf{w}^*\|^2$. If optimal Soft-SVM had error $\sum \xi_i \ge 1$, objective would be $\ge \|\mathbf{w}^*\|^2 > \frac{1}{2}\|\mathbf{w}^*\|^2$, contradiction. Thus $\boldsymbol{\xi} = \mathbf{0}$.

#### Question 5: Non-linear Perceptrons & Kernel Trick
* **a. Kernel Validity & Computation Savings:**
  $K(\mathbf{x}, \mathbf{z}) = (1 + \mathbf{x}^T \mathbf{z})^2 = \phi(\mathbf{x})^T \phi(\mathbf{z})$ for $\phi(\mathbf{x}) = [1, \sqrt{2}x_1, \sqrt{2}x_2, x_1^2, \sqrt{2}x_1 x_2, x_2^2]^T \in \mathbb{R}^6$.
  Evaluating $K$ in 2D takes **4 FLOPs**, vs. **11 FLOPs** in 6D, saving 7 FLOPs per evaluation.
* **b. Perceptron Algorithm & Hinge Loss SGD:**
  Perceptron update $w \leftarrow w + y_i x_i$ on misclassification $y_i (w^T x_i) \le 0$ is identical to SGD step on Hinge loss $\mathcal{L}(w) = \max(0, -y_i w^T x_i)$ with learning rate $\eta=1$.
* **c. Feature Space Perceptron & Dual Representation:**
  Updating $w \leftarrow w + y_i \phi(x_i)$ implies $w = \sum_{i=1}^n \alpha_i y_i \phi(x_i)$ where $\alpha_i$ counts misclassifications.
* **d. Kernel Perceptron Pseudocode:**
  Initialize $\alpha_1, \dots, \alpha_n = 0$. For sample $i$, if $y_i \sum_{j=1}^n \alpha_j y_j K(x_j, x_i) \le 0 \implies \alpha_i \leftarrow \alpha_i + 1$.
