# 💡 2023 Moed B — Official Solutions | פתרונות רשמיים — מועד ב׳ 2023

---

## 🔹 Part A: True/False & Yes/No Questions / חלק א'

### Question 1: Overfitting Reduction
* **a. No**. Adding more layers increases model capacity, compounding overfitting when test error is already much larger than training error.
* **b. Yes**. Regularizing weights ($L_2$ decay) restricts model capacity and shrinks weights, reducing overfitting and improving test error.
* **c. No**. Retraining on test data causes data leakage and destroys valid performance evaluation.
* **d. No**. Using a larger test set increases the statistical precision of test evaluation, but does not alter the empirical test error value of the trained model itself.

### Question 2: Core Principles
* **a. Correct (True)**. Convolutional layers reuse localized filter kernels across spatial locations, resulting in far fewer parameters than dense fully-connected layers.
* **b. Correct (True)**. Residual skip connections ($F(x) + x$) prevent vanishing gradients, enabling optimization of very deep neural networks.
* **c. Wrong (False)**. ReLU is $\max(0, x)$ (a single comparison FLOP), whereas Sigmoid requires exponentiation $\frac{1}{1 + e^{-x}}$, which is computationally expensive.
* **d. Correct (True)**. Naive Bayes assumes features are conditionally independent given the class label $P(x_1,\dots,x_d \mid y) = \prod_{j=1}^d P(x_j \mid y)$.

---

## 🔹 Part B: Open Questions / חלק ב'

### Question 3: Linear Classifiers & MAP Logistic Regression
* **a. Logistic Regression Formulation:**
  Model prediction: $P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}}}$.  
  NLL Loss Function:
  $$\mathcal{L}(\mathbf{w}) = -\sum_{i=1}^n \left[ y_i \ln \sigma(\mathbf{w}^T \mathbf{x}_i) + (1 - y_i) \ln (1 - \sigma(\mathbf{w}^T \mathbf{x}_i)) \right]$$
  * **Rationale:** Derived from Maximum Likelihood Estimation (MLE) under Bernoulli label distribution. Minimizing NLL maximizes the log-likelihood of observed dataset labels.

* **b. SGD Derivation & Step Size:**
  $$\nabla_{\mathbf{w}} \mathcal{L}_i = -(y_i - \sigma(\mathbf{w}^T \mathbf{x}_i)) \mathbf{x}_i$$
  SGD Update Rule:
  $$\mathbf{w} \leftarrow \mathbf{w} + \eta \left( y_i - \sigma(\mathbf{w}^T \mathbf{x}_i) \right) \mathbf{x}_i$$
  * **Step Size $\eta$:** Selected via cross-validation grid search or learning rate schedules (e.g., step decay $\eta_t = \frac{\eta_0}{\sqrt{t}}$ or cosine annealing) to ensure smooth convergence without divergence.

* **c. MAP Formulation with Gaussian Weight Prior:**
  Prior distribution: $P_0(w_i) = \mathcal{N}(0, 1) = \frac{1}{\sqrt{2\pi}} e^{-w_i^2 / 2} \implies \ln P_0(\mathbf{w}) = -\frac{1}{2} \|\mathbf{w}\|^2 + \text{const}$.  
  MAP Objective:
  $$\max_{\mathbf{w}} \ln P(\mathbf{w} \mid D) = \max_{\mathbf{w}} \left[ \sum_{i=1}^n \ln P(y_i \mid \mathbf{x}_i; \mathbf{w}) + \ln P_0(\mathbf{w}) \right]$$
  $$\min_{\mathbf{w}} \mathcal{L}_{\text{MAP}}(\mathbf{w}) = \mathcal{L}_{\text{NLL}}(\mathbf{w}) + \frac{1}{2} \|\mathbf{w}\|^2$$
  New SGD Update Rule:
  $$\mathbf{w} \leftarrow \mathbf{w}(1 - \eta) + \eta \left( y_i - \sigma(\mathbf{w}^T \mathbf{x}_i) \right) \mathbf{x}_i$$
  * **Effect of Prior:** The Gaussian prior acts as **$L_2$ weight decay regularization**. It shrinks weights toward 0 at every iteration step, suppressing large parameter values and curbing overfitting.

---

### Question 4: Large Margin Classifiers & 1-NN Network Equivalence
* **a. Prediction via Kernel $K$ vs. Feature Space $\Phi(\mathbf{x})$:**
  * **Kernel Prediction:** $\hat{y} = \text{sign}\left( \sum_{i=1}^n \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b \right)$ — Requires $O(n \cdot d_{\text{input}})$ operations over support vectors.
  * **Explicit Feature Space Prediction:** $\hat{y} = \text{sign}\left( \mathbf{w}^T \Phi(\mathbf{x}) + b \right)$ where $\mathbf{w} = \sum \alpha_i y_i \Phi(\mathbf{x}_i)$ — Requires $O(D_{\text{feature}})$ operations.
  * **Difference:** Evaluating $\Phi(\mathbf{x})$ directly requires mapping to high or infinite dimensions (e.g. RBF kernel), which is computationally intractable. Kernel functions compute inner products directly in low-dimensional input space ($O(d_{\text{input}})$).

* **b. 2-Layer Network Equivalence to Kernel SVM:**
  Layer 1 has $n$ neurons computing $h_i(\mathbf{x}) = y_i K(\mathbf{x}, \mathbf{x}_i)$. Layer 2 applies weights $\alpha_1, \dots, \alpha_n$ and sign activation:
  $$f(\mathbf{x}) = \text{sign}\left( \sum_{i=1}^n \alpha_i h_i(\mathbf{x}) \right) = \text{sign}\left( \sum_{i=1}^n \alpha_i y_i K(\mathbf{x}, \mathbf{x}_i) \right)$$
  This is mathematically identical to the decision function of a Dual Kernel SVM with zero bias $b=0$.

* **c. 1-NN Network Equivalence:**
  Choose an RBF (Gaussian) kernel $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$ and set parameter $\gamma \to \infty$.  
  Set network weights $\alpha_i = 1$ for all $i=1,\dots,n$.  
  As $\gamma \to \infty$:
  $$K(\mathbf{x}, \mathbf{x}_{i^*}) \to 1 \quad \text{for nearest neighbor } \mathbf{x}_{i^*} = \arg\min_{\mathbf{x}_i} \|\mathbf{x} - \mathbf{x}_i\|$$
  $$K(\mathbf{x}, \mathbf{x}_i) \to 0 \quad \text{for all } \mathbf{x}_i \neq \mathbf{x}_{i^*}$$
  $$f(\mathbf{x}) = \text{sign}\left( 1 \cdot y_{i^*} \cdot 1 + \sum_{i \neq i^*} 0 \right) = \text{sign}(y_{i^*}) = y_{i^*}$$
  This reproduces the exact 1-Nearest-Neighbor prediction rule!

---

### Question 5: Deep Networks & Multi-Output Architecture
* **a. 1-Hidden-Layer MLP Forward Pass:**
  $$\mathbf{h} = g_1(W_1^T \mathbf{x}) \in \mathbb{R}^k \quad \text{where } W_1 \in \mathbb{R}^{d \times k}$$
  $$\hat{y} = \mathbf{w}_2^T \mathbf{h} \in \mathbb{R} \quad \text{where } \mathbf{w}_2 \in \mathbb{R}^k$$
  $$\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$$

* **b. Chain Rule Derivatives & SGD Weight Updates:**
  
  #### 1. Output Layer Weight Gradient ($\frac{\partial \mathcal{L}}{\partial \mathbf{w}_2}$):
  By chain rule for component $w_{2, j}$ ($j=1,\dots,k$):
  $$\frac{\partial \mathcal{L}}{\partial w_{2, j}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_{2, j}}$$
  * Output error signal: $\delta_{\text{out}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} = \frac{\partial}{\partial \hat{y}} \left[ \frac{1}{2}(\hat{y} - y)^2 \right] = (\hat{y} - y) \in \mathbb{R}$.
  * Prediction derivative: $\frac{\partial \hat{y}}{\partial w_{2, j}} = \frac{\partial}{\partial w_{2, j}} \left( \sum_{i=1}^k w_{2, i} h_i \right) = h_j$.
  * Vector gradient:
    $$\frac{\partial \mathcal{L}}{\partial \mathbf{w}_2} = (\hat{y} - y) \mathbf{h} = \delta_{\text{out}} \mathbf{h} \in \mathbb{R}^k$$

  #### 2. Hidden Layer Weight Matrix Gradient ($\frac{\partial \mathcal{L}}{\partial W_1}$):
  For weight element $W_{1, m, j}$ connecting input feature $x_m$ to hidden neuron $j$:
  $$\frac{\partial \mathcal{L}}{\partial W_{1, m, j}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h_j} \cdot \frac{\partial h_j}{\partial z_j^{(1)}} \cdot \frac{\partial z_j^{(1)}}{\partial W_{1, m, j}}$$
  * Factor 1 (Output Error): $\frac{\partial \mathcal{L}}{\partial \hat{y}} = (\hat{y} - y)$.
  * Factor 2 (Output w.r.t. Hidden Activation): $\frac{\partial \hat{y}}{\partial h_j} = w_{2, j}$.
  * Factor 3 (Activation Derivative): $\frac{\partial h_j}{\partial z_j^{(1)}} = g_1'(z_j^{(1)})$.
  * Factor 4 (Pre-activation w.r.t. Weight): $\frac{\partial z_j^{(1)}}{\partial W_{1, m, j}} = x_m$.

  Combining Factors 1, 2, and 3 yields the backpropagated error signal for hidden neuron $j$:
  $$\delta_j^{(1)} \equiv \frac{\partial \mathcal{L}}{\partial z_j^{(1)}} = (\hat{y} - y) w_{2, j} g_1'(z_j^{(1)}) \implies \boldsymbol{\delta}^{(1)} = (\hat{y} - y) \mathbf{w}_2 \odot g_1'(\mathbf{z}^{(1)}) \in \mathbb{R}^k$$

  Multiplying by Factor 4 ($x_m$) gives matrix outer product representation:
  $$\frac{\partial \mathcal{L}}{\partial W_1} = \mathbf{x} (\boldsymbol{\delta}^{(1)})^T = \mathbf{x} \left[ (\hat{y} - y) \mathbf{w}_2 \odot g_1'(W_1^T \mathbf{x}) \right]^T \in \mathbb{R}^{d \times k}$$

  #### 3. SGD Weight Updates:
  $$\mathbf{w}_2^{(t+1)} = \mathbf{w}_2^{(t)} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{w}_2} = \mathbf{w}_2^{(t)} - \eta (\hat{y} - y) \mathbf{h}$$
  $$W_1^{(t+1)} = W_1^{(t)} - \eta \frac{\partial \mathcal{L}}{\partial W_1} = W_1^{(t)} - \eta \mathbf{x} \left[ (\hat{y} - y) \mathbf{w}_2^{(t)} \odot g_1'(W_1^{(t)T} \mathbf{x}) \right]^T$$

* **c. Multi-Task MLP ($y_1 \in \mathbb{R}, y_2 \in \{+1, -1\}$):**
  * **Architecture:** Shared hidden layer $\mathbf{h} = g_1(W_1^T \mathbf{x}) \in \mathbb{R}^k$. Output 1 (regression): $\hat{y}_1 = \mathbf{w}_{\text{out1}}^T \mathbf{h}$. Output 2 (classification): $\hat{y}_2 = \sigma(\mathbf{w}_{\text{out2}}^T \mathbf{h})$.
  * **Joint Loss Function:**
    $$\mathcal{L}_{\text{total}} = \frac{1}{2}(\hat{y}_1 - y_1)^2 - \left[ \frac{1+y_2}{2}\ln \hat{y}_2 + \frac{1-y_2}{2}\ln(1-\hat{y}_2) \right]$$
  * **Weight Update:** Backpropagate combined error signals $\boldsymbol{\delta}_{\text{hidden}} = \delta_1 \mathbf{w}_{\text{out1}} \odot g_1' + \delta_2 \mathbf{w}_{\text{out2}} \odot g_1'$ to update shared representation $W_1$.

---

### Question 6: Unsupervised Learning & PCA
* **a. PCA Dimensionality Reduction Pseudocode:**
  ```python
  def pca_reduce(X, r):
      # X is shape (n, d), target dimension r < d
      mean = np.mean(X, axis=0)
      X_centered = X - mean
      cov = np.cov(X_centered, rowvar=False)  # Shape (d, d)
      eigenvalues, eigenvectors = np.linalg.eigh(cov)
      
      # Select top r eigenvectors
      idx = np.argsort(eigenvalues)[::-1][:r]
      V_r = eigenvectors[:, idx]  # Shape (d, r)
      
      # Project centered data to r dimensions
      Z = np.dot(X_centered, V_r)  # Shape (n, r)
      return Z, V_r, mean
  ```

* **b. Reconstruction Error Minimization Proof:**
  Loss: $J(\mathbf{z}) = \|V\mathbf{z} - \mathbf{x}\|^2 = (V\mathbf{z} - \mathbf{x})^T (V\mathbf{z} - \mathbf{x}) = \mathbf{z}^T V^T V \mathbf{z} - 2\mathbf{z}^T V^T \mathbf{x} + \mathbf{x}^T \mathbf{x}$.  
  Using $V^T V = I_r$:
  $$J(\mathbf{z}) = \|\mathbf{z}\|^2 - 2\mathbf{z}^T V^T \mathbf{x} + \|\mathbf{x}\|^2$$
  Differentiating w.r.t $\mathbf{z}$ and setting to 0:
  $$\nabla_{\mathbf{z}} J = 2\mathbf{z} - 2V^T \mathbf{x} = \mathbf{0} \implies \mathbf{z}^* = V^T \mathbf{x}$$
  Reconstructed vector: $\hat{\mathbf{x}} = V \mathbf{z}^* = V V^T \mathbf{x}$.

* **c. PCA Pre-processing for Linear Classification (Trade-offs & $d=2, r=1$):**
  * **When it helps:** Reduces overfitting by eliminating noise and high-frequency irrelevant features when class separation aligns with high-variance directions.
  * **When it hurts:** PCA is **unsupervised** and maximizes variance without considering class labels $y$. If class separation lies along a low-variance direction, PCA projection collapses classes together.
  * **Geometric Counter-example ($d=2, r=1$):** Consider two class clusters at $(0, 1)$ with $y=+1$ and $(0, -1)$ with $y=-1$, each having a large horizontal spread along $x_1 \in [-10, 10]$. The primary direction of variance $v_1$ is horizontal ($x_1$ axis). Projecting to $r=1$ along $v_1$ maps both classes onto the line $y=0$, mixing $+1$ and $-1$ points completely and destroying linear separability.
