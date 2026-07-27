# 💡 2023 Moed A — Official Solutions | פתרונות רשמיים — מועד א׳ 2023

---

## 🔹 Part A: True/False & Yes/No Questions / חלק א'

### Question 1: Generalization Gap
* **a. Yes**. Reducing model parameters reduces capacity and variance, shrinking the gap between training and test error.
* **b. No**. Increasing expressivity increases capacity, worsening overfitting when test error is already much larger than training error.
* **c. No**. Training on test data causes severe data leakage and invalidates performance evaluation.
* **d. No**. Training on combined train + test data eliminates the test set, violating machine learning validation protocol.

### Question 2: Core Principles
* **a. Correct (True)**. Gradient descent with step size $\eta \le \frac{1}{L}$ is guaranteed to converge linearly to the unique global minimum of a strongly convex function.
* **b. Correct (True)**. $k$-NN is a non-parametric classifier. Given sufficient data samples in $\{0, 1\}^d$, local majority voting correctly classifies discrete hypercube vertices for parity.
* **c. Correct (True)**. Repeated backpropagation matrix multiplications $\prod_{l=1}^L W^{(l)}$ lead to exponential gradient decay ($\lambda^L \to 0$) or explosion ($\lambda^L \to \infty$).
* **d. Correct (True)**. Recurrent networks share weight matrices across time steps, enabling processing of variable sequence lengths $T$.

---

## 🔹 Part B: Open Questions / חלק ב'

### Question 1: Linear Classifiers
* **a. Perceptron Pseudocode & Convergence:**
  ```python
  def train_perceptron(X, y, max_epochs=100):
      w = np.zeros(X.shape[1])
      for epoch in range(max_epochs):
          errors = 0
          for i in range(len(y)):
              if y[i] * np.dot(w, X[i]) <= 0:
                  w += y[i] * X[i]
                  errors += 1
          if errors == 0:
              break
      return w
  ```
  * **Convergence Requirement:** Data must be **strictly linearly separable** by a hyperplane passing through the origin with margin $\gamma > 0$. The Novikoff theorem guarantees convergence in at most $(R/\gamma)^2$ updates.

* **b. Hinge Loss SGD & Differences:**
  Hinge Loss: $\mathcal{L}(\mathbf{w}) = \max(0, 1 - y_i \mathbf{w}^T \mathbf{x}_i)$.  
  Subgradient: $\nabla_{\mathbf{w}} \mathcal{L}_i = -y_i \mathbf{x}_i$ if $y_i \mathbf{w}^T \mathbf{x}_i < 1$, else $\mathbf{0}$.  
  Update Rule: $\mathbf{w} \leftarrow \mathbf{w} + \eta y_i \mathbf{x}_i$ (or with $L_2$ weight decay $\mathbf{w} \leftarrow (1 - \eta \lambda)\mathbf{w} + \eta y_i \mathbf{x}_i$).  
  * **Differences from Perceptron:**
    1. Hinge SGD enforces a **margin of 1** ($y_i \mathbf{w}^T \mathbf{x}_i \ge 1$), whereas Perceptron only checks classification sign ($y_i \mathbf{w}^T \mathbf{x}_i > 0$).
    2. Hinge SGD scales updates by learning rate $\eta$ and supports regularizers.

* **c. Hinge SGD with Momentum:**
  ```python
  def train_hinge_sgd_momentum(X, y, lr=0.01, beta=0.9, reg=1e-4, epochs=100):
      w = np.zeros(X.shape[1])
      v = np.zeros(X.shape[1])  # Velocity vector
      for epoch in range(epochs):
          for i in range(len(y)):
              margin = y[i] * np.dot(w, X[i])
              grad = reg * w  # L2 regularization gradient
              if margin < 1:
                  grad -= y[i] * X[i]
              v = beta * v + lr * grad
              w = w - v
      return w
  ```

---

### Question 2: Large Margin & Sphere Classifier
* **a. Soft-SVM Formulation:**
  $$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i \quad \text{s.t. } y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0 \quad \forall i=1,\dots,n$$
  * $\mathbf{w} \in \mathbb{R}^d, b \in \mathbb{R}$: Hyperplane parameters $\mathbf{w}^T \mathbf{x} + b = 0$.
  * $\xi_i \ge 0$: Slack variable measuring margin violation of point $i$.
  * $C > 0$: Tradeoff hyperparameter balancing margin width against error penalty.

* **b. Margin Equivalence Proof:**
  Geometric margin is $\rho = \frac{2}{\|\mathbf{w}\|}$ under functional margin constraint $y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1$.  
  Maximizing margin $\max_{\mathbf{w}} \frac{2}{\|\mathbf{w}\|}$ is equivalent to $\min_{\mathbf{w}} \frac{\|\mathbf{w}\|}{2}$, which shares the exact same minimizer as $\min_{\mathbf{w}} \frac{1}{2} \|\mathbf{w}\|^2$.

* **c. Sphere-Classifier Optimization Problem:**
  We wish to find center $\mathbf{a} \in \mathbb{R}^d$ and radius squared $R = r^2 > 0$ such that $+1$ points fall inside $\|\mathbf{x}_i - \mathbf{a}\|^2 \le R$ and $-1$ points fall outside.  
  Formulation with slack variables $\xi_i \ge 0$:
  $$\min_{\mathbf{a}, R, \boldsymbol{\xi}} R + C \sum_{i=1}^n \xi_i \quad \text{s.t. } y_i \left( R - \|\mathbf{x}_i - \mathbf{a}\|^2 \right) \ge 1 - \xi_i, \quad \xi_i \ge 0 \quad \forall i=1,\dots,n, \quad R > 0$$

---

### Question 3: Non-Linear Classifiers & ConvNets
* **a. Conv2D Architecture Details:**
  RGB input tensor of shape $3 \times H \times W$. The layer consists of $C_{\text{out}}$ 3D filter kernels, each of dimension $3 \times K_h \times K_w$. Each filter slides spatially across the input with stride $S$ and padding $P$, computing local dot products to generate spatial feature maps of size $C_{\text{out}} \times H_{\text{out}} \times W_{\text{out}}$. Non-linearity is introduced by applying element-wise $\text{ReLU}(z) = \max(0, z)$.

* **b. KL-Divergence & Cross-Entropy Equivalence:**
  $$D_{KL}(t \| \hat{y}) = \sum_{k=1}^K t_k \ln \frac{t_k}{\hat{y}_k} = \sum_{k=1}^K t_k \ln t_k - \sum_{k=1}^K t_k \ln \hat{y}_k$$
  For a one-hot ground-truth target $t_{y^*} = 1$ and $t_k = 0$ ($k \neq y^*$):
  $$t_{y^*} \ln t_{y^*} = 1 \ln 1 = 0 \implies \sum t_k \ln t_k = 0$$
  $$D_{KL}(t \| \hat{y}) = -\ln \hat{y}_{y^*}$$
  Minimizing KL-divergence is mathematically identical to minimizing Cross-Entropy $-\ln \hat{y}_{y^*}$, which directly maximizes the target class likelihood $\hat{y}_{y^*}$.

* **c. Super-Resolution Architecture ($100 \times 100 \to 200 \times 200$):**
  * **(i) Layers & Filter Dimensions:**
    1. Conv2D feature extraction: $3 \to 64$ channels, kernel $3 \times 3$, stride 1, padding 1 + ReLU.
    2. Residual blocks: 4 Conv2D layers ($64 \to 64$), kernel $3\times3$ + BatchNorm + ReLU.
    3. Transposed Conv2D (Deconvolution): $64 \to 64$ channels, kernel $4 \times 4$, stride 2, padding 1 (doubles spatial dimensions from $100 \times 100$ to $200 \times 200$).
    4. Output Conv2D: $64 \to 3$ channels, kernel $3 \times 3$, stride 1, padding 1.
  * **(ii) Training Protocol:**
    * **Loss:** Combination of Pixel MSE Loss $\|Y - \hat{Y}\|_2^2$ and Perceptual Loss (feature distance in pre-trained VGG-19 network).
    * **Data Collection:** Collect high-resolution $200 \times 200$ images ($Y$), downsample them by factor of 2 using bicubic interpolation to generate input $100 \times 100$ images ($X$), providing unlimited paired training data.

---

### Question 4: Unsupervised Learning & $K$-Means
* **a. $K$-Means Pseudocode:**
  ```python
  def kmeans(X, k, max_iters=100):
      # Randomly initialize centroids
      centroids = X[np.random.choice(len(X), k, replace=False)]
      for _ in range(max_iters):
          # Step 1: Assign samples to nearest centroid
          distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
          clusters = np.argmin(distances, axis=1)
          
          # Step 2: Update centroids to mean of assigned points
          new_centroids = np.array([X[clusters == j].mean(axis=0) for j in range(k)])
          if np.allclose(centroids, new_centroids):
              break
          centroids = new_centroids
      return centroids, clusters
  ```

* **b. $L_2$ Distortion Reduction Convergence Proof:**
  Distortion objective: $J(Z, C) = \sum_{i=1}^n \|\mathbf{x}_i - c_{z_i}\|^2$.
  1. **Assignment Step (Holding $C$ fixed):** Choosing $z_i = \arg\min_j \|\mathbf{x}_i - c_j\|^2$ explicitly minimizes $J$ over $Z$, ensuring $J(Z^{t+1}, C^t) \le J(Z^t, C^t)$.
  2. **Centroid Step (Holding $Z$ fixed):** Differentiating $\sum_{i \in S_j} \|\mathbf{x}_i - c_j\|^2$ w.r.t. $c_j$ and setting to zero:
     $$-2 \sum_{i \in S_j} (\mathbf{x}_i - c_j) = 0 \implies c_j^* = \frac{1}{|S_j|} \sum_{i \in S_j} \mathbf{x}_i$$
     This guarantees $J(Z^{t+1}, C^{t+1}) \le J(Z^{t+1}, C^t)$.
  Since $J \ge 0$ is non-negative and strictly non-increasing, and there are a finite number of dataset cluster partitions ($k^n$), $K$-Means is guaranteed to converge in finite steps.

* **c. $L_1$ Distance ($K$-Medians):**
  Minimizing $L_1$ distortion $J_{L1} = \sum_{i \in S_j} \sum_{m=1}^d |x_{i,m} - c_{j,m}|$ replaces the arithmetic mean with the **component-wise median**:
  $$c_{j,m}^* = \text{median}\left( \{ x_{i,m} \mid i \in S_j \} \right)$$
  ```python
  # Centroid update for K-Medians (L1 distance)
  new_centroids = np.array([np.median(X[clusters == j], axis=0) for j in range(k)])
  ```
