---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Resource
date_added: 2026-07-27
---
# 2023 Machine Learning Past Exams (Moed A & B)

**Course:** [[English]] (Bar-Ilan University)  
**Instructors:** Prof. Joseph Keshet & Prof. Gal Chechik  
**Master Index:** [[Past Exams Master Index]]

---

## 📝 2023 Moed A — Full Exam Paper & Official Solutions

### 🔹 Part A: True/False Questions (25 Points Total)

#### Question 1 (12.5 Points) — Overfitting & Generalization Gap
After training a deep network model, you observe that the test error is much larger than the training error. Mark which of the following actions is expected to reduce this generalization gap:
* **a.** Reduce the number of parameters in the model, and retrain: **Yes**. Reducing capacity lowers variance, shrinking the gap.
* **b.** Add more expressivity to the model, and retrain: **No**. Increasing expressivity increases model capacity, worsening overfitting.
* **c.** Retrain the model, but this time using the test data: **No**. This causes data leakage and invalidates test evaluation.
* **d.** Retrain the model using both the training and test data: **No**. This violates proper machine learning protocol.

#### Question 2 (12.5 Points) — Machine Learning Principles
* **a.** Gradient descent is guaranteed to converge when optimizing a strongly convex function: **Correct (True)**.
* **b.** A $k$-NN classifier can solve the parity problem given enough samples: **Correct (True)**. Non-parametric local neighbors correctly partition discrete hypercube vertices.
* **c.** Training very deep MLPs is hard because the gradients tend to vanish or explode: **Correct (True)**. Repeated matrix multiplication $\prod W^{(l)}$ causes exponential gradient scaling.
* **d.** RNN architectures can handle input sequences of different sizes: **Correct (True)**. Recurrent weight sharing across time steps supports arbitrary sequence lengths $T$.

---

### 🔹 Part B: Open Questions (Choose 3 out of 4, 25 Points Each)

#### Question 1: Linear Classifiers
* **a. Perceptron Pseudocode & Convergence Condition:**
  Initialize $\mathbf{w} = \mathbf{0}$. For epoch in range($E$): for $(\mathbf{x}_i, y_i)$: if $y_i(\mathbf{w}^T \mathbf{x}_i) \le 0 \implies \mathbf{w} \leftarrow \mathbf{w} + y_i \mathbf{x}_i$. Convergence is guaranteed if the dataset is **strictly linearly separable** with margin $\gamma > 0$ (bounded updates $\le (R/\gamma)^2$).
* **b. Hinge Loss SGD:**
  Objective $\min_{\mathbf{w}} \max(0, 1 - y_i \mathbf{w}^T \mathbf{x}_i)$. Gradient $\nabla_{\mathbf{w}} L = -y_i \mathbf{x}_i$ if $y_i \mathbf{w}^T \mathbf{x}_i < 1$, else $\mathbf{0}$. Update $\mathbf{w} \leftarrow \mathbf{w} + \eta y_i \mathbf{x}_i$. Unlike Perceptron, Hinge loss updates even on correctly classified points within the margin ($0 < y_i \mathbf{w}^T \mathbf{x}_i < 1$) and uses step size $\eta$.
* **c. Perceptron with Momentum:**
  Initialize $\mathbf{v} = \mathbf{0}, \mathbf{w} = \mathbf{0}$. On misclassification: $\mathbf{v} \leftarrow \beta \mathbf{v} + \eta (-y_i \mathbf{x}_i)$, $\mathbf{w} \leftarrow \mathbf{w} - \mathbf{v}$.

#### Question 2: Large Margin Classifiers & Sphere Classifier
* **a. Linear Soft-SVM Formulation:**
  $$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i \quad \text{s.t. } y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
* **b. Margin Equivalence Proof:**
  Geometric margin $\rho = \frac{2}{\|\mathbf{w}\|}$ under normalization $y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1$. Maximizing margin $\max_{\mathbf{w}} \frac{2}{\|\mathbf{w}\|}$ is mathematically identical to minimizing $\frac{1}{2}\|\mathbf{w}\|^2$.
* **c. Sphere-Classifier SVM Formulation:**
  To enclose $+1$ samples within ball $\|\mathbf{x} - \mathbf{a}\|^2 < R$ and $-1$ outside, with radius $R = r^2$ and center $\mathbf{a}$:
  $$\min_{\mathbf{a}, R, \boldsymbol{\xi}} R + C \sum_{i=1}^n \xi_i \quad \text{s.t. } y_i \left( R - \|\mathbf{x}_i - \mathbf{a}\|^2 \right) \ge 1 - \xi_i, \quad \xi_i \ge 0, \quad R > 0$$

#### Question 3: Non-Linear Classifiers & ConvNets
* **a. Conv2D Layer Architecture:**
  RGB input $3 \times H \times W$, $C_{\text{out}}$ filters of size $3 \times K_h \times K_w$, stride $S$, padding $P$. Spatial convolution followed by element-wise activation $\text{ReLU}(z) = \max(0, z)$.
* **b. KL-Divergence & Cross-Entropy for One-Hot Target:**
  $D_{KL}(t \| \hat{y}) = \sum_k t_k \ln \frac{t_k}{\hat{y}_k} = -\sum_k t_k \ln \hat{y}_k$ (since $t_k \ln t_k = 0$). For one-hot target $t_{y^*} = 1$, $D_{KL}(t \| \hat{y}) = -\ln \hat{y}_{y^*}$, which is identical to maximizing target class likelihood.
* **c. Super-Resolution ($100 \times 100 \to 200 \times 200$):**
  *(i)* Architecture: Conv2D feature extraction ($3 \to 64$), Transposed Conv2D ($4 \times 4$ kernel, stride 2, padding 1 for $2\times$ upsampling), final Conv2D ($64 \to 3$). *(ii)* Training: Pixel MSE loss $\|Y - \hat{Y}\|_2^2$ + VGG perceptual loss, trained on paired dataset created by bicubic downsampling $200 \times 200$ images.

#### Question 4: Unsupervised Learning & $K$-Means
* **a. $K$-Means Pseudocode:**
  Init $c_1,\dots,c_k$. Loop: $z_i = \arg\min_j \|\mathbf{x}_i - c_j\|^2$; update $c_j = \frac{1}{|S_j|} \sum_{i \in S_j} \mathbf{x}_i$. Repeat until convergence.
* **b. $K$-Means Convergence Proof ($L_2$ Distortion):**
  Distortion $J(Z, C) = \sum_{i=1}^n \|\mathbf{x}_i - c_{z_i}\|^2$. Assignment step minimizes $J$ w.r.t $Z$; centroid step sets $\nabla_{c_j} J = 0 \implies c_j = \frac{1}{|S_j|}\sum \mathbf{x}_i$, minimizing $J$ w.r.t $C$. Since $J \ge 0$ monotonically decreases over finite partitions ($k^n$), convergence is guaranteed.
* **c. $L_1$ Distance ($K$-Medians):**
  Minimizing $L_1$ distortion $\sum |\mathbf{x}_{i,m} - c_{j,m}|$ replaces arithmetic mean with **component-wise median**: $c_{j,m} = \text{median}(\{x_{i,m} \mid i \in S_j\})$.

---

## 📝 2023 Moed B — Full Exam Paper & Official Solutions

### 🔹 Part A: True/False Questions (25 Points Total)

#### Question 1 (12.5 Points) — Overfitting Reduction
* **a. Use a deeper model (add more layers), and retrain:** **No**.
* **b. Regularize the weights of the model during training:** **Yes**. $L_2$ regularization shrinks capacity.
* **c. Retrain the model but this time on the test data:** **No**.
* **d. Use a larger test set:** **No**. A larger test set improves evaluation precision, but does not alter model generalization.

#### Question 2 (12.5 Points) — Core ML Principles
* **a. A convolution layer typically has fewer parameters than a fully-connected layer:** **Correct (True)**. Local kernels & weight sharing reduce parameters.
* **b. Adding skip connections to a ConvNet can help with training very deep MLPs:** **Correct (True)**. Creates identity gradient highways.
* **c. ReLU is computationally less efficient than a Sigmoid:** **Wrong (False)**. ReLU is $\max(0, x)$ (1 comparison), whereas Sigmoid requires expensive $e^{-x}$ evaluation.
* **d. The Naive Bayes model assumes that features are independent:** **Correct (True)**. Assumes conditional independence given class.

---

### 🔹 Part B: Open Questions (Choose 3 out of 4, 25 Points Each)

#### Question 3: Linear Classifiers & MAP Logistic Regression
* **a. Logistic Regression Formulation:**
  $P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x})$. NLL Loss $\mathcal{L}(\mathbf{w}) = -\sum [y_i \ln \sigma + (1-y_i)\ln(1-\sigma)]$.
* **b. SGD Derivation & Step Size:**
  $\nabla \mathcal{L}_i = -(y_i - \sigma(\mathbf{w}^T \mathbf{x}_i))\mathbf{x}_i$. Update: $\mathbf{w} \leftarrow \mathbf{w} + \eta (y_i - \sigma(\mathbf{w}^T \mathbf{x}_i))\mathbf{x}_i$. Learning rate $\eta$ selected via validation grid search / decay schedules.
* **c. MAP with Gaussian Weight Prior:**
  Prior $P_0(w_i) = \mathcal{N}(0, 1) \implies \ln P_0(\mathbf{w}) = -\frac{1}{2}\|\mathbf{w}\|^2 + \text{const}$.
  MAP Objective: $\min_{\mathbf{w}} \mathcal{L}(\mathbf{w}) + \frac{1}{2}\|\mathbf{w}\|^2$ (equivalent to **$L_2$ weight decay**).
  New SGD Update: $\mathbf{w} \leftarrow \mathbf{w}(1 - \eta) + \eta (y_i - \sigma(\mathbf{w}^T \mathbf{x}_i))\mathbf{x}_i$. Prevents large weights, curbing overfitting.

#### Question 4: Large Margin Classifiers & 1-NN Network Equivalence
* **a. Dual SVM Prediction vs. Feature Space $\Phi(x)$:**
  Dual prediction $\hat{y} = \text{sign}(\sum \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b)$. Feature space prediction $\hat{y} = \text{sign}(\mathbf{w}^T \Phi(\mathbf{x}) + b)$. Kernel method avoids computing high/infinite-dimensional $\Phi(\mathbf{x})$ explicitly.
* **b. 2-Layer Network Equivalence to Kernel SVM:**
  Layer 1 outputs $h_i(\mathbf{x}) = y_i K(\mathbf{x}, \mathbf{x}_i)$. Layer 2 weights $\alpha_i$ with sign activation: $f(\mathbf{x}) = \text{sign}(\sum \alpha_i y_i K(\mathbf{x}, \mathbf{x}_i))$, identical to Dual Kernel SVM.
* **c. 1-NN Network Equivalence:**
  Choose RBF kernel $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$ with $\gamma \to \infty$. Set $\alpha_i = 1$. As $\gamma \to \infty$, $K(\mathbf{x}, \mathbf{x}_i) \to 1$ for nearest neighbor $i^*$ and $0$ otherwise, yielding $f(\mathbf{x}) = y_{i^*}$ (1-NN prediction).

#### Question 5: Deep Networks & Multi-Output Architecture
* **a. 1-Hidden-Layer MLP Forward Pass:**
  $\mathbf{h} = g_1(W_1^T \mathbf{x}) \in \mathbb{R}^k, \quad \hat{y} = \mathbf{w}_2^T \mathbf{h} \in \mathbb{R}, \quad l = \frac{1}{2}(\hat{y} - y)^2$.
* **b. Chain Rule Derivatives & SGD Weight Updates:**
  * **Output Layer Weight Gradient $\frac{\partial l}{\partial \mathbf{w}_2}$**:
    Chain rule over components $w_{2, j}$: $\frac{\partial l}{\partial w_{2, j}} = \frac{\partial l}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_{2, j}} = (\hat{y} - y) h_j \implies \frac{\partial l}{\partial \mathbf{w}_2} = (\hat{y} - y) \mathbf{h} = \delta_{\text{out}} \mathbf{h} \in \mathbb{R}^k$.
  * **Hidden Layer Weight Matrix Gradient $\frac{\partial l}{\partial W_1}$**:
    Chain rule over element $W_{1, m, j}$ connecting input feature $x_m$ to hidden neuron $j$:
    $$\frac{\partial l}{\partial W_{1, m, j}} = \frac{\partial l}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h_j} \cdot \frac{\partial h_j}{\partial z_j^{(1)}} \cdot \frac{\partial z_j^{(1)}}{\partial W_{1, m, j}} = (\hat{y} - y) \cdot w_{2, j} \cdot g_1'(z_j^{(1)}) \cdot x_m$$
    Backpropagated hidden error vector $\boldsymbol{\delta}^{(1)} = (\hat{y} - y) \mathbf{w}_2 \odot g_1'(W_1^T \mathbf{x}) \in \mathbb{R}^k$.  
    Outer product matrix representation:
    $$\frac{\partial l}{\partial W_1} = \mathbf{x} (\boldsymbol{\delta}^{(1)})^T = \mathbf{x} \left[ (\hat{y} - y) \mathbf{w}_2 \odot g_1'(W_1^T \mathbf{x}) \right]^T \in \mathbb{R}^{d \times k}$$
  * **SGD Updates**:
    $$\mathbf{w}_2^{(t+1)} = \mathbf{w}_2^{(t)} - \eta (\hat{y} - y) \mathbf{h}, \quad W_1^{(t+1)} = W_1^{(t)} - \eta \mathbf{x} \left[ (\hat{y} - y) \mathbf{w}_2^{(t)} \odot g_1'(W_1^{(t)T} \mathbf{x}) \right]^T$$
* **c. Multi-Task MLP ($y_1 \in \mathbb{R}, y_2 \in \{+1, -1\}$):**
  Shared hidden layer $\mathbf{h} = g_1(W_1^T \mathbf{x})$. Output 1 (regression): $\hat{y}_1 = \mathbf{w}_{out1}^T \mathbf{h}$. Output 2 (classification): $\hat{y}_2 = \sigma(\mathbf{w}_{out2}^T \mathbf{h})$. Loss $L_{total} = L_{MSE}(\hat{y}_1, y_1) + L_{BCE}(\hat{y}_2, y_2)$.

#### Question 6: Unsupervised Learning & PCA
* **a. PCA Dimensionality Reduction Pseudocode:**
  Center $\tilde{X} = X - \bar{X}$, covariance $\Sigma = \frac{1}{n} \tilde{X}^T \tilde{X}$, top $r$ eigenvectors $V_r \in \mathbb{R}^{d \times r}$, project $\mathbf{z} = V_r^T (\mathbf{x} - \bar{\mathbf{x}})$.
* **b. PCA Reconstruction Error Minimization Proof:**
  $J(\mathbf{z}) = \|V\mathbf{z} - \mathbf{x}\|^2 = \mathbf{z}^T V^T V \mathbf{z} - 2\mathbf{z}^T V^T \mathbf{x} + \|\mathbf{x}\|^2 = \|\mathbf{z}\|^2 - 2\mathbf{z}^T V^T \mathbf{x} + \|\mathbf{x}\|^2$.
  $\nabla_{\mathbf{z}} J = 2\mathbf{z} - 2V^T \mathbf{x} = \mathbf{0} \implies \mathbf{z}^* = V^T \mathbf{x}$. Reconstructed point $\hat{\mathbf{x}} = V V^T \mathbf{x}$.
* **c. PCA Pre-processing for Linear Classification (Trade-offs & $d=2, r=1$):**
  * *Helps*: Eliminates noisy low-variance features when classes separate along main principal components.
  * *Hurts*: PCA is unsupervised and ignores class labels. If class separation is along a low-variance direction, PCA projection collapses classes together.
  * *Illustration ($d=2, r=1$)*: Two clusters at $(0, 1)$ with $y=+1$ and $(0, -1)$ with $y=-1$, both spread horizontally along $x_1 \in [-10, 10]$. Top PCA component $v_1$ is horizontal axis $x_1$. Projecting to $r=1$ maps all points onto line $y=0$, destroying linear separability.
