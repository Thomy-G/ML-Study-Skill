---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Resource
date_added: 2026-07-26
---
# 2022 Machine Learning Past Exams (Moed A & B)

**Course:** [[English]] (Bar-Ilan University)  
**Instructors:** Prof. Joseph Keshet & Prof. Gal Chechik  
**Master Index:** [[Past Exams Master Index]]

---

## 📝 2022 Moed A — Exam Questions & Solutions

### Question 1: $k$-NN Distance Metric & Properties
**Problem Statement:**  
Tzvia collected data where each sample is represented by 2 features $\mathbf{x} \in \mathbb{R}^2$. She classifies new samples using a $k$-NN classifier with the custom distance function:
$$d(\mathbf{x}, \mathbf{y}) = |x_1 - y_1| + |x_2 - y_2|^2$$

1. **Part A:** Is $d(\mathbf{x}, \mathbf{y})$ a valid metric space distance function? Prove or disprove.
   * **Solution & Rubric:** Disprove. A valid metric must satisfy symmetry, non-negativity, identity of indiscernibles, and the **Triangle Inequality** ($d(\mathbf{x}, \mathbf{z}) \le d(\mathbf{x}, \mathbf{y}) + d(\mathbf{y}, \mathbf{z})$). Because of the unsymmetric squaring term $|x_2 - y_2|^2$, the triangle inequality fails (e.g. taking $x_2=0, y_2=1, z_2=2 \implies |0-2|^2 = 4 > |0-1|^2 + |1-2|^2 = 1+1=2$).

---

### Question 2: CNN Architecture, Shapes & Parameter Counting
**Problem Statement:**  
Yossi trains a ConvNet on CIFAR-10 ($3 \times 32 \times 32$ RGB images, 10 classes).  
Architecture layout:
* **Layer A:** `Conv2D(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)` $\to$ `ReLU` $\to$ `MaxPool2D(kernel_size=2, stride=2)`
* **Layer B:** `Conv2D(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1)` $\to$ `ReLU` $\to$ `MaxPool2D(kernel_size=2, stride=2)`
* **Layer C:** `Flatten` $\to$ `FullyConnected(out_features=10)`

#### Part A: Output Spatial Dimensions
Calculate input tensor dimensions for layers A, B, and C:
* **Layer A Output (before MaxPool):**  
  $$W_{\text{out}} = \left\lfloor \frac{32 - 3 + 2(1)}{1} \right\rfloor + 1 = 32 \implies (10 \times 32 \times 32)$$
  After $2 \times 2$ MaxPool: **$(10 \times 16 \times 16)$**.
* **Layer B Output (before MaxPool):**  
  $$W_{\text{out}} = \left\lfloor \frac{16 - 3 + 2(1)}{1} \right\rfloor + 1 = 16 \implies (10 \times 16 \times 16)$$
  After $2 \times 2$ MaxPool: **$(10 \times 8 \times 8)$**.
* **Layer C Input (Flattened):**  
  $$10 \times 8 \times 8 = \mathbf{640 \text{ features}}$$.

#### Part B: Parameter Counting (Including Biases)
* **Layer A Conv Weights + Biases:**  
  $$\text{Params}_A = (3 \times 3 \times 3 + 1) \times 10 = 28 \times 10 = \mathbf{280}$$
* **Layer B Conv Weights + Biases:**  
  $$\text{Params}_B = (3 \times 3 \times 10 + 1) \times 10 = 91 \times 10 = \mathbf{910}$$
* **Layer C FC Weights + Biases:**  
  $$\text{Params}_C = (640 + 1) \times 10 = \mathbf{6,410}$$
* **Total Network Parameters:** $280 + 910 + 6410 = \mathbf{7,600 \text{ parameters}}$.

---

### Question 3: Perceptron with Margin vs. Soft-SVM
**Problem Statement:**  
Gal has data points $\mathbf{x} \in \mathbb{R}^2$ (COVID-19 healthy vs. sick) that are **not linearly separable**.

1. **Part A:** Why does training a Margin Perceptron without bias fail to converge?
   * **Official Solution:** Perceptron assumes strict linear separability. If no hyperplane exists that separates the data, the Perceptron algorithm loops infinitely without terminating.
2. **Part B:** Would Soft-Margin SVM converge? Compare both methods.
   * **Official Solution:** Yes! Soft-SVM incorporates slack variables $\xi_i \ge 0$ with penalty $C \sum \xi_i$ in its objective function $\min_{\mathbf{w}} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum \xi_i$, allowing it to trade off margin width against classification errors on non-separable data.

---

## 📝 2022 Moed B — Computational Graphs & Backpropagation

### Question: 2-Layer MLP Backpropagation & Loss Computation
**Problem Statement:**  
Shoham trains a 3-class classification model using Negative Log-Likelihood (NLL) Loss:
$$\mathcal{L} = -\ln P(y = y_t \mid \mathbf{x}_t)$$
Network Architecture:
$$\mathbf{h} = \sigma(w_1 x + b_1)$$
$$\mathbf{z} = \mathbf{w}_2 \mathbf{h} + \mathbf{b}_2$$
$$\hat{\mathbf{p}} = \text{Softmax}(\mathbf{z})$$

Given parameters:
* $w_1 = 2, b_1 = -2$
* $\mathbf{w}_2 = [1, 0.5, -1]^T, \mathbf{b}_2 = [-1, 1, 0]^T$
* Input sample $x_i = 2$, true label $y_i = \text{Class C}$ (index 3).

#### Analytical Step-by-Step Loss Calculation:
1. **Hidden Layer Activation $\mathbf{h}$:**
   $$z_1 = w_1 x + b_1 = 2(2) - 2 = 2$$
   $$h = \sigma(2) = \frac{1}{1 + e^{-2}} \approx \mathbf{0.881}$$
2. **Logits Vector $\mathbf{z}$:**
   $$\mathbf{z} = h \cdot \mathbf{w}_2 + \mathbf{b}_2 = 0.881 \begin{bmatrix} 1 \\ 0.5 \\ -1 \end{bmatrix} + \begin{bmatrix} -1 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix} -0.119 \\ 1.441 \\ -0.881 \end{bmatrix}$$
3. **Softmax Probabilities $\hat{\mathbf{p}}$:**
   $$e^{\mathbf{z}} = \begin{bmatrix} e^{-0.119} \\ e^{1.441} \\ e^{-0.881} \end{bmatrix} \approx \begin{bmatrix} 0.888 \\ 4.225 \\ 0.414 \end{bmatrix}$$
   $$\sum e^{z_j} = 0.888 + 4.225 + 0.414 = 5.527$$
   $$\hat{\mathbf{p}} = \begin{bmatrix} 0.161 \\ 0.764 \\ 0.075 \end{bmatrix}$$
4. **Final NLL Loss for Target Class C ($p_3$):**
   $$\mathcal{L} = -\ln(p_3) = -\ln(0.075) \approx \mathbf{2.59}$$
