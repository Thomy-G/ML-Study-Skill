---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Resource
date_added: 2026-07-26
---
# 2024 Machine Learning Past Exams (Moed A, B & C)

**Course:** [[English]] (Bar-Ilan University)  
**Instructors:** Prof. Joseph Keshet & Prof. Gal Chechik  
**Master Index:** [[Past Exams Master Index]]

---

## 📝 2024 Moed A & B — Core Topics & Exam Questions

### Question 1: Linear Autoencoder & PCA Equivalence Proof
**Problem Statement:**  
Given dataset $X \in \mathbb{R}^{n \times d}$. Consider a linear autoencoder with bottleneck dimension $r < d$, parameterized by encoder matrix $W_e \in \mathbb{R}^{r \times d}$ and decoder matrix $W_d \in \mathbb{R}^{d \times r}$:
$$\hat{X} = X W_e^T W_d^T$$
Prove that under Mean Squared Error (MSE) reconstruction loss $\|X - \hat{X}\|_F^2$, the learned $r$-dimensional subspace spanned by the columns of $W_d$ is mathematically equivalent to the subspace spanned by the top $r$ Principal Components of PCA.

#### Official Proof Steps:
1. **Low-Rank Matrix Approximation (Eckart-Young-Mirsky Theorem):**  
   The optimal rank-$r$ approximation of centered data matrix $X$ under Frobenius norm is given by the SVD truncation $X_r = U_r \Sigma_r V_r^T$.
2. **Subspace Equivalence:**  
   While PCA enforces strict orthonormal ordering $V_r^T V_r = I_r$, the linear autoencoder finds matrices such that $W_d W_e = V_r V_r^T$. The weight vectors span the exact same global geometric subspace.

---

### Question 2: PCA Projection Reconstruction Error Minimization
**Problem Statement:**  
Let $V \in \mathbb{R}^{d \times r}$ be a matrix whose $r$ columns are orthonormal basis vectors of a linear subspace ($V^T V = I_r$). Prove that for any sample $\mathbf{x} \in \mathbb{R}^d$, the projection vector $\mathbf{z} \in \mathbb{R}^r$ minimizing the reconstruction error $\|V\mathbf{z} - \mathbf{x}\|^2$ is given by:
$$\mathbf{z}^* = V^T \mathbf{x}$$

#### Step-by-Step Analytical Proof:
1. **Expand the Squared Norm:**
   $$J(\mathbf{z}) = \|V\mathbf{z} - \mathbf{x}\|^2 = (V\mathbf{z} - \mathbf{x})^T (V\mathbf{z} - \mathbf{x}) = \mathbf{z}^T V^T V \mathbf{z} - 2 \mathbf{z}^T V^T \mathbf{x} + \mathbf{x}^T \mathbf{x}$$
2. **Apply Orthonormality $V^T V = I_r$:**
   $$J(\mathbf{z}) = \mathbf{z}^T \mathbf{z} - 2 \mathbf{z}^T V^T \mathbf{x} + \|\mathbf{x}\|^2$$
3. **Compute Gradient w.r.t. $\mathbf{z}$ and Set to Zero:**
   $$\nabla_{\mathbf{z}} J(\mathbf{z}) = 2\mathbf{z} - 2V^T \mathbf{x} = 0 \implies \mathbf{z}^* = V^T \mathbf{x}$$
4. **Reconstructed Point:**
   $$\hat{\mathbf{x}} = V \mathbf{z}^* = V V^T \mathbf{x}$$

---

### Question 3: Naive Bayes vs. Table Lookup & The Curse of Dimensionality
**Problem Statement:**  
Predict tennis win probability given features: Opponent Skill $O \in \{1..10\}$, Surface $C \in \{\text{Grass, Clay, Hard}\}$, Type $G \in \{\text{Doubles, Singles}\}$, Weather $W \in \{\text{Hot, Cold}\}$, Time $T \in \{\text{Day, Night}\}$.

1. **Part A (Full Lookup Table Size):**  
   Number of configurations: $10 \times 3 \times 2 \times 2 \times 2 = 240$ cells. Estimating probabilities via sample frequencies requires exponential samples $n \propto |\mathcal{X}|$ to prevent empty bins (Curse of Dimensionality).
2. **Part B (Naive Bayes Parameter Reduction):**  
   Naive Bayes assumes conditional independence given class label $y \in \{\text{Win}, \text{Loss}\}$:
   $$P(O, C, G, W, T \mid y) = P(O \mid y) P(C \mid y) P(G \mid y) P(W \mid y) P(T \mid y)$$
   Reduces parameter estimation from exponential $O(k^d)$ to linear $O(d \cdot k)$ parameters, avoiding sample sparsity.

---

### Question 4: Maximum-Margin Ranking SVM Formulation
**Problem Statement:**  
Given pairs $(x_i^{(1)}, x_i^{(2)})$ where $x_i^{(1)}$ must rank higher than $x_i^{(2)}$. Formulate max-margin ranking as a constrained optimization problem.

#### Formal Optimization Problem:
$$\min_{\mathbf{w}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i$$
$$\text{s.t. } \mathbf{w}^T x_i^{(1)} - \mathbf{w}^T x_i^{(2)} \ge 1 - \xi_i, \quad \xi_i \ge 0 \quad \forall i=1,\dots,n$$
*(Equivalent to standard binary SVM on difference vectors $\mathbf{d}_i = x_i^{(1)} - x_i^{(2)}$ with target label $y_i = +1$).*
