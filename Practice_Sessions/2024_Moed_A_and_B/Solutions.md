# 💡 2024 Moed A & B — Official Solutions | פתרונות רשמיים — מועד א׳ וב׳ 2024

---

## Question 1: Linear Autoencoder vs. PCA Equivalence Proof
## שאלה 1: אוטואנקודר לינארי ושקילות מרחבית ל-PCA

### 🇮🇱 עברית & 🇬🇧 English Analytical Proof

1. **Low-Rank Matrix Approximation (Eckart-Young-Mirsky Theorem):**  
   The optimal rank-$r$ reconstruction matrix $\hat{X}$ minimizing Frobenius loss $\|X - \hat{X}\|_F^2$ for a centered matrix $X \in \mathbb{R}^{n \times d}$ is given by the truncated Singular Value Decomposition (SVD):
   $$X_r = U_r \Sigma_r V_r^T$$
   where $V_r \in \mathbb{R}^{d \times r}$ contains the top $r$ right singular vectors (the principal directions of PCA).

2. **Subspace Equivalence / שקילות המרחב:**  
   The linear autoencoder output is $\hat{X} = X W_e^T W_d^T$. At global minimum loss:
   $$W_d W_e = V_r V_r^T$$
   While PCA requires orthogonal unit vectors $V_r^T V_r = I_r$, the autoencoder matrix $W_d \in \mathbb{R}^{d \times r}$ spans the **exact same $r$-dimensional linear subspace** in $\mathbb{R}^d$ as $V_r$, i.e., $\text{span}(W_d) = \text{span}(V_r)$.

---

## Question 2: PCA Projection Reconstruction Error Minimization
## שאלה 2: מזעור שגיאת שחזור בהטלת PCA

### 🇮🇱 עברית & 🇬🇧 English Derivation

1. **Expand Reconstruction Loss / הרחבת השגיאה الריבועית:**
   $$J(\mathbf{z}) = \|V\mathbf{z} - \mathbf{x}\|^2 = (V\mathbf{z} - \mathbf{x})^T (V\mathbf{z} - \mathbf{x})$$
   $$J(\mathbf{z}) = \mathbf{z}^T V^T V \mathbf{z} - 2 \mathbf{z}^T V^T \mathbf{x} + \mathbf{x}^T \mathbf{x}$$

2. **Apply Orthonormality $V^T V = I_r$ / שימוש בתכונת האורתונורמליות:**
   $$J(\mathbf{z}) = \mathbf{z}^T I_r \mathbf{z} - 2 \mathbf{z}^T V^T \mathbf{x} + \|\mathbf{x}\|^2 = \|\mathbf{z}\|^2 - 2 \mathbf{z}^T V^T \mathbf{x} + \|\mathbf{x}\|^2$$

3. **Gradient Condition $\nabla_{\mathbf{z}} J(\mathbf{z}) = 0$ / השוואת הנגזרת לאפס:**
   $$\nabla_{\mathbf{z}} J(\mathbf{z}) = 2\mathbf{z} - 2V^T \mathbf{x} = \mathbf{0}$$
   $$\mathbf{z}^* = V^T \mathbf{x}$$

Reconstructed point / הנקודה המשוחזרת: $\hat{\mathbf{x}} = V \mathbf{z}^* = V V^T \mathbf{x}$.

---

## Question 3: Naive Bayes vs. Table Lookup & Curse of Dimensionality
## שאלה 3: בייס נאיבי מול טבלת שכיחויות וקללת הממדיות

### 🇮🇱 עברית & 🇬🇧 English Analysis

1. **Part A (Full Lookup Table Size / גודל הטבלה השלמה):**
   * Total configurations: $10 \text{ (Opponent)} \times 3 \text{ (Surface)} \times 2 \text{ (Type)} \times 2 \text{ (Weather)} \times 2 \text{ (Time)} = \mathbf{240 \text{ cells}}$.
   * Estimating probabilities via sample frequencies requires exponentially large sample sizes $n \propto |\mathcal{X}|$ to ensure each bin has non-zero counts. Otherwise, empty bins yield undefined/zero probabilities (Curse of Dimensionality / קללת הממדיות).

2. **Part B (Naive Bayes Parameter Reduction / צמצום פרמטרים בבייס נאיבי):**
   * Naive Bayes assumes conditional independence of features given class label $y \in \{\text{Win}, \text{Loss}\}$:
     $$P(O, C, G, W, T \mid y) = P(O \mid y) P(C \mid y) P(G \mid y) P(W \mid y) P(T \mid y)$$
   * Instead of estimating 240 joint probability values, Naive Bayes only estimates separate 1D categorical distributions for each feature given $y$, reducing parameters to $(10 + 3 + 2 + 2 + 2) \times 2 = \mathbf{38 \text{ parameters}}$.

---

## Question 4: Max-Margin Ranking SVM Formulation
## שאלה 4: ניסוח אלגוריתם דירוג Ranking SVM עם שוליים מרביים

### 🇮🇱 עברית & 🇬🇧 English Formulation

Objective / בעיית האופטימיזציה:
$$\min_{\mathbf{w}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i$$
Subject to constraints / תחת האילוצים:
$$\mathbf{w}^T x_i^{(1)} - \mathbf{w}^T x_i^{(2)} \ge 1 - \xi_i, \quad \forall i=1,\dots,n$$
$$\xi_i \ge 0, \quad \forall i=1,\dots,n$$

*(Equivalent to binary SVM on difference vectors $\mathbf{d}_i = x_i^{(1)} - x_i^{(2)}$ with target labels $y_i = +1$).*
