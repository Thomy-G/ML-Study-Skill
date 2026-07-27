# 💡 2023 Moed A — Official Solutions | פתרונות רשמיים — מועד א׳ 2023

---

## Question 1: MLE for 1D Gaussian Mean
## שאלה 1: אומד סבירות מרבית (MLE) לתוחלת התפלגות נורמלית

### 🇮🇱 עברית & 🇬🇧 English Full Analytical Proof

1. **Likelihood Function $L(\mu)$ / פונקציית הנראות:**
   $$L(\mu) = \prod_{i=1}^n f(x_i \mid \mu, \sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$$

2. **Log-Likelihood Function $\ell(\mu)$ / פונקציית הלוג-נראות:**
   $$\ell(\mu) = \ln L(\mu) = \sum_{i=1}^n \left[ -\frac{1}{2}\ln(2\pi\sigma^2) - \frac{(x_i - \mu)^2}{2\sigma^2} \right]$$
   $$\ell(\mu) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^n (x_i - \mu)^2$$

3. **Stationarity Condition $\frac{\partial \ell}{\partial \mu} = 0$ / התנאי לנקודת קיצון:**
   $$\frac{\partial \ell}{\partial \mu} = -\frac{1}{2\sigma^2} \sum_{i=1}^n 2(x_i - \mu)(-1) = \frac{1}{\sigma^2} \sum_{i=1}^n (x_i - \mu) = 0$$
   $$\sum_{i=1}^n x_i - n\mu = 0 \implies n\mu = \sum_{i=1}^n x_i$$
   $$\hat{\mu}_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^n x_i = \bar{x}$$

---

## Question 2: Linear Regression & Ridge Regularization Duality
## שאלה 2: רגרסיה לינארית ודואליות רגולריזציית Ridge

### 🇮🇱 עברית & 🇬🇧 English Full Analytical Proof

#### Part A: Unconstrained Least Squares Solution / סעיף א'
Expand MSE loss in matrix form / הרחבת פונקציית ההפסד במטריצות:
$$J(\mathbf{w}) = \frac{1}{n} (X\mathbf{w} - \mathbf{y})^T (X\mathbf{w} - \mathbf{y}) = \frac{1}{n} (\mathbf{w}^T X^T X \mathbf{w} - 2\mathbf{w}^T X^T \mathbf{y} + \mathbf{y}^T \mathbf{y})$$

Differentiate with respect to vector $\mathbf{w}$ using matrix calculus rules ($\nabla_{\mathbf{w}} (\mathbf{w}^T A \mathbf{w}) = 2 A \mathbf{w}$ for symmetric $A$):
$$\nabla_{\mathbf{w}} J(\mathbf{w}) = \frac{2}{n} (X^T X \mathbf{w} - X^T \mathbf{y}) = \mathbf{0}$$
$$X^T X \mathbf{w} = X^T \mathbf{y}$$

Since $X^T X$ is invertible / מכיוון ש-$X^T X$ הפיכה:
$$\mathbf{w}^* = (X^T X)^{-1} X^T \mathbf{y}$$

---

#### Part B: Ridge Invertibility Proof / סעיף ב': הוכחת הפיכות מודל Ridge
Let $\mathbf{v} \in \mathbb{R}^d$ be any non-zero vector ($\mathbf{v} \neq \mathbf{0}$). Compute quadratic form / חישוב התבנית הריבועית:
$$\mathbf{v}^T (X^T X + \lambda I_d) \mathbf{v} = \mathbf{v}^T X^T X \mathbf{v} + \lambda \mathbf{v}^T I_d \mathbf{v} = (X\mathbf{v})^T (X\mathbf{v}) + \lambda \mathbf{v}^T \mathbf{v}$$
$$\mathbf{v}^T (X^T X + \lambda I_d) \mathbf{v} = \|X\mathbf{v}\|^2 + \lambda \|\mathbf{v}\|^2$$

Since $\|X\mathbf{v}\|^2 \ge 0$ and $\lambda \|\mathbf{v}\|^2 > 0$ for $\lambda > 0$ and $\mathbf{v} \neq \mathbf{0}$:
$$\mathbf{v}^T (X^T X + \lambda I_d) \mathbf{v} > 0 \quad \forall \mathbf{v} \neq \mathbf{0}$$

Thus, $(X^T X + \lambda I_d)$ is **strictly positive definite** ($A \succ 0$), which guarantees that all its eigenvalues are strictly positive ($\lambda_i > 0$), making it **always invertible**.
