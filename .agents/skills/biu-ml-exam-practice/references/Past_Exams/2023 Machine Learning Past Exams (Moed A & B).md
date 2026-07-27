---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Resource
date_added: 2026-07-26
---
# 2023 Machine Learning Past Exams (Moed A & B)

**Course:** [[English]] (Bar-Ilan University)  
**Instructors:** Prof. Joseph Keshet & Prof. Gal Chechik  
**Master Index:** [[Past Exams Master Index]]

---

## 📝 2023 Moed A — Exam Questions & Solutions

### Question 1: Maximum Likelihood Estimation (MLE) & Normal Distribution
**Problem Statement:**  
Given $n$ independent and identically distributed (i.i.d.) samples $x_1, \dots, x_n \sim \mathcal{N}(\mu, \sigma^2)$ drawn from a 1D Gaussian distribution with unknown mean $\mu$ and known variance $\sigma^2$.

#### Analytical Step-by-Step Proof:
1. **Likelihood Function $L(\mu)$:**
   $$L(\mu) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$$
2. **Log-Likelihood Function $\ell(\mu) = \ln L(\mu)$:**
   $$\ell(\mu) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^n (x_i - \mu)^2$$
3. **Stationarity Condition $\frac{\partial \ell}{\partial \mu} = 0$:**
   $$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2} \sum_{i=1}^n (x_i - \mu) = 0 \implies \sum_{i=1}^n x_i - n\mu = 0$$
4. **MLE Estimate $\hat{\mu}_{\text{MLE}}$:**
   $$\hat{\mu}_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^n x_i = \bar{x}$$

---

### Question 2: Linear Regression & Ridge Duality
**Problem Statement:**  
Consider linear regression with design matrix $X \in \mathbb{R}^{n \times d}$ and label vector $\mathbf{y} \in \mathbb{R}^n$.

1. **Part A (Unconstrained Least Squares Solution):**  
   If $X^T X$ is invertible ($n \ge d$ and $X$ has full column rank), the unique closed-form solution minimizing $\text{MSE}(\mathbf{w}) = \frac{1}{n} \|X\mathbf{w} - \mathbf{y}\|^2$ is:
   $$\mathbf{w}^* = (X^T X)^{-1} X^T \mathbf{y}$$
2. **Part B (Ridge Regression Regularization):**  
   When $d > n$ (overparameterized regime), $X^T X$ is singular and non-invertible. Adding $L_2$ weight decay $\frac{\lambda}{2}\|\mathbf{w}\|^2$ yields Ridge Regression:
   $$\mathbf{w}^*_{\text{Ridge}} = (X^T X + \lambda I_d)^{-1} X^T \mathbf{y}$$
   *(The matrix $X^T X + \lambda I_d$ is strictly positive-definite and always invertible for any $\lambda > 0$).*

---

## 📝 2023 Moed B — Softmax Gradient & Information Gain

### Question 3: Softmax Cross-Entropy Loss Gradient Derivation
**Problem Statement:**  
Derive the gradient of the Cross-Entropy loss $\mathcal{L} = -\ln P(y = k \mid \mathbf{x})$ with respect to the weight vector $\mathbf{w}_m$ of class $m$, where $P_k = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}}$.

#### Analytical Proof:
1. **Case 1: $m = y$ (Target Class Weight Vector):**
   $$\mathcal{L} = -\mathbf{w}_y^T \mathbf{x} + \ln\left( \sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}} \right)$$
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{w}_y} = -\mathbf{x} + \frac{e^{\mathbf{w}_y^T \mathbf{x}} \mathbf{x}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}} = (P_y - 1)\mathbf{x}$$
2. **Case 2: $m \neq y$ (Non-Target Class Weight Vector):**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{w}_m} = \frac{\partial}{\partial \mathbf{w}_m} \left( \ln \sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}} \right) = \frac{e^{\mathbf{w}_m^T \mathbf{x}} \mathbf{x}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}} = P_m \mathbf{x}$$
3. **Unified Vector Formula:**
   $$\nabla_{\mathbf{w}_m} \mathcal{L} = (P_m - \mathbf{1}\{y = m\}) \mathbf{x}$$

---

### Question 4: Decision Tree Information Gain & Entropy
**Problem Statement:**  
Given a dataset $S$ with 14 instances (9 positive, 5 negative). A categorical feature $A$ splits $S$ into $S_1$ (5 instances: 2 positive, 3 negative) and $S_2$ (9 instances: 7 positive, 2 negative). Compute the Information Gain $\text{IG}(S, A)$.

#### Step-by-Step Calculation:
1. **Parent Entropy $H(S)$:**
   $$H(S) = -\frac{9}{14}\log_2\left(\frac{9}{14}\right) - \frac{5}{14}\log_2\left(\frac{5}{14}\right) \approx 0.940 \text{ bits}$$
2. **Child Entropies $H(S_1)$ and $H(S_2)$:**
   $$H(S_1) = -\frac{2}{5}\log_2\left(\frac{2}{5}\right) - \frac{3}{5}\log_2\left(\frac{3}{5}\right) \approx 0.971 \text{ bits}$$
   $$H(S_2) = -\frac{7}{9}\log_2\left(\frac{7}{9}\right) - \frac{2}{9}\log_2\left(\frac{2}{9}\right) \approx 0.764 \text{ bits}$$
3. **Weighted Conditional Entropy $H(S \mid A)$:**
   $$H(S \mid A) = \frac{5}{14} H(S_1) + \frac{9}{14} H(S_2) = \frac{5}{14}(0.971) + \frac{9}{14}(0.764) \approx 0.838 \text{ bits}$$
4. **Information Gain $\text{IG}(S, A)$:**
   $$\text{IG}(S, A) = H(S) - H(S \mid A) = 0.940 - 0.838 = \mathbf{0.102 \text{ bits}}$$
