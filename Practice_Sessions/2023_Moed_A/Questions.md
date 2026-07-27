# 📝 2023 Moed A — Exam Practice Session | מועד א׳ 2023 — אימון למבחן

**Course / קורס:** Bar-Ilan University Machine Learning 89-511 / למידת מכונה, אוניברסיטת בר-אילן  
**Instructors / מרצים:** Prof. Gal Chechik & Prof. Joseph Keshet  

---

## 📌 Question 1: Maximum Likelihood Estimation (MLE) for 1D Gaussian Mean
## שאלה 1: אומד סבירות מרבית (MLE) לתוחלת התפלגות נורמלית

### 🇮🇱 עברית
נתונים $n$ מדגמים בלתי תלויים ובעלי אותה התפלגות (i.i.d.) $x_1, \dots, x_n \sim \mathcal{N}(\mu, \sigma^2)$ מתוך התפלגות נורמלית חד-ממדית עם תוחלת לא ידועה $\mu$ ושונות ידועה $\sigma^2$.

פפתח/י באופן אנליטי מפורט את אומד הסבירות המרבית (Maximum Likelihood Estimator - MLE) עבור התוחלת $\hat{\mu}_{\text{MLE}}$:
1. רשום/רשמי את פונקציית הנראות (Likelihood Function $L(\mu)$).
2. פתח/י את פונקציית הלוג-נראות (Log-Likelihood Function $\ell(\mu)$).
3. גזור/גזרי לפי $\mu$, השווה/י לאפס, ומצא/י את $\hat{\mu}_{\text{MLE}}$.

### 🇬🇧 English
Given $n$ independent and identically distributed (i.i.d.) samples $x_1, \dots, x_n \sim \mathcal{N}(\mu, \sigma^2)$ drawn from a 1D Gaussian distribution with unknown mean $\mu$ and known variance $\sigma^2$.

Derive the analytical Maximum Likelihood Estimator (MLE) for the mean $\hat{\mu}_{\text{MLE}}$ step-by-step:
1. Write down the Likelihood function $L(\mu)$.
2. Derive the Log-Likelihood function $\ell(\mu)$.
3. Differentiate with respect to $\mu$, set to zero, and solve for $\hat{\mu}_{\text{MLE}}$.

---

## 📌 Question 2: Linear Regression & Ridge Regularization Duality
## שאלה 2: רגרסיה לינארית ודואליות רגולריזציית Ridge

### 🇮🇱 עברית
בהנתן מודל רגרסיה לינארית עם מטריצת תכונות $X \in \mathbb{R}^{n \times d}$ ווקטור תגיות $\mathbf{y} \in \mathbb{R}^n$.

1. **סעיף א' (Unconstrained OLS):**  
   אם $X^T X$ היא מטריצה הפיכה ($n \ge d$ ו-$X$ מדרגת עמודות מלאה), הוכיחי/הוכח כי הפתרון היחיד במבנה סגור הממזער את שגיאת ה-MSE הוא:
   $$\mathbf{w}^* = (X^T X)^{-1} X^T \mathbf{y}$$

2. **סעיף ב' (Ridge Regularization):**  
   כאשר $d > n$ (משטר עודף פרמטרים), המטריצה $X^T X$ היא סינגולרית ואינה הפיכה. הוספת רגולריזציית $L_2$ מניבה את מודל Ridge:
   $$\mathbf{w}^*_{\text{Ridge}} = (X^T X + \lambda I_d)^{-1} X^T \mathbf{y}$$
   הוכח/י כי המטריצה $X^T X + \lambda I_d$ היא חיובית ממש (Strictly Positive Definite) ולכן הפיכה לכל $\lambda > 0$.

### 🇬🇧 English
Consider linear regression with design matrix $X \in \mathbb{R}^{n \times d}$ and label vector $\mathbf{y} \in \mathbb{R}^n$.

1. **Part A (Unconstrained Least Squares Solution):**  
   If $X^T X$ is invertible ($n \ge d$ and $X$ has full column rank), prove that the unique closed-form solution minimizing $\text{MSE}(\mathbf{w}) = \frac{1}{n} \|X\mathbf{w} - \mathbf{y}\|^2$ is:
   $$\mathbf{w}^* = (X^T X)^{-1} X^T \mathbf{y}$$

2. **Part B (Ridge Regression Regularization):**  
   When $d > n$ (overparameterized regime), $X^T X$ is singular and non-invertible. Adding $L_2$ weight decay $\frac{\lambda}{2}\|\mathbf{w}\|^2$ yields Ridge Regression:
   $$\mathbf{w}^*_{\text{Ridge}} = (X^T X + \lambda I_d)^{-1} X^T \mathbf{y}$$
   Prove that the matrix $X^T X + \lambda I_d$ is strictly positive-definite and always invertible for any $\lambda > 0$.
