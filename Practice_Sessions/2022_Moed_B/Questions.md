# 📝 2022 Moed B — Exam Practice Session | מועד ב׳ 2022 — אימון למבחן

**Course / קורס:** Bar-Ilan University Machine Learning 89-511 / למידת מכונה, אוניברסיטת בר-אילן  
**Instructors / מרצים:** Prof. Gal Chechik & Prof. Joseph Keshet  

---

## 📌 Question: 2-Layer MLP Backpropagation & Loss Computation
## שאלה: חישוב קדימה, הסתברויות Softmax ומפסיד NLL ברשת ניורונים

### 🇮🇱 עברית
שהם מאמנת מודל סיווג ל-3 מחלקות בעזרת פונקציית הפסד Negative Log-Likelihood (NLL):
$$\mathcal{L} = -\ln P(y = y_t \mid \mathbf{x}_t)$$

ארכיטקטורת הרשת:
$$\mathbf{h} = \sigma(w_1 x + b_1)$$
$$\mathbf{z} = \mathbf{w}_2 \mathbf{h} + \mathbf{b}_2$$
$$\hat{\mathbf{p}} = \text{Softmax}(\mathbf{z})$$

פרמטרי הרשת הנתונים:
* $w_1 = 2, \quad b_1 = -2$
* $\mathbf{w}_2 = [1, 0.5, -1]^T, \quad \mathbf{b}_2 = [-1, 1, 0]^T$
* דוגמת קלט $x_i = 2$, ותווית אמת $y_i = \text{Class C}$ (אינדקס 3).

1. **סעיף א':** חשב/י את האקטיבציה $h = \sigma(z_1)$ בשכבה הנסתרת (כאשר $\sigma(z) = \frac{1}{1 + e^{-z}}$).
2. **סעיף ב':** חשב/י את וקטור ה-Logits $\mathbf{z} \in \mathbb{R}^3$.
3. **סעיף ג':** חשב/י את וקטור ההסתברויות $\hat{\mathbf{p}} = \text{Softmax}(\mathbf{z})$.
4. **סעיף ד':** חשב/י את מחיר ההפסד הכללי $\mathcal{L} = -\ln(p_3)$ עבור המחלקה הנכונה (Class C).

---

### 🇬🇧 English
Shoham trains a 3-class classification model using Negative Log-Likelihood (NLL) Loss:
$$\mathcal{L} = -\ln P(y = y_t \mid \mathbf{x}_t)$$

Network Architecture:
$$\mathbf{h} = \sigma(w_1 x + b_1)$$
$$\mathbf{z} = \mathbf{w}_2 \mathbf{h} + \mathbf{b}_2$$
$$\hat{\mathbf{p}} = \text{Softmax}(\mathbf{z})$$

Given parameters:
* $w_1 = 2, \quad b_1 = -2$
* $\mathbf{w}_2 = [1, 0.5, -1]^T, \quad \mathbf{b}_2 = [-1, 1, 0]^T$
* Input sample $x_i = 2$, true label $y_i = \text{Class C}$ (index 3).

1. **Part A:** Calculate the hidden layer activation $h = \sigma(z_1)$ where $\sigma(z) = \frac{1}{1 + e^{-z}}$.
2. **Part B:** Compute the logits vector $\mathbf{z} \in \mathbb{R}^3$.
3. **Part C:** Calculate the softmax probabilities vector $\hat{\mathbf{p}} = \text{Softmax}(\mathbf{z})$.
4. **Part D:** Compute the final NLL loss value $\mathcal{L} = -\ln(p_3)$ for the true target class (Class C).
