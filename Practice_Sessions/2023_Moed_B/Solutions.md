# 💡 2023 Moed B — Official Solutions | פתרונות רשמיים — מועד ב׳ 2023

---

## Question 1: Softmax Cross-Entropy Loss Gradient Derivation
## שאלה 1: גזירת פונקציית הפסד Softmax Cross-Entropy

### 🇮🇱 עברית & 🇬🇧 English Full Analytical Derivation

Loss function / פונקציית ההפסד:
$$\mathcal{L} = -\ln P_y = -\ln \left( \frac{e^{\mathbf{w}_y^T \mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}} \right) = -\mathbf{w}_y^T \mathbf{x} + \ln \left( \sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}} \right)$$

#### Case 1: $m = y$ (Target Class / וקטור המחלקה הנכונה)
$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}_y} = \frac{\partial}{\partial \mathbf{w}_y} \left( -\mathbf{w}_y^T \mathbf{x} \right) + \frac{\partial}{\partial \mathbf{w}_y} \ln \left( \sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}} \right)$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}_y} = -\mathbf{x} + \frac{e^{\mathbf{w}_y^T \mathbf{x}} \mathbf{x}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}} = -\mathbf{x} + P_y \mathbf{x} = \mathbf{(P_y - 1)\mathbf{x}}$$

#### Case 2: $m \neq y$ (Non-Target Class / וקטור מחלקה שאינה נכונה)
$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}_m} = \frac{\partial}{\partial \mathbf{w}_m} \ln \left( \sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}} \right) = \frac{e^{\mathbf{w}_m^T \mathbf{x}} \mathbf{x}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}} = \mathbf{P_m \mathbf{x}}$$

#### Unified Gradient Expression / נוסחת נגזרת מאוחדת
$$\nabla_{\mathbf{w}_m} \mathcal{L} = (P_m - \mathbf{1}\{y = m\}) \mathbf{x}$$

---

## Question 2: Decision Tree Information Gain & Entropy
## שאלה 2: עצי החלטה, אנטרופיה ורווח מידע (Information Gain)

### 🇮🇱 עברית & 🇬🇧 English Numerical Calculations

#### 1. Parent Entropy $H(S)$ / אנטרופיית הורה
$$H(S) = -p_+ \log_2(p_+) - p_- \log_2(p_-) = -\frac{9}{14} \log_2\left(\frac{9}{14}\right) - \frac{5}{14} \log_2\left(\frac{5}{14}\right)$$
$$H(S) \approx -0.64286(-0.6374) - 0.35714(-1.4854) \approx 0.4097 + 0.5305 = \mathbf{0.9402 \text{ bits}}$$

#### 2. Child Entropies $H(S_1), H(S_2)$ / אנטרופיות ילדים
* **Subset $S_1$ (5 samples: 2+, 3-):**
  $$H(S_1) = -\frac{2}{5}\log_2\left(\frac{2}{5}\right) - \frac{3}{5}\log_2\left(\frac{3}{5}\right) \approx \mathbf{0.9710 \text{ bits}}$$
* **Subset $S_2$ (9 samples: 7+, 2-):**
  $$H(S_2) = -\frac{7}{9}\log_2\left(\frac{7}{9}\right) - \frac{2}{9}\log_2\left(\frac{2}{9}\right) \approx \mathbf{0.7642 \text{ bits}}$$

#### 3. Weighted Conditional Entropy $H(S \mid A)$ / אנטרופיה מותנית
$$H(S \mid A) = \frac{|S_1|}{|S|} H(S_1) + \frac{|S_2|}{|S|} H(S_2) = \frac{5}{14}(0.9710) + \frac{9}{14}(0.7642)$$
$$H(S \mid A) \approx 0.3468 + 0.4913 = \mathbf{0.8381 \text{ bits}}$$

#### 4. Information Gain $\text{IG}(S, A)$ / רווח מידע
$$\text{IG}(S, A) = H(S) - H(S \mid A) = 0.9402 - 0.8381 = \mathbf{0.1021 \text{ bits}}$$
