# 💡 2022 Moed B — Official Solutions | פתרונות רשמיים — מועד ב׳ 2022

---

## Question: 2-Layer MLP Backpropagation & Loss Computation
## שאלה: חישוב קדימה, הסתברויות Softmax ומפסיד NLL ברשת ניורונים

### 🇮🇱 עברית & 🇬🇧 English Step-by-Step Solution

#### Step 1: Hidden Layer Activation $h$ / שלב 1: אקטיבציה בשכבה הנסתרת
$$z_1 = w_1 x + b_1 = 2(2) + (-2) = 4 - 2 = 2$$
$$h = \sigma(2) = \frac{1}{1 + e^{-2}} = \frac{1}{1 + 0.1353} \approx \mathbf{0.8808}$$

---

#### Step 2: Logits Vector $\mathbf{z}$ / שלב 2: חישוב וקטור ה-Logits
$$\mathbf{z} = h \cdot \mathbf{w}_2 + \mathbf{b}_2 = 0.8808 \begin{bmatrix} 1 \\ 0.5 \\ -1 \end{bmatrix} + \begin{bmatrix} -1 \\ 1 \\ 0 \end{bmatrix}$$
$$\mathbf{z} = \begin{bmatrix} 0.8808 - 1 \\ 0.4404 + 1 \\ -0.8808 + 0 \end{bmatrix} = \begin{bmatrix} -0.1192 \\ 1.4404 \\ -0.8808 \end{bmatrix}$$

---

#### Step 3: Softmax Probabilities $\hat{\mathbf{p}}$ / שלב 3: הסתברויות Softmax
1. **Exponentiate logits / חישוב אקספוננטים:**
   $$e^{z_1} = e^{-0.1192} \approx 0.8876$$
   $$e^{z_2} = e^{1.4404} \approx 4.2224$$
   $$e^{z_3} = e^{-0.8808} \approx 0.4145$$

2. **Sum of Exponents / סכום המעריכים:**
   $$\sum_{j=1}^3 e^{z_j} = 0.8876 + 4.2224 + 0.4145 = 5.5245$$

3. **Softmax Vector / וקטור ההסתברויות:**
   $$\hat{\mathbf{p}} = \begin{bmatrix} \frac{0.8876}{5.5245} \\[4pt] \frac{4.2224}{5.5245} \\[4pt] \frac{0.4145}{5.5245} \end{bmatrix} \approx \begin{bmatrix} 0.1607 \\ 0.7643 \\ 0.0750 \end{bmatrix}$$

---

#### Step 4: Final NLL Loss $\mathcal{L}$ / שלב 4: חישוב הפסד NLL הסופי
True target label $y = \text{Class C}$ corresponds to index 3 ($p_3 = 0.0750$).
$$\mathcal{L} = -\ln(p_3) = -\ln(0.0750) \approx \mathbf{2.590}$$
