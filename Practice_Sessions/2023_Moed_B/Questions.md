# 📝 2023 Moed B — Exam Practice Session | מועד ב׳ 2023 — אימון למבחן

**Course / קורס:** Bar-Ilan University Machine Learning 89-511 / למידת מכונה, אוניברסיטת בר-אילן  
**Instructors / מרצים:** Prof. Gal Chechik & Prof. Joseph Keshet  

---

## 📌 Question 1: Softmax Cross-Entropy Loss Gradient Derivation
## שאלה 1: גזירת פונקציית הפסד Softmax Cross-Entropy

### 🇮🇱 עברית
פפתח/י אנליטית את הנגזרת של פונקציית ההפסד Cross-Entropy $\mathcal{L} = -\ln P(y = k \mid \mathbf{x})$ לפי וקטור המשקולות $\mathbf{w}_m$ של מחלקה $m$, כאשר ההסתברות $P_k$ מוגדרת כ:
$$P_k = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}}$$

1. **מקרה 1:** עבור $m = y$ (וקטור המשקולות של המחלקה הנכונה).
2. **מקרה 2:** עבור $m \neq y$ (וקטור המשקולות של מחלקה שאינה המחלקה הנכונה).
3. **נוסחה מאוחדת:** רשום/רשמי את הנגזרת המאוחדת בעזרת פונקציית האינדיקטור $\mathbf{1}\{y = m\}$.

### 🇬🇧 English
Derive the analytical gradient of the Cross-Entropy loss $\mathcal{L} = -\ln P(y = k \mid \mathbf{x})$ with respect to the weight vector $\mathbf{w}_m$ of class $m$, where $P_k = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}}$.

1. **Case 1:** For $m = y$ (target class weight vector).
2. **Case 2:** For $m \neq y$ (non-target class weight vector).
3. **Unified Formula:** Express the unified gradient using the indicator function $\mathbf{1}\{y = m\}$.

---

## 📌 Question 2: Decision Tree Information Gain & Entropy
## שאלה 2: עצי החלטה, אנטרופיה ורווח מידע (Information Gain)

### 🇮🇱 עברית
נתון אוסף נתונים $S$ המכיל 14 דוגמאות (9 חיוביות, 5 שליליות).  
תכונה קטגורית $A$ מפצלת את $S$ לשני תת-אוספים:
* $S_1$: 5 דוגמאות (2 חיוביות, 3 שליליות).
* $S_2$: 9 דוגמאות (7 חיוביות, 2 שליליות).

1. חשב/י את אנטרופיית הורה $H(S)$.
2. חשב/י את האנטרופיות של הילדים $H(S_1)$ ו-$H(S_2)$.
3. חשב/י את האנטרופיה המותנית המשוקללת $H(S \mid A)$.
4. חשב/י את רווח המידע $\text{IG}(S, A) = H(S) - H(S \mid A)$.

### 🇬🇧 English
Given a dataset $S$ with 14 instances (9 positive, 5 negative). A categorical feature $A$ splits $S$ into two subsets:
* $S_1$: 5 instances (2 positive, 3 negative).
* $S_2$: 9 instances (7 positive, 2 negative).

1. Compute parent entropy $H(S)$.
2. Compute child entropies $H(S_1)$ and $H(S_2)$.
3. Compute weighted conditional entropy $H(S \mid A)$.
4. Compute Information Gain $\text{IG}(S, A) = H(S) - H(S \mid A)$.
