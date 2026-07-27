# 💡 2022 Moed A — Official Solutions | פתרונות רשמיים — מועד א׳ 2022

---

## Question 1: $k$-NN Distance Metric & Metric Space Properties
## שאלה 1: פונקציית מרחק ב-$k$-NN ותכונות מרחב מטרי

### 🇮🇱 עברית — פתרון מלא
**תשובה:** הגידול אינו מטריקה תקפה ($d(\mathbf{x}, \mathbf{y})$ אינה פונקציית מרחק).  
פונקציית מרחק במרחב מטרי חייבת לקיים 4 תנאים:
1. אי-שליליות: $d(\mathbf{x}, \mathbf{y}) \ge 0$
2. זהות הבלתי-מובחנים: $d(\mathbf{x}, \mathbf{y}) = 0 \iff \mathbf{x} = \mathbf{y}$
3. סימטריה: $d(\mathbf{x}, \mathbf{y}) = d(\mathbf{y}, \mathbf{x})$
4. **אי-שוויון המשולש**: $d(\mathbf{x}, \mathbf{z}) \le d(\mathbf{x}, \mathbf{y}) + d(\mathbf{y}, \mathbf{z})$

**דוגמה נגדית להפרכת אי-שוויון המשולש:**  
נבחר שלוש נקודות ב-$\mathbb{R}^2$:  
$\mathbf{x} = (0, 0)$, $\mathbf{y} = (0, 1)$, $\mathbf{z} = (0, 2)$.

* $d(\mathbf{x}, \mathbf{z}) = |0-0| + |0-2|^2 = 0 + 4 = 4$
* $d(\mathbf{x}, \mathbf{y}) = |0-0| + |0-1|^2 = 0 + 1 = 1$
* $d(\mathbf{y}, \mathbf{z}) = |0-0| + |1-2|^2 = 0 + 1 = 1$

מתקבל: $d(\mathbf{x}, \mathbf{z}) = 4 > d(\mathbf{x}, \mathbf{y}) + d(\mathbf{y}, \mathbf{z}) = 1 + 1 = 2$.  
אי-שוויון המשולש נכשל, ולכן הפונקציה אינה מטריקה תקפה.

---

### 🇬🇧 English — Full Solution
**Answer:** Disprove. $d(\mathbf{x}, \mathbf{y})$ is **not** a valid metric.  
A valid metric space distance function must satisfy:
1. Non-negativity: $d(\mathbf{x}, \mathbf{y}) \ge 0$
2. Identity of indiscernibles: $d(\mathbf{x}, \mathbf{y}) = 0 \iff \mathbf{x} = \mathbf{y}$
3. Symmetry: $d(\mathbf{x}, \mathbf{y}) = d(\mathbf{y}, \mathbf{x})$
4. **Triangle Inequality**: $d(\mathbf{x}, \mathbf{z}) \le d(\mathbf{x}, \mathbf{y}) + d(\mathbf{y}, \mathbf{z})$

**Counterexample violating Triangle Inequality:**  
Choose three points in $\mathbb{R}^2$:  
$\mathbf{x} = (0, 0)$, $\mathbf{y} = (0, 1)$, $\mathbf{z} = (0, 2)$.

* $d(\mathbf{x}, \mathbf{z}) = |0-0| + |0-2|^2 = 4$
* $d(\mathbf{x}, \mathbf{y}) = |0-0| + |0-1|^2 = 1$
* $d(\mathbf{y}, \mathbf{z}) = |0-0| + |1-2|^2 = 1$

Here $d(\mathbf{x}, \mathbf{z}) = 4 > d(\mathbf{x}, \mathbf{y}) + d(\mathbf{y}, \mathbf{z}) = 2$. Thus, Triangle Inequality fails.

---

## Question 2: CNN Shapes & Parameter Counting
## שאלה 2: ממדי פלט וחישוב פרמטרים ב-CNN

### 🇮🇱 עברית & 🇬🇧 English Solutions

#### Part A: Spatial Dimensions / סעיף א': ממדים מרחביים
$$\text{Output Spatial Dim} = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$

* **Layer A Conv2D Output:** $\lfloor \frac{32 - 3 + 2(1)}{1} \rfloor + 1 = 32 \implies \mathbf{(10 \times 32 \times 32)}$
  * **After $2 \times 2$ MaxPool (Stride 2):** $\mathbf{(10 \times 16 \times 16)}$
* **Layer B Conv2D Output:** $\lfloor \frac{16 - 3 + 2(1)}{1} \rfloor + 1 = 16 \implies \mathbf{(10 \times 16 \times 16)}$
  * **After $2 \times 2$ MaxPool (Stride 2):** $\mathbf{(10 \times 8 \times 8)}$
* **Layer C Input (Flattened):** $10 \times 8 \times 8 = \mathbf{640 \text{ features}}$

#### Part B: Parameter Counting / סעיף ב': חישוב פרמטרים
$$\text{Params} = (K_h \times K_w \times C_{\text{in}} + 1) \times C_{\text{out}}$$

* **Layer A:** $(3 \times 3 \times 3 + 1) \times 10 = 28 \times 10 = \mathbf{280 \text{ parameters}}$
* **Layer B:** $(3 \times 3 \times 10 + 1) \times 10 = 91 \times 10 = \mathbf{910 \text{ parameters}}$
* **Layer C (FC):** $(640 + 1) \times 10 = \mathbf{6,410 \text{ parameters}}$
* **Total Parameters:** $280 + 910 + 6410 = \mathbf{7,600 \text{ parameters}}$.

---

## Question 3: Perceptron with Margin vs. Soft-SVM
## שאלה 3: פרספטרון עם שוליים מול Soft-SVM

### 🇮🇱 עברית — פתרון מלא
1. **סעיף א':** אלגוריתם ה-Perceptron מניח הפרדה לינארית מלאה במרחב התכונות. כאשר הנתונים אינם ניתנים להפרדה לינארית, לא קיימת יתר-מישור (Hyperplane) המפרידה את כל הנקודות עם שוליים $\gamma > 0$. לפיכך, משפטי ההתכנסות של ה-Perceptron אינם מתקיימים, והאלגוריתם נכנס ללולאה אינסופית ללא עצירה.
2. **סעיף ב':** אלגוריתם Soft-Margin SVM יתכנס בוודאות. Soft-SVM מכניס משתני הרפיה $\xi_i \ge 0$ אשר מאפשרים לנקודות מסוימות להפר את השוליים או להימצא בצד הלא נכון של הישר, תוך הענשת השגיאה בפונקציית המטרה:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i$$
הבעיה היא קמורה לחלוטין (Convex Optimization) ולכן בעלת פתרון יחיד וגלובלי.

### 🇬🇧 English — Full Solution
1. **Part A:** Margin Perceptron requires strict linear separability. When data is non-linearly separable, no hyperplane satisfies $y_i(\mathbf{w}^T \mathbf{x}_i) \ge \gamma > 0$ for all samples. As a result, the convergence guarantees break down, and the algorithm loops infinitely.
2. **Part B:** Soft-Margin SVM is guaranteed to converge. It introduces non-negative slack variables $\xi_i \ge 0$ allowing samples to violate the margin with penalty $C \sum \xi_i$ in its convex objective:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i$$
This is a strictly convex quadratic optimization problem with a unique global minimum.
