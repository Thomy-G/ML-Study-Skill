# 📝 2025 Moed A — Complete Exam Practice Session | מועד א׳ 2025 — אימון למבחן מלא

**Course / קורס:** Bar-Ilan University Machine Learning 89-511 / למידת מכונה, אוניברסיטת בר-אילן  
**Instructors / מרצים:** Prof. Gal Chechik & Prof. Joseph Keshet  

---

## 🔹 Part A: True/False Questions (21 Points Total)
## חלק א': שאלות נכון / לא נכון (21 נקודות)

### 📌 Question 1 (12 Points / 12 נקודות)
Mark whether each statement below is **Correct (True)** or **Incorrect (False)** / סמן/י נכון או לא נכון:

* **a.** Stochastic gradient descent has fewer computations per update than full gradient descent.  
  *לשיטת SGD יש פחות חישובים בכל עדכון מאשר לגרדיאנט דסנט מלא (Full Batch Gradient Descent).*
* **b.** Using ADAM instead of using SGD reduces overfitting.  
  *שימוש ב-ADAM במקום ב-SGD מפחית את התאמת היתר (Overfitting).*
* **c.** BatchNorm makes training of deep networks converge faster.  
  *שכבת BatchNorm גורמת לאימון רשתות עמוקות להתכנס מהר יותר.*
* **d.** With SGD, Weight Decay has the equivalent effect of adding a regularization term to the loss function.  
  *ב-SGD, לשימוש ב-Weight Decay יש אפקט שקול להוספת איבר רגולריזציה לפונקציית ההפסד.*

---

### 📌 Question 2 (9 Points / 9 נקודות)
Mark whether each statement below is **Correct (True)** or **Incorrect (False)** / סמן/י נכון או לא נכון:

* **a.** Let $S_n = \{(x_i, y_i)\}_{i=1}^n$ be a set of $n$ labeled samples and $H$ be a hypothesis class with $\text{VCdim}(H) = d < n$. Then, there cannot be $h \in H$ such that $h$ classifies the samples without errors.  
  *תהי $S_n$ קבוצה של $n$ דוגמאות מסומנות ו- $H$ מחלקת היפותזות עם $\text{VCdim}(H) = d < n$. אזי לא יכולה להיות היפותזה $h \in H$ המסווגת את הדוגמאות ללא שגיאות.*
* **b.** $K$-Means clustering partitions the input space with a linear decision boundary when $K=2$.  
  *אלגוריתם $K$-Means מחלק את מרחב הקלט עם גבול החלטה לינארי כאשר $K=2$.*
* **c.** A BatchNorm layer has two learnable parameters per feature, for shift and scale.  
  *לשכבת BatchNorm יש שני פרמטרים נלמדים לכל תכונה (Feature), עבור הזזה (Shift) וגודל (Scale).*

---

## 🔹 Part B: Open Questions (Choose 2 out of 3, 40 Points Each)
## חלק ב': שאלות פתוחות (בחר/י 2 מתוך 3, 40 נקודות לשאלה)

---

### 📌 Question 3: Logistic Regression
### שאלה 3: רגרסיה לוגיסטית

Consider training logistic regression models for two datasets $D_1, D_2$ with binary labels $D_1 = \{(x_i^{(1)}, y_i^{(1)})\}_{i=1}^{n_1}, D_2 = \{(x_i^{(2)}, y_i^{(2)})\}_{i=1}^{n_2}$ where $x_i^{(1)}, x_i^{(2)} \ge 0$ and $y_i^{(1)}, y_i^{(2)} \in \{0, 1\}$. Assume datasets are distinct.

* **a.** Derive the Negative Log-Likelihood loss (NLL loss), assuming it is applied to the first dataset $D_1$.  
  *פתח/י את פונקציית ההפסד NLL עבור אוסף הנתונים הראשון $D_1$.*
* **b.** Apply the chain rule to find the gradient of the loss, showing that it equals to:  
  $$\nabla \text{Err}_{D_1}(w) = -\sum_{i=1}^{n_1} x_i^{(1)} \left(y_i^{(1)} - \sigma(w x_i^{(1)})\right)$$  
  *השתמש/י בכלל השרשרת למציאת הנגזרת של פונקציית ההפסד והוכח/י את הנוסחה מעלה.*
* **c.** Show that the gradient of the error is a strictly monotonically increasing function of $w$, i.e., if $w^1 > w^2$ then $\nabla \text{Err}_{D_1}(w^1) > \nabla \text{Err}_{D_1}(w^2)$.  
  *הוכח/י שנגזרת ההפסד היא פונקציה מונוטונית עולה ממש לפי $w$.*
* **d.** Let $w^{(1)}$ be the logistic regression coefficient from training on $D_1$, and $w^{(2)}$ be the coefficient from training on $D_2$. Let $w^*$ be the coefficient from training on $D_1 \cup D_2$.  
  * **(i)** Show that $\nabla \text{Err}_{D_1 \cup D_2}(w^*) = \nabla \text{Err}_{D_1}(w^*) + \nabla \text{Err}_{D_2}(w^*)$.  
  * **(ii)** Show that if $w^* < \min(w^{(1)}, w^{(2)})$, then $\nabla \text{Err}_{D_1 \cup D_2}(w^*) < 0$.  
  * **(iii)** Show that if $w^* > \max(w^{(1)}, w^{(2)})$, then $\nabla \text{Err}_{D_1 \cup D_2}(w^*) > 0$. Conclude that $\min(w^{(1)}, w^{(2)}) \le w^* \le \max(w^{(1)}, w^{(2)})$.

---

### 📌 Question 4: Naïve Bayes
### שאלה 4: בייס נאיבי

Let $\mathbf{x} = (x_1, \dots, x_d) \in \mathbb{R}^d$ and $Y \in \{0, 1\}$. Denote class prior $P(Y=1) = \pi_1, P(Y=0) = \pi_0 = 1 - \pi_1$. Assume Gaussian distributions $P(x_j \mid Y=c) = \mathcal{N}(\mu_{c,j}, \sigma_j^2)$ with shared variance $\sigma_j^2$ across classes, and conditional independence of features given class.

* **a.** Write the conditional log-likelihood $\ln P(\mathbf{x} \mid Y=c)$ in terms of $x_j, \mu_{0,j}, \mu_{1,j}$ and $\sigma_j^2$.  
  *רשום/רשמי את הלוג-נראות המותנית כפונקציה של הפרמטרים.*
* **b.** Derive the log-posterior ratio: $\ln\left(\frac{P(Y=1 \mid \mathbf{x})}{P(Y=0 \mid \mathbf{x})}\right)$ in terms of $x_j, \mu_{0,j}, \mu_{1,j}$ and $\sigma_j^2$.  
  *פתח/י את יחס הלוג-פוסטריור.*
* **c.** Prove that Naïve Bayes in this case is a linear classifier. Express the decision boundary explicitly in terms of the model parameters and define the decision rule.  
  *הוכח/י שמסווג בייס נאיבי במקרה זה הוא מסווג לינארי וגזור את גבול ההחלטה והכלל המפורש.*

---

### 📌 Question 5: $K$-Means & Expectation-Maximization (EM)
### שאלה 5: $K$-Means ואלגוריתם תוחלת-מזעור (EM)

Let $x_1, \dots, x_n \in \mathbb{R}^d$ be i.i.d. samples from a Gaussian Mixture Model (GMM) with parameters $\theta = (\pi_k, \mu_k, \Sigma_k : k=1,\dots,K)$. The GMM likelihood is $L(\theta) = \prod_{i=1}^n \left( \sum_{k=1}^K \pi_k \mathcal{N}(x_i; \mu_k, \Sigma_k) \right)$.

* **a.** Add latent binary variables $z_i = (z_{i,1}, \dots, z_{i,K})$ indicating cluster membership. Write full likelihood $p_\theta(x_1,\dots,x_n, z_1,\dots,z_n)$.  
  *הוסף/י משתנים חבויים $z_i$ ורשום/רשמי את פונקציית הנראות המלאה.*
* **b.** Show using Bayes rule that $q(z_{i,k}=1) = \frac{\pi_k \mathcal{N}(x_i; \mu_k, \Sigma_k)}{\sum_l \pi_l \mathcal{N}(x_i; \mu_l, \Sigma_l)}$. Assume $\pi_k = \frac{1}{K}, \Sigma_k = \sigma^2 I$. For each sample $x_i$, assume a single closest mean $\mu_{k_i^*}$. Write the posterior for limit $\sigma \to 0$ and show probabilities become binary $\{0, 1\}$.  
  *הראה/י את פיתוח הפוסטריור ובחן/בדוק/י את הגבול $\sigma \to 0$.*
* **c.** Write the $K$-means algorithm. For $\sigma \to 0$, determine the condition under which the E-step from (b) is equivalent to Step 1 of $K$-means and show equivalence.  
  *רשום/רשמי את אלגוריתם $K$-Means והוכח/י שקילות של שלב ה-E.*
* **d.** Recall M-step formula $\mu_k^{\text{new}} = \frac{\sum q(z_{i,k}=1) x_i}{\sum q(z_{i,k}=1)}$. Show that as $\sigma \to 0$, M-step is equivalent to Step 2 of $K$-means. Deduce that EM performs the exact same steps as $K$-means.  
  *הוכח/י שקילות של שלב ה-M והסק/י שקילות מלאה בין EM ל-$K$-Means.*
