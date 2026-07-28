# 📝 Advanced VC-Dimension & Gradient Convergence Practice Session | אימון בנושאי ממד VC ורגרסיה

**Course / קורס:** Bar-Ilan University Machine Learning 89-511 / למידת מכונה, אוניברסיטת בר-אילן  
**Topics Covered / נושאים:** PAC Learning & Realizability, VC-Dimension of Axis-Aligned Hyper-Rectangles ($H_{\text{rec}}^d$), Gradient Descent Convergence of Linear Regression, Deep Learning Backpropagation & Imbalanced Loss.

---

## 🔹 Part A: True/False Questions (20 Points Total)
## חלק א': שאלות נכון / לא נכון (20 נקודות)

### 📌 Question 1 (10 Points) — Learning Theory & ERM
* **a.** In the realizable case, the empirical training error of the optimal hypothesis is zero.  
  *במקרה הניתן למימוש (Realizable), שגיאת האימון האמפירית של ההיפותזה האופטימלית היא אפס.*
* **b.** The Empirical Risk Minimizer (ERM) hypothesis is always unique.  
  *היפותזת ממוזער הסיכון האמפירי (ERM) היא תמיד יחידה.*
* **c.** Under the 0-1 loss function, the true expected generalization error of a hypothesis $h$ equals the probability that $h$ misclassifies a randomly drawn data point.  
  *תחת הפסד 0-1, שגיאת ההכללה הנדרשת של היפותזה $h$ שווה להסתברות ש-$h$ תסווג באופן שגוי דגימה אקראית.*

---

### 📌 Question 2 (10 Points) — Deep Learning & Cross-Validation
* **a.** In deep learning, most errors are due to issues with expressivity rather than generalization and optimization.  
  *בלמידה עמוקה, מרבית השגיאות נובעות מבעיות בכושר הביטוי (Expressivity) ולא מבעיות הכללה ואופטימיזציה.*
* **b.** A primary benefit of using $K$-fold cross-validation is to get a more reliable estimate of a model's performance on unseen data compared to a single train/validation split.  
  *היתרון המרכזי בשימוש ב-$K$-fold cross-validation הוא קבלת הערכה אמינה יותר של ביצועי המודל על נתונים שלא נראו בהשוואה לחלוקה יחידה של אימון/אימות.*
* **c.** Within a Skip block in ResNet, the output of the convolution layers is added to an identity function over the input.  
  *בתוך בלוק דילוג (Skip block) ב-ResNet, פלט שכבות הקונבולוציה מתווסף לפונקציית הזהות של הקלט.*
* **d.** Typically, filters learned in the early layers of convolutional neural networks capture low-level visual features compared to later layers which capture semantic ones.  
  *בדרך כלל, פילטרים שנלמדים בשכבות המוקדמות של רשת קונבולוציה תופסים מאפיינים ויזואליים ברמה נמוכה בהשוואה לשכבות מאוחרות התופסות מאפיינים סמנטיים.*

---

## 🔹 Part B: Open Questions (Solve 2 out of 3, 40 Points Each)
## חלק ב': שאלות פתוחות (בחר/י 2 מתוך 3, 40 נקודות לשאלה)

---

### 📌 Question 3: Convergence of Linear Regression
### שאלה 3: התכנסות רגרסיה לינארית

We aim to solve a linear regression problem. Given a dataset $\{(x_i, y_i)\}_{i=1}^n$ where $x_i \in \mathbb{R}^d$ and $y_i \in \mathbb{R}$, let $X \in \mathbb{R}^{n \times d}$ be the sample design matrix and $y \in \mathbb{R}^n$ be the label vector.

* **a.** Formally define the linear regression problem as an optimization task. State the optimization objective $L(w)$ as a function of parameter vector $w \in \mathbb{R}^d$.  
  *הגדר/י פורמלית את בעיית הרגרסיה הלינארית כמשימת אופטימיזציה ונסח/י את פונקציית המטרה $L(w)$.*
* **b.** Denoting the objective function by $L(w)$, its gradient is $\nabla L(w) = \frac{2}{n}X^T(Xw - y)$. Assuming $X^T X$ is invertible, prove that $w^* = (X^T X)^{-1}X^T y$ is the unique optimal solution. Provide a simplified version if the sample matrix is orthonormal ($X^T X = I$).  
  *בהנחה ש-$X^T X$ הפיכה, הוכח/י ש-$w^*$ הוא פתרון יחיד ורשום/רשמי פישוט כאשר $X^T X = I$.*
* **c.** Assume we minimize $L(w)$ using gradient descent with learning rate $\eta > 0$. Prove recursively that:
  $$w_t - w^* = \left(I - \frac{2\eta}{n}X^T X\right)^t (w_0 - w^*)$$
  *הוכח/י באינדוקציה/רקורסיה את נוסחת התכנסות הצעדים של gradient descent.*
* **d.** Consider the 1-dimensional case ($d=1$). Prove that a necessary and sufficient condition for convergence $w_t \to w^*$ as $t \to \infty$ is:
  $$0 < \eta < \frac{n}{\|X\|^2}$$
  *עבור המקרה ה-1 ממדי ($d=1$), הוכח/י כי תנאי הכרחי ומספיק להתכנסות $w_t \to w^*$ הוא $0 < \eta < \frac{n}{\|X\|^2}$.*

---

### 📌 Question 4: VC-Dimension of Axis-Aligned Hyper-Rectangles
### שאלה 4: ממד VC של מלבנים מוצבים על הצירים

This question focuses on the VC-dimension for $d$-dimensional axis-aligned rectangles.

* **a.** Let $H_{\text{int}}$ be the hypothesis class of intervals over $\mathbb{R}$. Show that there exists a set $C \subseteq \mathbb{R}$ of size 2 shattered by $H_{\text{int}}$.  
  *הראה/י כי קיימת קבוצה של 2 נקודות ב-$\mathbb{R}$ הנשברת ע״י מחלקת הקטעים $H_{\text{int}}$.*
* **b.** Show that any set of 3 points cannot be shattered by $H_{\text{int}}$, proving that $\text{VCdim}(H_{\text{int}}) = 2$.  
  *הוכח/י שכל קבוצה של 3 נקודות אינה ניתנת לשבירה ע״י $H_{\text{int}}$, ובכך הוכח/י ש-$\text{VCdim}(H_{\text{int}}) = 2$.*
* **c.** Let $H_{\text{rec}}^d$ be $d$-dimensional axis-aligned hyper-rectangles. Show that a set $C \subseteq \mathbb{R}^d$ of size $2d$ can be shattered.  
  *הראה/י כי קבוצה בגודל $2d$ ניתנת לשבירה ע״י $H_{\text{rec}}^d$.*
* **d.** Show that any set of $2d+1$ points cannot be shattered by $H_{\text{rec}}^d$, proving that $\text{VCdim}(H_{\text{rec}}^d) = 2d$.  
  *הוכח/י שכל קבוצה של $2d+1$ נקודות אינה ניתנת לשבירה ע״י $H_{\text{rec}}^d$, ובכך הוכח/י ש-$\text{VCdim}(H_{\text{rec}}^d) = 2d$.*

---

### 📌 Question 5: Backpropagation & Imbalanced Loss
### שאלה 5: Backpropagation והפסד עבור נתונים לא מאוזנים

Consider classifying RGB images with giraffes ($y^{(i)} = 1$) or without giraffes ($y^{(i)} = 0$) using the multi-layer architecture:
$$z_1^{(i)} = W_1 x^{(i)} + b_1, \quad a_1^{(i)} = \text{ReLU}(z_1^{(i)}), \quad z_2^{(i)} = W_2 a_1^{(i)} + b_2, \quad \hat{y}^{(i)} = \sigma(z_2^{(i)})$$
$$L^{(i)} = \alpha y^{(i)} \ln(\hat{y}^{(i)}) + \beta(1 - y^{(i)}) \ln(1 - \hat{y}^{(i)})$$

* **a.** What are the dimensions of bias vectors $b_1$ and $b_2$?  
  *מהם הממדים של וקטורי ההטיה $b_1$ ו-$b_2$?*
* **b.** The dataset has 200 images with a giraffe and 2000 without. Give appropriate values for $\alpha, \beta$ assuming that in the real world the class distribution is 50-50.  
  *הדאטה סט מכיל 200 תמונות עם ג׳רפה ו-2000 בלעדיה. מצא/י ערכים מתאימים ל-$\alpha, \beta$ בהנחה שבעולם האמיתי ההתפלגות היא 50-50.*
* **c.** Derive the update rules for all parameters ($W_1, b_1, W_2, b_2$) assuming gradient descent.  
  *גזור/גזרי את כללי העדכון עבור כל הפרמטרים ב-Gradient Descent.*
* **d.** Add $L_2$ regularization with parameter $\lambda$. Write the new update rule for $W_1$. Should your answer to part (b) change?  
  *הוסף/י רגולריזציית $L_2$ ורשום/רשמי את כלל העדכון החדש עבור $W_1$. האם תשובתך לסעיף (ב) צריכה להשתנות?*
