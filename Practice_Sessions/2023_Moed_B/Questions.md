# 📝 2023 Moed B — Complete Exam Practice Session | מועד ב׳ 2023 — אימון למבחן מלא

**Course / קורס:** Bar-Ilan University Machine Learning 89-511 / למידת מכונה, אוניברסיטת בר-אילן  
**Instructors / מרצים:** Prof. Gal Chechik & Prof. Joseph Keshet  

---

## 🔹 Part A: True/False & Yes/No Questions (25 Points Total)
## חלק א': שאלות נכון / לא נכון (25 נקודות)

### 📌 Question 1 (12.5 Points / 12.5 נקודות) — Overfitting Reduction
After training a deep network model, you observe that the test error is much larger than the training error. Which of the following is expected to improve (reduce) the empirical test error?
*לאחר אימון מודל רשת עמוקה, נצפה כי שגיאת המבחן גדולה בהרבה משגיאת האימון. אילו מהפעולות הבאות צפויה לשפר (להפחית) את שגיאת המבחן האמפירית?*

* **a.** Use a deeper model (add more layers), and retrain (Yes/No)  
  *להשתמש במודל עמוק יותר (להוסיף שכבות) ולאמן מחדש.*
* **b.** Regularize the weights of the model during training (Yes/No)  
  *להפעיל רגולריזציה על משקולות המודל בזמן האימון.*
* **c.** Retrain the model but this time on the test data (Yes/No)  
  *לאמן מחדש את המודל, אך הפעם על נתוני המבחן.*
* **d.** Use a larger test set (Yes/No)  
  *להשתמש בקבוצת מבחן גדולה יותר.*

---

### 📌 Question 2 (12.5 Points / 12.5 נקודות) — Core Principles
Mark which of the following statements is correct (Correct/Wrong) / סמן/י נכון או שגוי:

* **a.** A convolution layer typically has fewer parameters than a fully-connected layer.  
  *לשכבת קונבולוציה יש בדרך כלל פחות פרמטרים מאשר לשכבה מקושרת במלואה (Fully-Connected).*
* **b.** Adding skip connections to a ConvNet can help with training very deep MLPs.  
  *הוספת חיבורי דילוג (Skip Connections) ברשת קונבולוציה מסייעת באימון רשתות עמוקות מאוד.*
* **c.** ReLU is computationally less efficient than a Sigmoid.  
  *פונקציית האקטיבציה ReLU יעילה פחות מבחינה חישובית מפונקציית Sigmoid.*
* **d.** The Naive Bayes model assumes that features are independent.  
  *מודל בייס נאיבי מניח שהתכונות (Features) הן בלתי תלויות.*

---

## 🔹 Part B: Open Questions (Choose 3 out of 4, 25 Points Each)
## חלק ב': שאלות פתוחות (בחר/י 3 מתוך 4, 25 נקודות לשאלה)

---

### 📌 Question 3: Linear Classifiers & MAP Logistic Regression
### שאלה 3: מסווגים לינאריים ורגרסיה לוגיסטית MAP

* **a.** Define the optimization problem of logistic regression for binary classification. Formally define all variables, write the loss function, and explain the rationale behind the loss.  
  *הגדר/י פורמלית את בעיית האופטימיזציה של רגרסיה לוגיסטית. רשום/רשמי את פונקציית ההפסד והסבר/י את הרציונל העומד מאחוריה.*
* **b.** Derive an SGD algorithm for optimizing logistic regression. Simplify the gradient to reach a simple weight update rule. How would you select the step size (learning rate)?  
  *פפתח/י אלגוריתם SGD לאופטימיזציה של רגרסיה לוגיסטית וגזור/גזרי עדכון משקולות פשוט. כיצד תבחר/י את קצב הלמידה?*
* **c.** A student suggested using a Maximum A Posteriori (MAP) approach instead of Maximum Likelihood, with a Gaussian prior over weight vector $\mathbf{w}$: $P_0(w_i) = \mathcal{N}(0, 1)$ for each component $w_i$. Write the optimization problem with the new loss, derive the SGD update step, and explain the effect of this prior on the solution $\mathbf{w}$.  
  *ניסוח ואופטימיזציית MAP עם פרייור גאוסי $\mathcal{N}(0,1)$ על המשקולות. רשום/רשמי את עדכון ה-SGD החדש והסבר/י את השפעת הפרייור.*

---

### 📌 Question 4: Large Margin Classifiers & 1-NN Network Equivalence
### שאלה 4: מסווגי שוליים רחבים ושקילות לרשת 1-NN

* **a.** Consider a separable Kernel SVM classifier. Write the formula for predicting label $y$ of new sample $\mathbf{x}$ using kernel $K(\mathbf{x}, \mathbf{z})$, and using feature map $\Phi(\mathbf{x})$. Explain differences in runtime and sample complexity.  
  *רשום/רשמי את נוסחת החיזוי של Kernel SVM בעזרת $K$ ובעזרת $\Phi$. הסבר/י את ההבדל בסיבוכיות.*
* **b.** Consider a 2-layer network: Layer 1 has $n$ nodes with outputs $h_i(\mathbf{x}) = y_i K(\mathbf{x}, \mathbf{x}_i)$. Layer 2 has weights $\alpha_1, \dots, \alpha_n$ and sign activation. Show this network is equivalent to a Kernel SVM.  
  *הוכח/י שרשת 2-שכבות הזו שקולה למסווג Kernel SVM.*
* **c.** Choose a kernel $K$ and hyperparameter configuration for $\alpha_1, \dots, \alpha_n$ such that this network becomes equivalent to a 1-Nearest-Neighbor (1-NN) classifier.  
  *בחר/י גרעין $K$ ומקבלים $\alpha_i$ כך שהרשת תהיה שקולה בדיוק לאלגוריתם 1-NN.*

---

### 📌 Question 5: Deep Networks & Multi-Output Architecture
### שאלה 5: רשתות עמוקות וארכיטקטורה בעלת מספר פלטים

* **a.** Consider an MLP with 1 hidden layer of $k$ units and linear output neuron. Write the mathematical formula of the forward pass for input $\mathbf{x} \in \mathbb{R}^d$ with weights $W_1 \in \mathbb{R}^{d \times k}$ and $\mathbf{w}_2 \in \mathbb{R}^k$.  
  *רשום/רשמי את נוסחאות המעבר קדימה (Forward Pass) של רשת MLP בעלת שכבה נסתרת אחת.*
* **b.** For MSE loss, compute derivatives $\frac{\partial l}{\partial \mathbf{w}_2}$ and $\frac{\partial l}{\partial W_1}$ using chain rule, and write the SGD weight update steps.  
  *חשב/י את הנגזרות לפי כלל השרשרת עבור הפסד MSE ורשום/רשמי את עדכוני ה-SGD.*
* **c.** For input $\mathbf{x}$, there are two targets: continuous $y_1 \in \mathbb{R}$ and binary label $y_2 \in \{+1, -1\}$. Describe an MLP architecture with 1 shared hidden layer and 2 output neurons, state the loss function, and describe the weight update step.  
  *תאר/י ארכיטקטורת רשת מרובת-משימות (Multi-Task) בעלת שכבה נסתרת משותפת ושני פלטים (רגרסיה וסיווג).*

---

### 📌 Question 6: Unsupervised Learning & PCA
### שאלה 6: למידה לא מפוקחת ו-PCA

* **a.** Write pseudo-code for reducing data dimensionality with Principal Component Analysis (PCA) from $\mathbb{R}^d$ to $\mathbb{R}^r$ ($d > r > 0$).  
  *רשום/רשמי פסאודו-קוד להורדת ממדיות בעזרת PCA.*
* **b.** Let $V \in \mathbb{R}^{d \times r}$ be a matrix of orthonormal basis vectors ($V^T V = I_r$). Prove that projection vector $\mathbf{z} \in \mathbb{R}^r$ minimizing reconstruction error $\|V\mathbf{z} - \mathbf{x}\|^2$ satisfies $\mathbf{z} = V^T \mathbf{x}$.  
  *הוכח/י כי וקטור ההטלה הממזער את שגיאת השחזור הוא $\mathbf{z} = V^T \mathbf{x}$.*
* **c.** Given labeled dataset $(\mathbf{x}_i, y_i)$, a student suggests applying PCA reduction from $d$ to $r$ first, then training a linear classifier on projected features. Explain when this improves vs. hurts performance. Illustrate for $d=2, r=1$.  
  *הסבר/י מתי הפחתת ממדיות באמצעות PCA לפני סיווג מועילה ומתי היא פוגעת בביצועים. הדגם/הדגימי עבור $d=2, r=1$.*
