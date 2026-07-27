# 📝 2023 Moed A — Complete Exam Practice Session | מועד א׳ 2023 — אימון למבחן מלא

**Course / קורס:** Bar-Ilan University Machine Learning 89-511 / למידת מכונה, אוניברסיטת בר-אילן  
**Instructors / מרצים:** Prof. Gal Chechik & Prof. Joseph Keshet  

---

## 🔹 Part A: True/False & Yes/No Questions (25 Points Total)
## חלק א': שאלות נכון / לא נכון (25 נקודות)

### 📌 Question 1 (12.5 Points / 12.5 נקודות) — Generalization Gap
After training a deep network model, you observe that the test error is much larger than the training error. Which of the following is expected to reduce this generalization gap (regardless of other effects)?
*לאחר אימון מודל רשת עמוקה, נצפה כי שגיאת המבחן גדולה בהרבה משגיאת האימון. אילו מהפעולות הבאות צפויה לצמצם את פער ההכללה?*

* **a.** Reduce the number of parameters in the model, and retrain (Yes/No)  
  *להפחית את מספר הפרמטרים במודל ולאמן מחדש.*
* **b.** Add more expressivity to the model, and retrain (Yes/No)  
  *להוסיף כושר ביטוי (Expressivity) למודל ולאמן מחדש.*
* **c.** Retrain the model, but this time using the test data (Yes/No)  
  *לאמן מחדש את המודל, אך הפעם תוך שימוש בנתוני המבחן.*
* **d.** Retrain the model using both the training and test data (Yes/No)  
  *לאמן מחדש את המודל תוך שימוש בנתוני האימון והמבחן יחד.*

---

### 📌 Question 2 (12.5 Points / 12.5 נקודות) — Core Principles
Mark which of the following statements is correct (Correct/Wrong) / סמן/י נכון או שגוי:

* **a.** Gradient descent is guaranteed to converge when optimizing a strongly convex function.  
  *אלגוריתם Gradient Descent מובטח להתכנס בעת אופטימיזציה של פונקציה קמורה חזקה.*
* **b.** A KNN classifier can solve the parity problem given enough samples.  
  *מסווג KNN יכול לפתור את בעיית הזוגיות (Parity Problem) בהינתן מספיק דוגמאות.*
* **c.** Training very deep MLPs is hard because the gradients tend to vanish or explode.  
  *אימון רשתות MLP עמוקות מאוד הוא קשה מכיוון שהגרדיאנטים נוטים להיעלם או להתפוצץ.*
* **d.** RNN architectures can handle input sequences of different sizes.  
  *ארכיטקטורות RNN יכולות להתמודד עם סדרות קלט בגדלים שונים.*

---

## 🔹 Part B: Open Questions (Choose 3 out of 4, 25 Points Each)
## חלק ב': שאלות פתוחות (בחר/י 3 מתוך 4, 25 נקודות לשאלה)

---

### 📌 Question 1: Linear Classifiers
### שאלה 1: מסווגים לינאריים

* **a.** Write pseudo-code for the Perceptron algorithm described in class. What properties should the data obey to guarantee that the algorithm converges?  
  *רשום/רשמי פסאודו-קוד לאלגוריתם ה-Perceptron. אילו תכונות חייבות להתקיים בנתונים כדי להבטיח התכנסות?*
* **b.** Write an algorithm for training a linear separator using SGD for a (margin-based) Hinge loss. Write explicitly the gradient of the loss. Discuss the differences between this algorithm and the Perceptron from (a).  
  *רשום/רשמי אלגוריתם לאימון מפריד לינארי בעזרת SGD עבור פונקציית הפסד Hinge Loss. רשום/רשמי מפורשות את הנגזרת ודון/דוני בהבדלים אל מול ה-Perceptron.*
* **c.** How would your algorithm in (b) change if you use momentum? Write pseudo code for the new algorithm.  
  *כיצד ישתנה האלגוריתם מסעיף ב' בשימוש ב-Momentum? רשום/רשמי פסאודו-קוד חדש.*

---

### 📌 Question 2: Large Margin Classifiers & Sphere Classifier
### שאלה 2: מסווגי שוליים רחבים ומסווג כדורי

* **a.** Write the optimization problem for linear SVM with slack variables. Be very careful to include all constraints, explain the parameters and what values are being optimized over.  
  *רשום/רשמי את בעיית האופטימיזציה של Soft-SVM. הסבר/י את המשתנים והאילוצים.*
* **b.** For a binary classification problem, define the margin $\rho$ of a given linear classifier $\mathbf{w}$. Show that the quadratic SVM optimization problem is equivalent to maximizing the margin of a linear classifier (assume linearly separable data).  
  *הגדר/י את השוליים $\rho$ והוכח/י שקילות בין מקסום השוליים למזעור $\|\mathbf{w}\|^2$.*
* **c.** Consider the "sphere-classifier" problem: Given centroid $\mathbf{a} \in \mathbb{R}^d$ and radius $r \in \mathbb{R}^+$, predict $\hat{y} = +1$ if $\|\mathbf{x} - \mathbf{a}\| < r$ and $\hat{y} = -1$ otherwise. Given samples $\mathbf{x}_1, \dots, \mathbf{x}_n \in \mathbb{R}^d$ with labels $y_1, \dots, y_n \in \{+1, -1\}$, write an SVM-like constrained optimization problem using slack variables to find the optimal center $\mathbf{a}$ and radius $r$.  
  *נסח/י בעיית אופטימיזציה בסגנון SVM למציאת המרכז $\mathbf{a}$ והרדיוס $r$ של מסווג כדורי עם משתני הרפיה.*

---

### 📌 Question 3: Non-Linear Classifiers & Deep Networks
### שאלה 3: מסווגים לא-לינאריים ורשתות עמוקות

* **a.** Describe an architecture of a convolutional layer in a ConvNet designed for RGB image classification. Describe in detail the dimension of filters, how they are applied to the input, and how filters are combined with ReLUs.  
  *תאר/י ארכיטקטורה של שכבת קונבולוציה לסיווג תמונות RGB (ממדי מסננים, הפעלה, ו-ReLU).*
* **b.** Show that for a multi-class classifier with KL-divergence loss $D_{KL}(t \| \hat{y})$, when the ground-truth distribution $t$ is a one-hot vector, minimizing Cross-Entropy loss is equivalent to maximizing the target class likelihood.  
  *הוכח/י כי עבור תווית One-Hot, מזעור KL-Divergence שקול למקסום הסבירות (Likelihood) של המחלקה הנכונה.*
* **c.** Build a deep architecture that increases photo resolution from $100 \times 100$ RGB to $200 \times 200$.  
  * **(i)** Describe the set of layers, filter dimensions, and upsampling mechanism.  
  * **(ii)** Describe how you would train the model: loss function and how to acquire a large-scale ground truth dataset.

---

### 📌 Question 4: Unsupervised Learning & $K$-Means
### שאלה 4: למידה לא מפוקחת ו-$K$-Means

* **a.** Write pseudo-code for the $K$-Means algorithm for clustering $n$ samples $\mathbf{x}_1, \dots, \mathbf{x}_n \in \mathbb{R}^d$ into $k$ clusters with centroids $c_1, \dots, c_k \in \mathbb{R}^d$.  
  *רשום/רשמי פסאודו-קוד לאלגוריתם $K$-Means.*
* **b.** Prove that $K$-Means converges by showing that each update step reduces the total $L_2$ distortion $\sum_{i=1}^n \|\mathbf{x}_i - c_{z_i}\|_2^2$.  
  *הוכח/י התכנסות של $K$-Means ע״י הרחבת פונקציית העיוות.*
* **c.** How would the algorithm change if we wish to minimize $L_1$ distance $\|x - c\|_1 = \sum_{j=1}^d |x_j - c_j|$? Write the new pseudo-code and component update.  
  *כיצד ישתנה האלגוריתם במיזעור מרחק $L_1$ ($K$-Medians)?*
