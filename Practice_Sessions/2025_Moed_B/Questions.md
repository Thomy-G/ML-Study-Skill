# 📝 2025 Moed B — Complete Exam Practice Session | מועד ב׳ 2025 — אימון למבחן מלא

**Course / קורס:** Bar-Ilan University Machine Learning 89-511 / למידת מכונה, אוניברסיטת בר-אילן  
**Instructors / מרצים:** Prof. Gal Chechik & Prof. Joseph Keshet  

---

## 🔹 Part A: True/False Questions (21 Points Total)
## חלק א': שאלות נכון / לא נכון (21 נקודות)

### 📌 Question 1 (12 Points / 12 נקודות)
Mark whether each statement below is **Correct (True)** or **Incorrect (False)** / סמן/י נכון או לא נכון:

* **a.** Dropout reduces overfitting during neural network training.  
  *שכבת הניכוי (Dropout) מפחיתה את התאמת היתר (Overfitting) בזמן אימון רשתות ניורונים.*
* **b.** Residual connections in ResNets improve the inference stage in very deep networks due to the vanishing gradient problem.  
  *חיבורי שארית (Residual connections) ברשתות ResNet משפרים את שלב ההסקה (Inference) ברשתות עמוקות מאוד בשל בעיית הגרדיאנט הנעלם.*
* **c.** Parameter sharing in convolutional networks makes them more robust to translations of objects in the image.  
  *שיתוף פרמטרים (Parameter sharing) ברשתות קונבולוציה הופך אותן לעמידות יותר להזזות (Translations) של אובייקטים בתמונה.*
* **d.** In cases where $p \gg n$ (many more parameters than data), increasing the complexity of the model will usually lead to a smaller test error.  
  *במקרים שבהם $p \gg n$ (הרבה יותר פרמטרים מנתונים), הגדלת סיבוכיות המודל תוביל בדרך כלל לשגיאת מבחן קטנה יותר.*

---

### 📌 Question 2 (9 Points / 9 נקודות)
Mark whether each statement below is **Correct (True)** or **Incorrect (False)** / סמן/י נכון או לא נכון:

* **a.** F1-score is the harmonic mean of recall and precision.  
  *מדד F1-score הוא הממוצע ההרמוני של ה-Recall וה-Precision.*
* **b.** The reconstruction objective function in a Denoising Autoencoder (DAE) assumes the corruption process is governed by Gaussian noise.  
  *פונקציית המטרה לשחזור באוטואנקודר רועש (DAE) מניחה שתהליך ההשחתה נשלט ע״י רעש גאוסי.*
* **c.** A hypothesis class with a VC-dimension of 0 can contain more than one hypothesis.  
  *מחלקת היפותזות עם ממד VC השווה ל-0 יכולה להכיל יותר מהיפותזה אחת.*

---

## 🔹 Part B: Open Questions (Choose 2 out of 3, 40 Points Each)
## חלק ב': שאלות פתוחות (בחר/י 2 מתוך 3, 40 נקודות לשאלה)

---

### 📌 Question 3: Linear Regression Models
### שאלה 3: מודלי רגרסיה לינארית

* **a.** Formally define the linear regression problem with a Mean Square Error (MSE) objective function for $d$-dimensional input. State the condition under which linear regression has a unique optimal solution. Under this condition, provide the closed-form optimal solution.  
  *הגדר/י פורמלית את בעיית הרגרסיה הלינארית עם פונקציית הפסד MSE עבור קלט $d$-ממדי. ציין/י את התנאי לקיום פתרון אופטימלי יחיד ורשום/רשמי את הפתרון במבנה סגור.*
* **b.** Given input pairs $\mathbf{x}_i = (x_{i,1}, x_{i,2}) \in \mathbb{R}^2$ and outputs $y_i \in \mathbb{R}$. We propose model $\hat{y}_i = w_1^2 x_{i,1} + w_2^2 x_{i,2}$. Find the optimal values for $w_1$ (for MSE) given $w_2$ is fixed.  
  *נתון מודל $\hat{y}_i = w_1^2 x_{i,1} + w_2^2 x_{i,2}$. מצא/י את הערכים האופטימליים עבור $w_1$ כאשר $w_2$ קבוע.*
* **c.** Let $x_i \in \mathbb{R}$ and assume access to unlimited data. Model A: $y = w^2 x, w \in \mathbb{R}$ vs. Model B: $y = w x, w \in \mathbb{R}$. Under what data conditions would Model A be preferred? Under what conditions would Model B be preferred?  
  *באילו תנאי נתונים מודל א' עדיף ובאילו מודל ב' עדיף?*
* **d.** Replace Model A with Model C: $y = w_1^2 x + w_2 x$ for $w_1, w_2 \in \mathbb{R}$. How many optimal parameter sets exist for Model B and Model C?  
  *כמה קבוצות פרמטרים אופטימליות קיימות עבור מודל ב' ומודל ג'?*
* **e.** Solutions for models B and C are found using gradient descent with $0$ MSE training error. How likely is model C to achieve better generalization error than model B?  
  *בהינתן אימון ע״י גרדיאנט דסנט, האם מודל ג' צפוי להשיג שגיאת הכללה טובה יותר ממודל ב'?*

---

### 📌 Question 4: Binary Classification & Soft-SVM
### שאלה 4: סיווג בינארי ו-Soft-SVM

Given $n$ points $\mathbf{x}_1, \dots, \mathbf{x}_n \in \mathbb{R}^d$ with labels $y_1, \dots, y_n \in \{+1, -1\}$. We train linear classifier $\mathbf{x} \mapsto \text{sign}(\mathbf{w}^T \mathbf{x})$.

* **a.** Formally define Soft-SVM (without bias) as an optimization problem. Explain each variable and the hyperparameter $C$.  
  *הגדר/י פורמלית את בעיית Soft-SVM ללא Bias כבעיית אופטימיזציה. הסבר/י כל משתנה ופרמטר $C$.*
* **b.** Assume examples are orthonormal ($\mathbf{x}_i^T \mathbf{x}_j = 1$ if $i=j$ and $0$ otherwise). Prove training set is linearly separable, specifically showing $\mathbf{w} = \sum_{j: y_j=+1} \mathbf{x}_j - \sum_{j: y_j=-1} \mathbf{x}_j$ separates data.  
  *הנח/י שהדוגמאות אורתונורמליות. הוכיחי/הוכח שהנתונים ניתנים להפרדה לינארית ע״י וקטור המשקולות המוגדר.*
* **c.** Assume data is linearly separable. Let $\mathbf{w}^*$ be the Hard-SVM solution with $\|\mathbf{w}^*\|^2 > 0$. Prove Soft-SVM solution perfectly separates data when $C = \|\mathbf{w}^*\|^2$ (achieving 0 training error).  
  *הוכח/י שפתרון Soft-SVM משיג שגיאת אימון אפס כאשר $C = \|\mathbf{w}^*\|^2$.*

---

### 📌 Question 5: Non-linear Perceptrons & Kernel Trick
### שאלה 5: פרספטרון לא-לינארי והטריק הגרעיני

* **a.** Let $\mathbf{x}, \mathbf{z} \in \mathbb{R}^2$. Prove $K(\mathbf{x}, \mathbf{z}) = (1 + \mathbf{x}^T \mathbf{z})^2$ is a kernel by showing equivalence to feature map inner product $\phi(\mathbf{x})^T \phi(\mathbf{z})$. How many FLOPs are saved by computing kernel vs. explicit feature dot product?  
  *הוכח/י כי פונקציית הגרעין תקפה וחשב/י חיסכון בפעולות חישוב.*
* **b.** Write Perceptron algorithm pseudocode. Also write Perceptron as SGD optimization for Hinge loss (zero-margin).  
  *רשום/רשמי פסאודו-קוד ל-Perceptron ונסח/י אותו כ-SGD על Hinge Loss.*
* **c.** Map examples to $\phi(\mathbf{x}) \in \mathbb{R}^D$ where $D \gg d$. Update pseudocode for $\phi(\mathbf{x}_i)$ and show learned weights $\mathbf{w} = \sum_{i=1}^n \alpha_i y_i \phi(\mathbf{x}_i)$.  
  *עדכן/י את הפסאודו-קוד במרחב התכונות והוכח/י ייצוג דואלי.*
* **d.** Define kernel $K(\mathbf{x}, \mathbf{x}') = \phi(\mathbf{x})^T \phi(\mathbf{x}')$. Update pseudocode using kernel trick to learn parameters $\alpha_1, \dots, \alpha_n$.  
  *רשום/רשמי פסאודו-קוד מלא ל-Kernel Perceptron.*
