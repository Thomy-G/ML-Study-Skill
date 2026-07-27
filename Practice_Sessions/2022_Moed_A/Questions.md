# 📝 2022 Moed A — Exam Practice Session | מועד א׳ 2022 — אימון למבחן

**Course / קורס:** Bar-Ilan University Machine Learning 89-511 / למידת מכונה, אוניברסיטת בר-אילן  
**Instructors / מרצים:** Prof. Gal Chechik & Prof. Joseph Keshet  

---

## 📌 Question 1: $k$-NN Distance Metric & Metric Space Properties
## שאלה 1: פונקציית מרחק ב-$k$-NN ותכונות מרחב מטרי

### 🇮🇱 עברית
צביה אוספת נתונים שבהם כל דוגמה מיוצגת על ידי שני תכונות $\mathbf{x} \in \mathbb{R}^2$. היא מסווגת דוגמאות חדשות בעזרת מסווג $k$-NN המשתמש בפונקציית המרחק המותאמת אישית הבאה:
$$d(\mathbf{x}, \mathbf{y}) = |x_1 - y_1| + |x_2 - y_2|^2$$

**סעיף א':** האם $d(\mathbf{x}, \mathbf{y})$ היא פונקציית מרחק תקפה במרחב מטרי (Metric Space)? הוכח או הפרך.

### 🇬🇧 English
Tzvia collected dataset samples where each sample is represented by 2 features $\mathbf{x} \in \mathbb{R}^2$. She classifies new samples using a $k$-NN classifier with the following custom distance function:
$$d(\mathbf{x}, \mathbf{y}) = |x_1 - y_1| + |x_2 - y_2|^2$$

**Part A:** Is $d(\mathbf{x}, \mathbf{y})$ a valid metric space distance function? Prove or disprove.

---

## 📌 Question 2: CNN Architecture, Output Spatial Shapes & Parameter Counting
## שאלה 2: ארכיטקטורת CNN, מוקד פלט וחישוב פרמטרים

### 🇮🇱 עברית
יוסי מאמן רשת קונבולוציה (ConvNet) על אוסף הנתונים CIFAR-10 (תמונות RGB בגודל $3 \times 32 \times 32$, 10 מחלקות).  
מבנה הרשת:
* **שכבה א' (Layer A):** `Conv2D(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)` $\to$ `ReLU` $\to$ `MaxPool2D(kernel_size=2, stride=2)`
* **שכבה ב' (Layer B):** `Conv2D(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1)` $\to$ `ReLU` $\to$ `MaxPool2D(kernel_size=2, stride=2)`
* **שכבה ג' (Layer C):** `Flatten` $\to$ `FullyConnected(out_features=10)`

1. **סעיף א':** חבר/י את הממדים המרחביים של הטנסורים בפלט של שכבות A, B ו-C.
2. **סעיף ב':** חשב/י את מספר הפרמטרים הלמדניים (משקולות + הטיות / Biases) בכל שכבה בנפרד, ואת סך הכל הפרמטרים ברשת.

### 🇬🇧 English
Yossi trains a ConvNet on CIFAR-10 ($3 \times 32 \times 32$ RGB images, 10 classes).  
Architecture layout:
* **Layer A:** `Conv2D(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)` $\to$ `ReLU` $\to$ `MaxPool2D(kernel_size=2, stride=2)`
* **Layer B:** `Conv2D(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1)` $\to$ `ReLU` $\to$ `MaxPool2D(kernel_size=2, stride=2)`
* **Layer C:** `Flatten` $\to$ `FullyConnected(out_features=10)`

1. **Part A:** Calculate input and output spatial tensor dimensions for layers A, B, and C.
2. **Part B:** Calculate parameter counting (weights + biases) for each layer individually, and the total network parameters.

---

## 📌 Question 3: Perceptron with Margin vs. Soft-Margin SVM
## שאלה 3: פרספטרון עם שוליים מול Soft-Margin SVM

### 🇮🇱 עברית
לגל יש נקודות נתונים $\mathbf{x} \in \mathbb{R}^2$ (בריאים מול חולים) **שאינם ניתנים להפרדה לינארית**.

1. **סעיף א':** מדוע אימון פרספטרון עם שוליים (Margin Perceptron) ללא הטיה (Bias) אינו מתכנס על נתונים אלו?
2. **סעיף ב':** האם אלגוריתם Soft-Margin SVM יתכנס? השווה בין שתי השיטות והסבר את מנגנון משתני הרפיה (Slack Variables $\xi_i$).

### 🇬🇧 English
Gal has dataset points $\mathbf{x} \in \mathbb{R}^2$ (healthy vs. sick) that are **not linearly separable**.

1. **Part A:** Why does training a Margin Perceptron without bias fail to converge?
2. **Part B:** Would Soft-Margin SVM converge? Compare both methods and explain the slack variable $\xi_i$ mechanism.
