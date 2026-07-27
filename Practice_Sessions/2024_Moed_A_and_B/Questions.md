# 📝 2024 Moed A & B — Exam Practice Session | מועד א׳ וב׳ 2024 — אימון למבחן

**Course / קורס:** Bar-Ilan University Machine Learning 89-511 / למידת מכונה, אוניברסיטת בר-אילן  
**Instructors / מרצים:** Prof. Gal Chechik & Prof. Joseph Keshet  

---

## 📌 Question 1: Linear Autoencoder & PCA Subspace Equivalence Proof
## שאלה 1: אוטואנקודר לינארי ושקילות מרחבית ל-PCA

### 🇮🇱 עברית
נתון אוסף נתונים ממורכז $X \in \mathbb{R}^{n \times d}$. נתבונן באוטואנקודר לינארי בעל ממד מצר (Bottleneck Dimension) $r < d$, המוגדר ע״י מטריצת מקודד $W_e \in \mathbb{R}^{r \times d}$ ומטריצת מפענח $W_d \in \mathbb{R}^{d \times r}$:
$$\hat{X} = X W_e^T W_d^T$$

הוכח/י כי תחת שגיאת שחזור של ריבועים פחותים (MSE Frobenius Norm $\|X - \hat{X}\|_F^2$), המרחב הלינארי מדרגה $r$ הנפרס ע״י עמודות $W_d$ שקול מתמטית למרחב הנפרס ע״י $r$ הרכיבים הראשונים של ניתוח רכיבים ראשיים (PCA).

### 🇬🇧 English
Given centered data matrix $X \in \mathbb{R}^{n \times d}$. Consider a linear autoencoder with bottleneck dimension $r < d$, parameterized by encoder matrix $W_e \in \mathbb{R}^{r \times d}$ and decoder matrix $W_d \in \mathbb{R}^{d \times r}$:
$$\hat{X} = X W_e^T W_d^T$$
Prove that under Mean Squared Error (MSE) reconstruction loss $\|X - \hat{X}\|_F^2$, the learned $r$-dimensional subspace spanned by the columns of $W_d$ is mathematically equivalent to the subspace spanned by the top $r$ Principal Components of PCA.

---

## 📌 Question 2: PCA Projection Reconstruction Error Minimization
## שאלה 2: מזעור שגיאת שחזור בהטלת PCA

### 🇮🇱 עברית
תהי $V \in \mathbb{R}^{d \times r}$ מטריצה אשר עמודותיה הן וקטורי בסיס אורתונורמליים ($V^T V = I_r$).  
הוכח/י כי עבור כל דוגמה $\mathbf{x} \in \mathbb{R}^d$, וקטור ההטלה $\mathbf{z} \in \mathbb{R}^r$ שממזער את שגיאת השחזור $\|V\mathbf{z} - \mathbf{x}\|^2$ נתון ע״י:
$$\mathbf{z}^* = V^T \mathbf{x}$$

### 🇬🇧 English
Let $V \in \mathbb{R}^{d \times r}$ be a matrix whose $r$ columns are orthonormal basis vectors of a linear subspace ($V^T V = I_r$). Prove that for any sample $\mathbf{x} \in \mathbb{R}^d$, the projection vector $\mathbf{z} \in \mathbb{R}^r$ minimizing the reconstruction error $\|V\mathbf{z} - \mathbf{x}\|^2$ is given by:
$$\mathbf{z}^* = V^T \mathbf{x}$$

---

## 📌 Question 3: Naive Bayes vs. Table Lookup & Curse of Dimensionality
## שאלה 3: בייס נאיבי מול טבלת שכיחויות וקללת הממדיות

### 🇮🇱 עברית
חיזוי סיכויי ניצחון במשחק טניס לפי התכונות הבאות: רמת יריב $O \in \{1..10\}$, משטח $C \in \{\text{דשא, חימר, קשה}\}$, סוג $G \in \{\text{זוגות, יחידים}\}$, מזג אוויר $W \in \{\text{חם, קר}\}$, זמן $T \in \{\text{יום, לילה}\}$.

1. **סעיף א':** מצא/י את גודל טבלת ההסתברויות השלמה (Full Lookup Table Size). הסבר/י מדוע אומדן הסתברויות מתדרים נתקל ב"קללת הממדיות".
2. **סעיף ב':** הסבר/י כיצד הנחת העצמאות המותנית של מסווג בייס נאיבי (Naive Bayes) מפחיתה את מספר הפרמטרים ממעריכי $O(k^d)$ ללינארי $O(d \cdot k)$.

### 🇬🇧 English
Predict tennis win probability given features: Opponent Skill $O \in \{1..10\}$, Surface $C \in \{\text{Grass, Clay, Hard}\}$, Type $G \in \{\text{Doubles, Singles}\}$, Weather $W \in \{\text{Hot, Cold}\}$, Time $T \in \{\text{Day, Night}\}$.

1. **Part A:** Calculate the total configurations for a full table lookup. Explain why sample frequency estimation suffers from the Curse of Dimensionality.
2. **Part B:** Explain how Naive Bayes conditional independence assumption reduces parameters from exponential $O(k^d)$ to linear $O(d \cdot k)$.

---

## 📌 Question 4: Maximum-Margin Ranking SVM Formulation
## שאלה 4: ניסוח אלגוריתם דירוג Ranking SVM עם שוליים מרביים

### 🇮🇱 עברית
נתונים זוגות דוגמאות $(x_i^{(1)}, x_i^{(2)})$ שבהם $x_i^{(1)}$ צריך להיות מדורג גבוה יותר מ-$x_i^{(2)}$. ניסח/י את הבעיה כבעיית אופטימיזציה מוכללת של SVM עם שוליים מרביים (Max-Margin Ranking SVM) הכוללת משתני הרפיה $\xi_i \ge 0$.

### 🇬🇧 English
Given pairs $(x_i^{(1)}, x_i^{(2)})$ where $x_i^{(1)}$ must rank higher than $x_i^{(2)}$. Formulate max-margin ranking as a constrained optimization problem incorporating slack variables $\xi_i \ge 0$.
