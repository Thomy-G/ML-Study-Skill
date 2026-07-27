# ✍️ 2023 Moed B — Answers Template | דף תשובות — מועד ב׳ 2023

**Student Name / שם הסטודנט/ית:**  
**Date / תאריך:**  

---

## 🔹 Part A: True/False Questions / חלק א': שאלות נכון/לא נכון

### Question 1: Overfitting Reduction / שאלה 1
* **a.** [Yes / No] — *Reasoning / נימוק:*
* **b.** [Yes / No] — *Reasoning / נימוק:*
* **c.** [Yes / No] — *Reasoning / נימוק:*
* **d.** [Yes / No] — *Reasoning / נימוק:*

### Question 2: Core Principles / שאלה 2
* **a.** [Correct / Wrong] — *Reasoning / נימוק:*
* **b.** [Correct / Wrong] — *Reasoning / נימוק:*
* **c.** [Correct / Wrong] — *Reasoning / נימוק:*
* **d.** [Correct / Wrong] — *Reasoning / נימוק:*

---

## 🔹 Part B: Open Questions / חלק ב': שאלות פתוחות

### Question 3: Linear Classifiers & MAP Logistic Regression / שאלה 3
* **a.** Binary logistic regression formulation, loss function, & rationale:
For logistic regression optimizing the likelihood is the same as minimizing the log likelihood, therefore we write it as such:
$$
\displaylines{
let: \\
y_{i} \in \Set{ -1,1 } \text{ be a label for the true data}\\
\hat{y} \in \Set{ -1,1 } \text{ predictions made by the model}\\
x_{i}\in \mathbb{R}^{d} \text{ the data that comes with the label as } (x_{i}, y_{i})\\
w \text{The weight vector}\\
\text{The negative log likelyhood is equivalent to the probability of getting a wrong answer}\\
\mathcal{L}_{NLL} = -\ln(L(x))
}
$$

* **b.** SGD derivation, gradient simplification, & learning rate selection:
 $$
 \displaylines{
 \left[ \sum_{i=1}^n \ln \left( 1 + e^{-y_i \mathbf{w}^T \mathbf{x}_i} \right) \right]^{'} = \sum_{i=1}^n \left[  \ln \left( 1 + e^{-y_i \mathbf{w}^T \mathbf{x}_i} \right) \right]^{'}  = \frac{1}{1 + e^{-y_i \mathbf{w}^T \mathbf{x}_i}} \cdot \ln \left( (1 + e^{-y_i \mathbf{w}^T \mathbf{x}_i})^{'} \right) =\\
 \frac{1}{1 + e^{-y_i \mathbf{w}^T \mathbf{x}_i}} \cdot \ln \left( (1)' + (e^{-y_i \mathbf{w}^T \mathbf{x}_i})^{'} \right) = \frac{1}{1 + e^{-y_i \mathbf{w}^T \mathbf{x}_i}} \cdot \ln \left(  -y_i x_{i}e^{-y_i \mathbf{w}^T \mathbf{x}_i} \right) =\\
 
 }
 $$
* **c.** MAP formulation with Gaussian prior $\mathcal{N}(0,1)$, new SGD update, & $L_2$ regularization effect:
  ```text
  
  ```

---

### Question 4: Large Margin & 1-NN Network Equivalence / שאלה 4
* **a.** Kernel SVM prediction formula via $K$ vs $\Phi$ & complexity analysis:
  ```text
  
  ```
* **b.** 2-layer network structure & Kernel SVM equivalence proof:
  ```text
  
  ```
* **c.** RBF kernel parameter selection for 1-NN classification equivalence:
  ```text
  
  ```

---

### Question 5: Deep Networks & Multi-Output Architecture / שאלה 5
* **a.** 1-hidden-layer MLP forward pass mathematical formula:
  ```text
  
  ```
* **b.** Chain rule derivatives $\frac{\partial l}{\partial \mathbf{w}_2}, \frac{\partial l}{\partial W_1}$ & SGD updates:
  ```text
  
  ```
* **c.** Multi-task architecture (regression $y_1$ + classification $y_2$), joint loss, & backprop update:
  ```text
  
  ```

---

### Question 6: Unsupervised Learning & PCA / שאלה 6
* **a.** PCA dimensionality reduction pseudocode ($d \to r$):
  ```python
  
  ```
* **b.** Reconstruction error minimization proof $\mathbf{z} = V^T \mathbf{x}$:
  ```text
  
  ```
* **c.** PCA pre-processing trade-offs for classification & $d=2, r=1$ geometric counter-example:
  ```text
  
  ```
