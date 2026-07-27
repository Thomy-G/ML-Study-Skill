# ✍️ 2025 Moed A — Answers Template | דף תשובות — מועד א׳ 2025

**Student Name / שם הסטודנט/ית:**  
**Date / תאריך:**  

---

## 🔹 Part A: True/False Questions / חלק א': שאלות נכון/לא נכון

### Question 1 / שאלה 1
* **a.** [True / False] — *Reasoning / נימוק:*
  True, Stochastic gradient descent uses less data to calculate the gradient so it performs less calculations
<div align="right">
<table style="border: 1px solid #ddd; border-radius: 4px; background: rgba(130, 130, 130, 0.07); padding: 8px; font-size: 13px; font-family: system-ui; width: fit-content; text-align: left;">
  <tr><td><strong>Part 1a Score:</strong></td><td><strong style="color: #2e7d32;">3 / 3 Points (100%)</strong></td></tr>
  <tr><td colspan="2" style="border-top: 1px dotted #ccc; padding-top: 4px; color: var(--text-muted);">Correct! SGD computes gradients on m << n samples per update, reducing per-iteration FLOPs.</td></tr>
</table>
</div>

* **b.** [True / False] — *Reasoning / נימוק:*
  Not necessarily, adam can find a minimum faster which if anything makes overfitting worse, overiftting should be addressed at the learning algorithm level, not the optimization step layer
<div align="right">
<table style="border: 1px solid #ddd; border-radius: 4px; background: rgba(130, 130, 130, 0.07); padding: 8px; font-size: 13px; font-family: system-ui; width: fit-content; text-align: left;">
  <tr><td><strong>Part 1b Score:</strong></td><td><strong style="color: #2e7d32;">3 / 3 Points (100%)</strong></td></tr>
  <tr><td colspan="2" style="border-top: 1px dotted #ccc; padding-top: 4px; color: var(--text-muted);">Correct! Adam accelerates optimization convergence but does not inherently regularize overfitting.</td></tr>
</table>
</div>

* **c.** [True / False] — *Reasoning / נימוק:*
  True, BatchNorm allows higher learning rates that allow deep networks to converge faster
<div align="right">
<table style="border: 1px solid #ddd; border-radius: 4px; background: rgba(130, 130, 130, 0.07); padding: 8px; font-size: 13px; font-family: system-ui; width: fit-content; text-align: left;">
  <tr><td><strong>Part 1c Score:</strong></td><td><strong style="color: #2e7d32;">3 / 3 Points (100%)</strong></td></tr>
  <tr><td colspan="2" style="border-top: 1px dotted #ccc; padding-top: 4px; color: var(--text-muted);">Correct! BatchNorm reduces covariate shift and smoothes loss gradients, enabling larger learning rates.</td></tr>
</table>
</div>

* **d.** [True / False] — *Reasoning / נימוק:*
  *(Unanswered)*
  

### Question 2 / שאלה 2
* **a.** [True / False] — *Reasoning / נימוק:*
* **b.** [True / False] — *Reasoning / נימוק:*
* **c.** [True / False] — *Reasoning / נימוק:*

---

## 🔹 Part B: Open Questions / חלק ב': שאלות פתוחות

### Question 3: Logistic Regression / שאלה 3: רגרסיה לוגיסטית
* **a.** NLL loss derivation for $D_1$:
  ```text
  
  ```
* **b.** Gradient derivation using chain rule:
  ```text
  
  ```
* **c.** Monotonicity proof of $\nabla \text{Err}_{D_1}(w)$:
  ```text
  
  ```
* **d. (i)-(iii)** Interval bound proofs for $w^*$:
  ```text
  
  ```

---

### Question 4: Naïve Bayes / שאלה 4: בייס נאיבי
* **a.** Conditional log-likelihood:
  ```text
  
  ```
* **b.** Log-posterior ratio:
  ```text
  
  ```
* **c.** Linear decision boundary & rule proof:
  ```text
  
  ```

---

### Question 5: $K$-Means & EM / שאלה 5: $K$-Means ו-EM
* **a.** Full joint likelihood:
  ```text
  
  ```
* **b.** Posterior responsibilities and limit $\sigma \to 0$:
  ```text
  
  ```
* **c.** $K$-Means algorithm & E-step equivalence:
  ```text
  
  ```
* **d.** M-step equivalence & EM-$K$-Means identity proof:
  ```text
  
  ```
