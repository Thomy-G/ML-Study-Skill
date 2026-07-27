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
  True, 
<div align="right">
<table style="border: 1px solid #ddd; border-radius: 4px; background: rgba(130, 130, 130, 0.07); padding: 8px; font-size: 13px; font-family: system-ui; width: fit-content; text-align: left;">
  <tr><td><strong>Part 1d Score:</strong></td><td><strong style="color: #2e7d32;">3 / 3 Points (100%)</strong></td></tr>
  <tr><td colspan="2" style="border-top: 1px dotted #ccc; padding-top: 4px; color: var(--text-muted);">Correct! Under SGD, weight decay w_t+1 = (1-eta*lambda)w_t - eta*grad is algebraically identical to L2 regularization.</td></tr>
</table>
</div>

### Question 2 / שאלה 2
* **a.** [True / False] — *Reasoning / נימוק:* false
<div align="right">
<table style="border: 1px solid #ddd; border-radius: 4px; background: rgba(130, 130, 130, 0.07); padding: 8px; font-size: 13px; font-family: system-ui; width: fit-content; text-align: left;">
  <tr><td><strong>Part 2a Score:</strong></td><td><strong style="color: #2e7d32;">3 / 3 Points (100%)</strong></td></tr>
  <tr><td colspan="2" style="border-top: 1px dotted #ccc; padding-top: 4px; color: var(--text-muted);">Correct! VCdim = d < n means H cannot shatter *all* 2^n labelings, but 0-error hypotheses can still exist for specific datasets.</td></tr>
</table>
</div>

* **b.** [True / False] — *Reasoning / נימוק:* False
<div align="right">
<table style="border: 1px solid #ddd; border-radius: 4px; background: rgba(130, 130, 130, 0.07); padding: 8px; font-size: 13px; font-family: system-ui; width: fit-content; text-align: left;">
  <tr><td><strong>Part 2b Score:</strong></td><td><strong style="color: #c62828;">0 / 3 Points (0%)</strong></td></tr>
  <tr><td colspan="2" style="border-top: 1px dotted #ccc; padding-top: 4px; color: var(--text-muted);">Incorrect (Statement is True). For K=2, the decision boundary is the perpendicular bisector plane ||x - mu1||^2 = ||x - mu2||^2, which simplifies to 2(mu2 - mu1)^T x + ||mu1||^2 - ||mu2||^2 = 0 (a linear hyperplane!).</td></tr>
</table>
</div>

* **c.** [True / False] — *Reasoning / נימוק:* True
<div align="right">
<table style="border: 1px solid #ddd; border-radius: 4px; background: rgba(130, 130, 130, 0.07); padding: 8px; font-size: 13px; font-family: system-ui; width: fit-content; text-align: left;">
  <tr><td><strong>Part 2c Score:</strong></td><td><strong style="color: #2e7d32;">3 / 3 Points (100%)</strong></td></tr>
  <tr><td colspan="2" style="border-top: 1px dotted #ccc; padding-top: 4px; color: var(--text-muted);">Correct! BatchNorm has 2 learnable parameters per feature: scale gamma and shift beta.</td></tr>
</table>
</div>

---

## 📊 Part A Score Tally

| Question  | Part       | User Answer | Correct Answer | Score  | Max Points | Feedback                                                                                                      |
| :-------- | :--------- | :---------- | :------------- | :----- | :--------- | :------------------------------------------------------------------------------------------------------------ |
| **Q1**    | 1a         | True        | True           | 3      | 3          | Correct! SGD calculates gradients on m << n samples.                                                          |
| **Q1**    | 1b         | False       | False          | 3      | 3          | Correct! Adam accelerates optimization, doesn't regularize.                                                   |
| **Q1**    | 1c         | True        | True           | 3      | 3          | Correct! BatchNorm enables higher learning rates.                                                             |
| **Q1**    | 1d         | True        | True           | 3      | 3          | Correct! SGD Weight Decay == L2 Loss Regularization.                                                          |
| **Q2**    | 2a         | False       | False          | 3      | 3          | Correct! VCdim < n doesn't rule out 0-error hypotheses.                                                       |
| **Q2**    | 2b         | False       | **True**       | **0**  | 3          | **Incorrect!** K=2 bisector plane is linear: $2(\mu_2 - \mu_1)^T \mathbf{x} + \|\mu_1\|^2 - \|\mu_2\|^2 = 0$. |
| **Q2**    | 2c         | True        | True           | 3      | 3          | Correct! Scale $\gamma$ and shift $\beta$ parameters.                                                         |
| **TOTAL** | **Part A** |             |                | **18** | **21**     | **Grade: 85.7% (18/21)**                                                                                      |

---

## 🔹 Part B: Open Questions / חלק ב': שאלות פתוחות

### Question 3: Logistic Regression / שאלה 3: רגרסיה לוגיסטית
* **a.** NLL loss derivation for $D_1$ ($y_i^{(1)} \in \{0, 1\}$):
For Bernoulli likelihood $P(y_i^{(1)} \mid x_i^{(1)}; w) = \sigma(w x_i^{(1)})^{y_i^{(1)}} \left( 1 - \sigma(w x_i^{(1)}) \right)^{1 - y_i^{(1)}}$:
$$\text{Err}_{D_1}(w) = -\ln \prod_{i=1}^{n_1} P(y_i^{(1)} \mid x_i^{(1)}; w) = -\sum_{i=1}^{n_1} \left[ y_i^{(1)} \ln \sigma(w x_i^{(1)}) + (1 - y_i^{(1)}) \ln \left(1 - \sigma(w x_i^{(1)})\right) \right]$$

* **b.** Gradient derivation using chain rule ($y_i^{(1)} \in \{0, 1\}$):
Using the Sigmoid derivative identity $\sigma'(z) = \sigma(z)(1 - \sigma(z))$:

1. Derivative of first term $\ln \sigma(w x_i^{(1)})$:
   $$\frac{\partial}{\partial w} \ln \sigma(w x_i^{(1)}) = \frac{\sigma'(w x_i^{(1)}) \cdot x_i^{(1)}}{\sigma(w x_i^{(1)})} = \frac{\sigma(w x_i^{(1)})(1 - \sigma(w x_i^{(1)})) x_i^{(1)}}{\sigma(w x_i^{(1)})} = (1 - \sigma(w x_i^{(1)})) x_i^{(1)}$$

2. Derivative of second term $\ln(1 - \sigma(w x_i^{(1)}))$:
   $$\frac{\partial}{\partial w} \ln(1 - \sigma(w x_i^{(1)})) = \frac{-\sigma'(w x_i^{(1)}) \cdot x_i^{(1)}}{1 - \sigma(w x_i^{(1)})} = \frac{-\sigma(w x_i^{(1)})(1 - \sigma(w x_i^{(1)})) x_i^{(1)}}{1 - \sigma(w x_i^{(1)})} = -\sigma(w x_i^{(1)}) x_i^{(1)}$$

3. Combining both terms:
   $$\nabla \text{Err}_{D_1}(w) = -\sum_{i=1}^{n_1} \left[ y_i^{(1)} (1 - \sigma(w x_i^{(1)})) x_i^{(1)} - (1 - y_i^{(1)}) \sigma(w x_i^{(1)}) x_i^{(1)} \right]$$
   $$= -\sum_{i=1}^{n_1} x_i^{(1)} \left[ y_i^{(1)} - y_i^{(1)}\sigma(w x_i^{(1)}) - \sigma(w x_i^{(1)}) + y_i^{(1)}\sigma(w x_i^{(1)}) \right]$$
   $$= -\sum_{i=1}^{n_1} x_i^{(1)} \left( y_i^{(1)} - \sigma(w x_i^{(1)}) \right) \quad \blacksquare$$
* **c.** Monotonicity proof of $\nabla \text{Err}_{D_1}(w)$:
$$
\displaylines{
w^{1}>w^{2} \implies \nabla \text{Err}_{D_1}(w^{1}) = -\sum_{i=1}^{n_1} x_i^{(1)} \left(y_i^{(1)} - \sigma(w^{1} x_i^{(1)})\right) = -\sum_{i=1}^{n_1} x_i^{(1)} \left(y_i^{(1)} - \frac{1}{1+e^{-w^{1} x_i^{(1)}}} \right) >\\ 
-\sum_{i=1}^{n_1} x_i^{(1)} \left(y_i^{(1)} - \frac{1}{1+e^{-w^{2} x_i^{(1)}}} \right) = \nabla \text{Err}_{D_1}(w^{2})\\ \\
w^{1}>w^{2} \implies e^{-w^{1} x_i^{(1)}}< e^{-w^{2} x_i^{(1)}} \implies \frac{1}{1+e^{-w^{1} x_i^{(1)}}} > \frac{1}{1+e^{-w^{2} x_i^{(1)}}}\\
\implies -\frac{1}{1+e^{-w^{1} x_i^{(1)}}} < -\frac{1}{1+e^{-w^{2} x_i^{(1)}}} \implies \\
\sum_{i=1}^{n_1} x_i^{(1)} \left(y_i^{(1)} - \frac{1}{1+e^{-w^{1} x_i^{(1)}}} \right) < \sum_{i=1}^{n_1} x_i^{(1)} \left(y_i^{(1)} - \frac{1}{1+e^{-w^{2} x_i^{(1)}}} \right)\\
- \sum_{i=1}^{n_1} x_i^{(1)} \left(y_i^{(1)} - \frac{1}{1+e^{-w^{1} x_i^{(1)}}} \right) > - \sum_{i=1}^{n_1} x_i^{(1)} \left(y_i^{(1)} - \frac{1}{1+e^{-w^{2} x_i^{(1)}}} \right)\\
}
$$
* **d. (i)-(iii)** Interval bound proofs for $w^*$:

$$
\displaylines{
\nabla Err_{D_{1}\cup D_{2}} = -\sum_{i=1}^{n_1} x_i^{(1)} \left(y_i^{(1)} - \sigma(w^{1} x_i^{(1)})\right)
}
$$
ii
$$
\displaylines{
w^{*}<\min(w^{(1)}, w^{(2)}) \implies \nabla Err_{D_{1}\cup D_{2}}(w^{*})<0\\
\nabla Err_{D_{1}\cup D_{2}}(w^{*}) = \nabla \text{Err}_{D_1}(w^*) + \nabla \text{Err}_{D_2}(w^*) =\\ -\sum_{i=1}^{n_1} x_i^{(1)} \left(y_i^{(1)} - \sigma(w^{*} x_i^{(1)})\right) -\sum_{i=1}^{n_2} x_i^{(2)} \left(y_i^{(2)} - \sigma(w^{*} x_i^{(2)})\right)\\
\text{Given that the gradient is monotonically decreasing for both}\\
w*<w^{(1)} \implies -\sum_{i=1}^{n_1} x_i^{(1)} \left(y_i^{(1)} - \sigma(w^{*} x_i^{(1)})\right) < -\sum_{i=1}^{n_1} x_i^{(1)} \left(y_i^{(1)} - \sigma(w^{(1)} x_i^{(1)})\right)\\
w^{*} < w^{(2)} \implies -\sum_{i=1}^{n_2} x_i^{(2)} \left(y_i^{(2)} - \sigma(w^{*} x_i^{(2)})\right) < -\sum_{i=1}^{n_2} x_i^{(2)} \left(y_i^{(2)} - \sigma(w^{(2)} x_i^{(2)})\right)\\
\text{Therefore}\\
w*<w^{(1)} \land w^{*} < w^{(2)} \implies  
}
$$


iii
$$
\displaylines{
w^{*}>\max(w^{(1)}, w^{(2)})\\
\text{By the same logic as part ii}\\
w^{*}>\max(w^{(1)}, w^{(2)}) \implies \nabla Err_{D_{1}\cup D_{2}}(w^{*}) = \nabla \text{Err}_{D_1}(w^*) + \nabla \text{Err}_{D_2}(w^*) >\\ \nabla \text{Err}_{D_1}(w^{(1)}) + \nabla \text{Err}_{D_2}(w^{(2)}) > 0+0\\

}
$$

---

### Question 4: Naïve Bayes / שאלה 4: בייס נאיבי
* **a.** Conditional log-likelihood:
$$
\displaylines{
\ln P(x|Y=c ) = \ln \prod_{i=1}^{d} P(x_{i}|Y=c) = \ln \prod_{i=1}^{d} \mathcal{N}(\mu_{c,j}, \sigma^{2}_{j}) = \ln \prod_{i=1}^{d} \frac{1}{\sqrt{ 2\pi\sigma^{2} }} \exp \left( - \frac{(x-\mu_{c,j})^{2}}{2\sigma^{2}} \right) = \\
\sum_{i=1}^{d} \ln \left( \frac{1}{\sqrt{ 2\pi\sigma^{2} }} \exp \left( - \frac{(x-\mu_{c,j})^{2}}{2\sigma^{2}} \right) \right) = \sum_{i=1}^{d} \ln \left( \frac{1}{\sqrt{ 2\pi\sigma^{2} }} \right) \ln \left( \exp \left( - \frac{(x-\mu_{c,j})^{2}}{2\sigma^{2}} \right) \right)   =\\
\sum_{i=1}^{d} \ln \left( \frac{1}{\sqrt{ 2\pi\sigma^{2} }} \right)  \left( - \frac{(x-\mu_{c,j})^{2}}{2\sigma^{2}} \right) = \sum_{i=1}^{d} (\ln \left( 1 \right) - \ln (\sqrt{ 2\pi\sigma^{2} }))  \left( - \frac{(x-\mu_{c,j})^{2}}{2\sigma^{2}} \right)=\\
\sum_{i=1}^{d} (0 - \ln (\sqrt{ 2\pi\sigma^{2} }))  \left( - \frac{(x-\mu_{c,j})^{2}}{2\sigma^{2}} \right)= \sum_{i=1}^{d} -\ln (\sqrt{ 2\pi\sigma^{2} })  \left( - \frac{(x-\mu_{c,j})^{2}}{2\sigma^{2}} \right)
}
$$
* **b.** Log-posterior ratio:
$$
\displaylines{
\ln\left(\frac{P(Y=1 \mid \mathbf{x})}{P(Y=0 \mid \mathbf{x})}\right) = \ln(P(Y=1 \mid \mathbf{x})) - \ln (P(Y=0 \mid \mathbf{x})) =\\ \ln\left( \frac{\prod_{i=1}^{d} P(x_{i}|Y=1)\cdot \pi_{1}}{\mathcal{N}(\mu_{1,j})} \right) - \ln (P(Y=0 \mid \mathbf{x})\cdot \pi_{0})
}
$$
* **c.** Linear decision boundary & rule proof:

$$
\displaylines{

}
$$

---

### Question 5: $K$-Means & EM / שאלה 5: $K$-Means ו-EM
* **a.** Full joint likelihood:

* **b.** Posterior responsibilities and limit $\sigma \to 0$:
  ```text
  
  ```
* **c.** $K$-Means algorithm & E-step equivalence:
  ```text
  
  ```
* **d.** M-step equivalence & EM-$K$-Means identity proof:
  ```text
  
  ```
