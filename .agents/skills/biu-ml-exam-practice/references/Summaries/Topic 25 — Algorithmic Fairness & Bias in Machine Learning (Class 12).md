---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Summary
date_added: 2026-07-21
---
# Topic 25 — Algorithmic Fairness & Bias in Machine Learning (Class 12)

**MOC:** [[Machine learning MOC]]
**Course:** [[English]]

# Study Guide: Topic 25 — Algorithmic Fairness & Bias in Machine Learning (Class 12)

## 1. Context: Causality vs. Correlation in Machine Learning

Machine learning models excel at finding statistical patterns, but they struggle to differentiate between correlation and causation. This is a primary source of algorithmic bias.

### Observational vs. Interventional Probabilities
*   **Correlation (Observation):** Evaluates $P(Y \mid X)$. For example, if we observe that the presence of puddles ($X$) strongly correlates with rain ($Y$), the conditional probability $P(\text{Rain} \mid \text{Puddle}) > P(\text{Rain})$ is high. However, this does not mean puddles cause rain.
*   **Causation (Intervention):** Evaluates $P(Y \mid \text{do}(X))$, representing the probability of $Y$ when we actively force the state of $X$ to happen (the "do-calculus"). If we intervene using a hose to create a puddle, the probability of rain does not change: $P(\text{Rain} \mid \text{do}(\text{Puddle})) = P(\text{Rain})$.

### Confounding Variables
Two variables can appear statistically correlated simply because a third, unseen variable (a **confounder**) causes both.
*   *Example:* There is a strong statistical correlation between a country's chocolate consumption per capita and its number of Nobel Laureates. However, chocolate consumption does not cause Nobel Prizes; both are correlated with an underlying confounder (e.g., a country's GDP or wealth).

### How Models Learn Non-Causal Biases
Deep learning models optimize for predictive accuracy on the training data. If a non-causal feature happens to be highly predictive in the dataset, the model will use it.
*   *Example (Loan Default Risk):* In a credit-scoring model, a protected attribute like skin color ($A$) might correlate with historical social class ($C$), which correlates with default rate ($Y$). Even if skin color has no causal link to creditworthiness, the model may use skin color as a proxy feature. Consequently, a creditworthy individual from a minority group may be unfairly denied a loan based on non-causal correlations learned by the model.

---

## 2. Biases Due to Data Imbalance

Datasets naturally reflect the demographics of the society they are collected from. By definition, datasets contain fewer samples of minority groups than majority groups.

### The Sample Size vs. Error Rate Relation
In statistical learning, the generalization error of a model decreases as the number of training samples $n$ increases. Specifically, the estimation error bound scales as:

$$\text{Error} \propto \frac{1}{\sqrt{n}}$$

Because $n_{\text{minority}} \ll n_{\text{majority}}$, the model has significantly less information about minority populations. This data imbalance mathematically guarantees that the model will have **higher error rates for minority groups**, even if the model is structurally unbiased.

---

## 3. Mathematical Definitions of Group Fairness

Let $X$ represent the non-protected features of an individual, $A \in \{0, 1\}$ represent a binary protected attribute (e.g., race, gender), $y \in \{0, 1\}$ represent the true label, and $\hat{y} \in \{0, 1\}$ represent the model's binary classification prediction.

### 1. Demographic Parity (Statistical Parity)
Demographic parity requires that the likelihood of receiving a positive prediction is completely independent of the protected attribute:

$$P(\hat{y} = 1 \mid A = 0) = P(\hat{y} = 1 \mid A = 1)$$

*   *Limitation:* It ignores the true base rates of the target variable $y$. If the true class distribution differs across groups, enforcing demographic parity can force the model to make suboptimal or incorrect predictions for qualified candidates.

### 2. Equalized Odds
Equalized odds requires that the predictor has equal true positive rates (TPR) and equal false positive rates (FPR) across both groups:

$$P(\hat{y} = 1 \mid A = 0, y = c) = P(\hat{y} = 1 \mid A = 1, y = c) \quad \text{for } c \in \{0, 1\}$$

*   *TPR Equality (Equal Opportunity):* For $c = 1$: $P(\hat{y} = 1 \mid A=0, y=1) = P(\hat{y} = 1 \mid A=1, y=1)$.
*   *FPR Equality:* For $c = 0$: $P(\hat{y} = 1 \mid A=0, y=0) = P(\hat{y} = 1 \mid A=1, y=0)$.

### 3. Calibration
Calibration requires that among all individuals assigned a specific probability score $p = f(X)$, the proportion of actual positive outcomes is identical regardless of group membership:

$$P(y = 1 \mid f(X) = p, A = 0) = P(y = 1 \mid f(X) = p, A = 1) = p$$

#### Post-Hoc Probability Calibration Methods
Standard deep neural networks trained with Cross-Entropy Loss are often overconfident (e.g. predicting $0.9$ confidence when empirical accuracy is only $70\%$). To fix miscalibration:

1. **Histogram Binning:** Groups uncalibrated predicted probabilities into $K$ discrete bins and replaces every prediction in a bin with the empirical true positive rate of that bin. (Simple, but outputs discrete step values).
2. **Isotonic Regression:** Learns a non-parametric, monotonic (non-decreasing) step function mapping raw outputs to calibrated probabilities.
3. **Platt Scaling:** Fits a 2-parameter logistic regression model over raw model outputs/logits:
   $$\text{calibrated\_p} = \sigma(a \cdot z + b)$$
4. **Temperature Scaling:** A single-parameter variant of Platt Scaling for multiclass Softmax models. Divides all raw logits $\mathbf{z}$ by a scalar temperature $T > 0$ before applying Softmax:
   $$P_i = \frac{\exp(z_i / T)}{\sum_{j=1}^K \exp(z_j / T)}$$
   *(If $T > 1$, it softens the output probability distribution; if $T < 1$, it sharpens it).*

---

## 4. The Impossibility Theorem of Fairness

A key theoretical result in algorithmic fairness (proven by Chouldechova in 2017 and Kleinberg et al. in 2016) states that **it is mathematically impossible to simultaneously satisfy Demographic Parity, Equalized Odds, and Calibration** unless:
1.  The classifier makes zero errors (is perfect: $f(X) = y$).
2.  The base rates of the target variable are identical across groups ($P(y = 1 \mid A = 0) = P(y = 1 \mid A = 1)$).

When base rates differ, choosing one definition of fairness mathematically violates the others, forcing system designers to make explicit ethical trade-offs.

---

## 5. Individual Fairness

As an alternative to group-level metrics, **Individual Fairness** (Dwork et al. 2012) operates on the principle that "similar individuals should be treated similarly."

### Mathematical Formulation
Let $x, x'$ be the input feature vectors of two individuals (excluding the protected attribute $A$). Let $d(\cdot, \cdot)$ be a task-specific metric measuring similarity in feature space, and let $d_{\text{out}}(f(x), f(x'))$ measure the distance between the model's predictions.

Individual fairness is defined as:

$$d_{\text{out}}(f(x), f(x')) \le L \cdot d(x, x')$$

where $L \ge 0$ is a Lipschitz constant. This constraint ensures that if two candidates are close in similarity according to a metric $d$, their prediction outcomes must be correspondingly close.

---
## 🔗 Navigation
**Previous:** [[Topic 24 — Advanced Mathematical Proofs & Derivations]] | **Next:** [[Machine learning MOC]]
