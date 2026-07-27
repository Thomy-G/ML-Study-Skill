---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Summary
date_added: 2026-06-21
---
# Topic 11 — Ensemble Learning, Bagging & Boosting (Class 06C)

**MOC:** [[Machine learning MOC]]
**Course:** [[English]]

# Study Guide: Topic 11 — Ensemble Learning, Bagging & Boosting (Class 06C)

## 1. Context: Combining Weak Learners to Form a Strong Model

Until now, our strategy for reducing error focused on optimizing a single complex hypothesis function (e.g., growing a deep decision tree or training a massive neural network). However, highly complex models are inherently prone to high variance and overfitting.

**Ensemble Learning** takes the opposite approach: instead of training one highly complex model, we train a collection of multiple simpler, moderately performing models—called **weak learners** (e.g., decision tree stumps of depth 1)—and combine their individual predictions into a single unified output.

- **A Weak Learner** is a classifier that performs only slightly better than random guessing (i.e., its error rate is strictly less than $0.5$ for binary tasks: $\epsilon < 0.5$).
    
- **A Strong Learner** is an arbitrary classifier that can achieve high accuracy with an arbitrarily low error rate ($\epsilon \rightarrow 0$).
    

## 2. Bagging vs. Boosting

Ensemble paradigms generally target different components of the generalization error:

|**Dimension / View**|**Bagging (Bootstrap Aggregating)**|**Boosting (e.g., AdaBoost)**|
|---|---|---|
|**Primary Error Targeted**|**Variance Reduction** (Counteracts overfitting).|**Bias Reduction** (Counteracts underfitting).|
|**Learner Relationship**|**Parallel:** Weak models are trained completely independently of one another.|**Sequential:** Weak models are trained one after another; each model corrects the errors of its predecessor.|
|**Data Sample Strategy**|**Bootstrap Sampling:** Each model gets a random slice of data sampled with replacement.|**Adaptive Reweighting:** The entire dataset is re-weighted at each step to highlight hard examples.|
|**Combination Rule**|Uniform majority voting or simple averaging.|Weighted linear combination based on learner accuracy.|

## 3. The AdaBoost Algorithm (Adaptive Boosting)

**AdaBoost** is the classic sequential boosting framework for binary classification ($y \in \{-1, +1\}$). It forces a sequence of weak base classifiers $h_t(\mathbf{x})$ to focus on the training instances that were misclassified by earlier models in the sequence.

### Mathematical Execution Protocol

Given a training dataset $\mathcal{D} = \{(\mathbf{x}_1, y_1), \dots, (\mathbf{x}_n, y_n)\}$:

1. **Initialize the Data Weights Uniformly:** At the first step ($t=1$), assign an equal distribution weight to every training instance:
    
    $$D_1(i) = \frac{1}{n} \quad \text{for all } i = 1, \dots, n$$
    
2. **Iterate Sequentially for $t = 1, \dots, T$:**
    
    - Train a base weak classifier $h_t: \mathcal{X} \rightarrow \{-1, +1\}$ using the current data weight distribution $D_t$.
        
    - Calculate the weighted training error rate $\epsilon_t$ of this base classifier:
        
        $$\epsilon_t = \sum_{i=1}^n D_t(i) \cdot \mathbb{I}\{h_t(\mathbf{x}_i) \neq y_i\}$$
        
    - If $\epsilon_t \geq 0.5$, terminate the loop (the learner is no better than random guessing).
        
    - Compute the operational voting weight $\alpha_t$ for this classifier based on its accuracy:
        
        $$\alpha_t = \frac{1}{2} \ln\left( \frac{1 - \epsilon_t}{\epsilon_t} \right)$$
        
    - Update the weight distribution for all data instances to prepare for the next model ($t+1$):
        
        $$D_{t+1}(i) = \frac{D_t(i) \cdot \exp\left( -\alpha_t y_i h_t(\mathbf{x}_i) \right)}{Z_t}$$
        
        where $Z_t$ is a normalization constant chosen so that the updated weights sum to 1 ($\sum_i D_{t+1}(i) = 1$):
        
        $$Z_t = \sum_{i=1}^n D_t(i) \exp\left( -\alpha_t y_i h_t(\mathbf{x}_i) \right)$$
        
3. **Final Combined Classifier:** The final ensemble prediction is a weighted vote across all $T$ weak classifiers:
    
    $$H(\mathbf{x}) = \text{sign}\left( \sum_{t=1}^T \alpha_t h_t(\mathbf{x}) \right)$$
    

## 4. The Weight Update Mechanics

The update term $\exp\left( -\alpha_t y_i h_t(\mathbf{x}_i) \right)$ dynamically alters the importance of data points based on correctness:

- **If classified correctly ($y_i h_t(\mathbf{x}_i) = +1$):** The weight is multiplied by $e^{-\alpha_t}$ (since $\alpha_t > 0$, this shrinks the weight).
    
- **If misclassified ($y_i h_t(\mathbf{x}_i) = -1$):** The weight is multiplied by $e^{+\alpha_t}$ (this amplifies the weight, forcing the next weak learner to focus on this point).
    

## 5. Python / NumPy Implementation Snippet

This block shows how to compute a single complete AdaBoost iteration step including error evaluation, voting weight calculation, and data distribution updates:

```python
import numpy as np

def perform_adaboost_step(X, y, weights, predict_fn):
    """
    Computes a single sequential AdaBoost update step.
    y: 1D array of true labels (-1 or +1)
    weights: 1D array containing current data weight distribution D_t
    predict_fn: A function handle that outputs predictions (-1 or +1) for the weak learner
    """
    n = len(y)
    
    # Get predictions from the current trained weak learner
    y_pred = predict_fn(X)
    
    # 1. Compute weighted error: epsilon_t = sum(D_t(i) * I(y_pred != y))
    misclassified = (y_pred != y).astype(float)
    epsilon_t = np.sum(weights * misclassified)
    
    # Avoid division-by-zero boundary issues if error is perfectly 0
    epsilon_t = max(epsilon_t, 1e-10)
    
    # 2. Compute learner voting weight alpha_t
    alpha_t = 0.5 * np.log((1.0 - epsilon_t) / epsilon_t)
    
    # 3. Update the weight distribution: D_(t+1) = D_t * exp(-alpha * y * y_pred)
    updated_weights = weights * np.exp(-alpha_t * y * y_pred)
    
    # 4. Normalize weights so they sum to 1
    Z_t = np.sum(updated_weights)
    updated_weights /= Z_t
    
    return alpha_t, updated_weights, epsilon_t
```

## 6. Solved Exam-Style Examples

### Example 1: Calculating Voting Weights and Normalized Distribution

**Problem Statement:** You have a small dataset with 4 training instances. At iteration step $t$, the current weight distribution is uniform: $D_t = [0.25, 0.25, 0.25, 0.25]^T$. Your trained weak classifier $h_t(\mathbf{x})$ correctly classifies samples 1, 2, and 3, but misclassifies sample 4.

Calculate the exact analytical value of the classifier's voting weight $\alpha_t$, and compute the updated, normalized weight distribution vector $D_{t+1}$.

#### Step-by-Step Analytical Derivation

1. **Compute the weighted training error rate $\epsilon_t$:** Since only sample 4 is misclassified, we sum the weights of the failures:
    
    $$\epsilon_t = \sum_{i \in \text{Fail}} D_t(i) = D_t(4) = 0.25$$
    
2. **Calculate the classifier's voting weight $\alpha_t$:**
    
    $$\alpha_t = \frac{1}{2} \ln\left( \frac{1 - \epsilon_t}{\epsilon_t} \right) = \frac{1}{2} \ln\left( \frac{1 - 0.25}{0.25} \right) = \frac{1}{2} \ln\left( \frac{0.75}{0.25} \right) = \frac{1}{2} \ln(3) \approx 0.5493$$
    
3. **Compute the unnormalized updated weights $\tilde{D}_{t+1}$:**
    
    - For correct samples (1, 2, 3): $\tilde{D}_{t+1}(i) = 0.25 \cdot e^{-\alpha_t} = 0.25 \cdot e^{-\frac{1}{2}\ln(3)} = 0.25 \cdot \frac{1}{\sqrt{3}} \approx 0.1443$
        
    - For the misclassified sample (4): $\tilde{D}_{t+1}(4) = 0.25 \cdot e^{+\alpha_t} = 0.25 \cdot e^{+\frac{1}{2}\ln(3)} = 0.25 \cdot \sqrt{3} \approx 0.4330$
        
4. **Calculate the normalization factor $Z_t$:**
    
    $$Z_t = \sum_{i=1}^4 \tilde{D}_{t+1}(i) = 3 \cdot \left(0.25 \cdot \frac{1}{\sqrt{3}}\right) + \left(0.25 \cdot \sqrt{3}\right) = 0.25 \cdot \left( \frac{3}{\sqrt{3}} + \sqrt{3} \right) = 0.25 \cdot (2\sqrt{3}) = 0.5\sqrt{3} \approx 0.8660$$
    
5. **Normalize to find the final weight distribution vector $D_{t+1}$:**
    
    - For correct samples ($i=1,2,3$): $D_{t+1}(i) = \frac{0.25 \cdot \frac{1}{\sqrt{3}}}{0.5\sqrt{3}} = \frac{0.25}{0.5 \cdot 3} = \frac{0.25}{1.5} = \frac{1}{6} \approx 0.1667$
        
    - For the misclassified sample ($i=4$): $D_{t+1}(4) = \frac{0.25 \cdot \sqrt{3}}{0.5\sqrt{3}} = \frac{0.25}{0.5} = \frac{1}{2} = 0.5000$
        

**Final Exam Answer:** The voting weight is exactly **$\alpha_t = \frac{\ln(3)}{2}$**. The updated, normalized weight distribution vector is **$D_{t+1} = [\frac{1}{6}, \frac{1}{6}, \frac{1}{6}, \frac{1}{2}]^T$**. Notice that the weight of the misclassified instance has doubled from $0.25$ to $0.50$, forcing the next learner to prioritize it.

### Example 2: Proving the Normalized Error Bound Property

**Problem Statement:** Prove that the total sum of data weights assigned to all misclassified instances after an AdaBoost update step is always exactly equal to $0.5$, prior to running the next iteration. That is, show that $\epsilon_{t+1 \text{ (on old model)}} = 0.5$.

#### Step-by-Step Analytical Derivation

1. Let $\mathcal{M}$ be the set of indices misclassified by model $h_t$, and let $\mathcal{C}$ be the set of correctly classified indices.
    
2. By definition, the weighted error rate is $\epsilon_t = \sum_{i \in \mathcal{M}} D_t(i)$, and the complement is $1-\epsilon_t = \sum_{i \in \mathcal{C}} D_t(i)$.
    
3. Write out the sum of the normalized updated weights for the misclassified instances in set $\mathcal{M}$:
    
    $$\sum_{i \in \mathcal{M}} D_{t+1}(i) = \frac{\sum_{i \in \mathcal{M}} D_t(i) e^{\alpha_t}}{Z_t} = \frac{e^{\alpha_t} \sum_{i \in \mathcal{M}} D_t(i)}{Z_t} = \frac{\epsilon_t e^{\alpha_t}}{Z_t}$$
    
4. Express the normalization constant $Z_t$ by splitting the summation across the correct and incorrect sets:
    
    $$Z_t = \sum_{i \in \mathcal{C}} D_t(i) e^{-\alpha_t} + \sum_{i \in \mathcal{M}} D_t(i) e^{\alpha_t} = (1-\epsilon_t)e^{-\alpha_t} + \epsilon_t e^{\alpha_t}$$
    
5. Substitute the analytical definition of $\alpha_t = \ln\sqrt{\frac{1-\epsilon_t}{\epsilon_t}}$, which implies $e^{\alpha_t} = \sqrt{\frac{1-\epsilon_t}{\epsilon_t}}$ and $e^{-\alpha_t} = \sqrt{\frac{\epsilon_t}{1-\epsilon_t}}$:
    
    $$Z_t = (1-\epsilon_t)\sqrt{\frac{\epsilon_t}{1-\epsilon_t}} + \epsilon_t \sqrt{\frac{1-\epsilon_t}{\epsilon_t}} = \sqrt{\epsilon_t(1-\epsilon_t)} + \sqrt{\epsilon_t(1-\epsilon_t)} = 2\sqrt{\epsilon_t(1-\epsilon_t)}$$
    
6. Substitute both the $Z_t$ expression and the $e^{\alpha_t}$ expression back into the misclassified weight sum formula:
    
    $$\sum_{i \in \mathcal{M}} D_{t+1}(i) = \frac{\epsilon_t \cdot \sqrt{\frac{1-\epsilon_t}{\epsilon_t}}}{2\sqrt{\epsilon_t(1-\epsilon_t)}} = \frac{\sqrt{\epsilon_t(1-\epsilon_t)}}{2\sqrt{\epsilon_t(1-\epsilon_t)}} = \frac{1}{2} = 0.5$$
    

**Final Exam Answer:** This algebraic proof demonstrates that **after re-weighting, the total mass of the misclassified examples always equals exactly $0.5$**. This shows that the current weak learner is rendered completely ineffective (performing no better than random guessing) on the newly updated distribution, forcing the ensemble to select a structurally different classifier for the next step.

---

## 6. Gradient Boosting & XGBoost (Extreme Gradient Boosting)

While AdaBoost re-weights data points based on classification errors using Exponential Loss, **Gradient Boosting** generalizes boosting to any arbitrary differentiable loss function $\mathcal{L}(y, f(\mathbf{x}))$ (e.g., MSE for regression, Log Loss for classification).

---

### 6.1 The Gradient Boosting Machine (GBM) Principle

Gradient Boosting constructs an additive model $F_T(\mathbf{x}) = \sum_{t=1}^T \gamma_t h_t(\mathbf{x})$ sequentially. At iteration step $t$:

1. **Compute Pseudo-Residuals (Negative Gradients):**  
   Instead of re-weighting sample points, each new weak base learner $h_t(\mathbf{x})$ (typically a shallow decision tree) is trained to fit the **negative gradient** of the loss function evaluated at the current model predictions:
   $$r_{i, t} = -\left[ \frac{\partial \mathcal{L}(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)} \right]_{F(\mathbf{x}) = F_{t-1}(\mathbf{x})}$$
   *(For Mean Squared Error loss $\mathcal{L} = \frac{1}{2}(y - F(\mathbf{x}))^2$, the negative gradient is simply the standard residual $r_{i,t} = y_i - F_{t-1}(\mathbf{x}_i)$).*

2. **Fit Weak Learner & Step Optimization:**  
   Train a base tree $h_t(\mathbf{x})$ to predict $r_{i,t}$, then solve a 1D optimization problem to find the optimal step size $\gamma_t$:
   $$\gamma_t = \arg\min_{\gamma} \sum_{i=1}^n \mathcal{L}\Big(y_i, F_{t-1}(\mathbf{x}_i) + \gamma h_t(\mathbf{x}_i)\Big)$$

3. **Model Update:**  
   $$F_t(\mathbf{x}) = F_{t-1}(\mathbf{x}) + \nu \cdot \gamma_t h_t(\mathbf{x})$$
   *(where $\nu \in (0, 1]$ is a learning rate shrinkage hyperparameter).*

---

### 6.2 XGBoost (Extreme Gradient Boosting)

**XGBoost** (Chen & Guestrin, 2016) is an advanced, highly regularized, and parallelized implementation of Gradient Boosting.

#### 1. 2nd-Order Taylor Series Loss Approximation
At step $t$, XGBoost approximates the objective loss using a 2nd-order Taylor expansion around $\hat{y}^{(t-1)}$:

$$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ \mathcal{L}(y_i, \hat{y}^{(t-1)}) + g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t)$$

where:
* **Gradient:** $g_i = \frac{\partial \mathcal{L}(y_i, \hat{y}^{(t-1)})}{\partial \hat{y}^{(t-1)}}$
* **Hessian:** $h_i = \frac{\partial^2 \mathcal{L}(y_i, \hat{y}^{(t-1)})}{\partial (\hat{y}^{(t-1)})^2}$
* **Tree Complexity Regularization $\Omega(f_t)$:**
  $$\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
  *(where $T$ is the number of terminal leaves and $w_j$ are leaf weight values).*

#### 2. Optimal Leaf Weight Formula
For a fixed tree structure with leaf instance index sets $I_j = \{i \mid q(\mathbf{x}_i) = j\}$, the optimal weight $w_j^*$ for leaf $j$ is derived analytically:

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

#### 3. Tree Split Evaluation Metric (Gain)
The quality score gain when splitting a node into left $I_L$ and right $I_R$ child subsets is:

$$\text{Gain} = \frac{1}{2} \left[ \frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma$$