---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Summary
date_added: 2026-06-21
---
# Topic 6 — Generalization Error Decomposition & Loss Functions

**MOC:** [[Machine learning MOC]]
**Course:** [[English]]

# Study Guide: Topic 6 — Generalization Error Decomposition & Loss Functions

## 1. The Approximation-Estimation-Optimization Error Breakdown

When we look at the final performance of a model found in practice, its total error compared to the absolute best possible theoretical function in the universe can be decomposed into three distinct gaps.

Let:

- $\mathcal{F}$ be the space of all possible mathematical functions.
    
- $f^*$ be the true, ideal target labeling function (the "ground truth").
    
- $\mathcal{H}$ be our chosen hypothesis space (e.g., all linear models or a specific neural network architecture).
    
- $f_{\mathcal{H}}$ be the absolute best function _inside_ our hypothesis space: $f_{\mathcal{H}} = \arg\min_{h \in \mathcal{H}} L_{\text{gen}}(h)$.
    
- $\hat{h}_{\text{global}}$ be the hypothesis inside $\mathcal{H}$ that minimizes the _training_ error globally if we had infinite optimization time.
    
- $\hat{h}_{\text{learned}}$ be the actual empirical hypothesis our algorithm outputs in practice.
    

```
   Space of all possible functions (F)
 ┌─────────────────────────────────────────────────────────┐
 │   f* (True Target Function)                             │
 │       │                                                 │
 │       ▼ [Approximation Error]                           │
 │   ┌──────────────────────────────┐                      │
 │   │  Hypothesis Space (H)        │                      │
 │   │  f_H (Best in H)             │                      │
 │   │      │                       │                      │
 │   │      ▼ [Estimation Error]    │                      │
 │   │  h_global (Best on Train)    │                      │
 │   │      │                       │                      │
 │   │      ▼ [Optimization Error]  │                      │
 │   │  h_learned (Found in Reality)│                      │
 │   └──────────────────────────────┘                      │
 └─────────────────────────────────────────────────────────┘
```

### The Three Errors

1. **Approximation Error (Expressivity Gap):** The distance between the true function $f^*$ and the best possible function in our chosen class $f_{\mathcal{H}}$. This measures what we lose by restricting ourselves to a specific model family. _Solution:_ Use a larger, more expressive hypothesis space (e.g., add layers to an MLP).
    
2. **Estimation Error (Generalization Gap):** The penalty we pay for only having a _finite sample_ of data instead of the true infinite distribution. It tracks how much the empirical training minimizer deviates from the true risk minimizers inside $\mathcal{H}$. _Solution:_ Collect more training data or add regularization.
    
3. **Optimization Error (Training Error Gap):** The gap between what our optimization algorithm actually found ($\hat{h}_{\text{learned}}$) versus the absolute minimum training error possible ($\hat{h}_{\text{global}}$). It happens because our algorithms get stuck in local minima, plateaus, or run out of epochs. _Solution:_ Tune the learning rate, use better optimizers (like ADAM), or apply BatchNorm.
    

## 2. The Bias-Variance Tradeoff (For Mean Squared Error)

When dealing with continuous regression targets using the Mean Squared Error (MSE) loss, the expected generalization error on a novel test point $x$ can be decomposed cleanly via algebraic variance expansion.

### The Formula

Let $y = f(x) + \epsilon$ be the true label, where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ is intrinsic random environmental noise. Let $h(x; S)$ be the hypothesis trained on a random dataset sample $S$.

The expected error over all possible training datasets $S$ is:

$$\mathbb{E}_S \left[ (y - h(x; S))^2 \right] = \text{Bias}\big(h(x)\big)^2 + \text{Variance}\big(h(x)\big) + \sigma^2$$

Where:

- $\text{Bias}\big(h(x)\big) = \mathbb{E}_S[h(x; S)] - f(x)$
    
    Measures how far off the _average_ prediction over all possible datasets is from the true functional value. High bias leads to **underfitting**.
    
- $\text{Variance}\big(h(x)\big) = \mathbb{E}_S\left[ (h(x; S) - \mathbb{E}_S[h(x; S)])^2 \right]$
    
    Measures how wildly the model's predictions fluctuate if it is trained on different random data slices. High variance leads to **overfitting**.
    
- $\sigma^2$ is the **Irreducible Error**. The baseline noise inherent to the physics of the data collection process itself; no model can ever surpass this limit.
    

## 3. Formal Classification Loss Functions

For classification tasks ($y \in \{-1, +1\}$), we evaluate the model's correctness using the **margin** $y \cdot h(x)$. A positive margin ($y \cdot h(x) > 0$) means a correct prediction, while a negative margin means a mistake.

1. **Zero-One Loss ($\ell_{0\text{-}1}$):**
    
    $$\ell_{0\text{-}1}(h(x), y) = \mathbb{I}\{y \cdot h(x) \leq 0\}$$
    
    _Rationale:_ Directly counts classification mistakes. However, it is a step function with a gradient of zero everywhere, making it completely impossible to optimize using gradient descent.
    
2. **Hinge Loss ($\ell_{\text{hinge}}$):**
    
    $$\ell_{\text{hinge}}(h(x), y) = \max(0, 1 - y \cdot h(x))$$
    
    _Rationale:_ A convex, continuous upper bound for the $0\text{-}1$ loss used directly in **Support Vector Machines (SVM)**. It penalizes mistakes and even penalizes correct predictions if they are too close to the decision boundary (margin $< 1$).
    
3. **Logistic / Cross-Entropy Loss ($\ell_{\text{logistic}}$):**
    
    $$\ell_{\text{logistic}}(h(x), y) = \ln(1 + e^{-y \cdot h(x)})$$
    
    _Rationale:_ Smooth, continuous, and differentiable everywhere. It scales infinitely with increasingly bad predictions, providing reliable gradient signals for training binary classifiers via Logistic Regression.
    

## 4. Python Implementation Snippet

This block shows how to programmatically calculate these classification loss profiles given raw prediction margins:

```python
import numpy as np

def compute_classification_losses(y_true, scalar_scores):
    """
    y_true: 1D array of true labels (-1 or +1)
    scalar_scores: 1D array of raw model output scores h(x)
    """
    margins = y_true * scalar_scores
    
    zero_one = np.where(margins <= 0, 1.0, 0.0)
    hinge = np.maximum(0, 1.0 - margins)
    logistic = np.log(1.0 + np.exp(-margins))
    
    return {
        "zero_one": np.mean(zero_one),
        "hinge": np.mean(hinge),
        "logistic": np.mean(logistic)
    }
```

## 5. Solved Exam-Style Examples

### Example 1: Interpreting an Error Dip (From 2022 Moed A)

**Problem Statement:** During a deep network optimization loop, you monitor the validation error over successive epochs. At epoch 30, you notice a sharp, sudden step-down drop in the validation error curve. Which of the following engineering adjustments caused this artifact?

_(i) Shifting training from 1 GPU to 2 GPUs._

_(ii) Decaying/dropping the optimization learning rate ($\eta$) by a factor of 10._

_(iii) Adjusting the mini-batch sample size from 100 to 102._

Provide a rigorous technical justification.

#### Analytical Evaluation

- _(i) Is incorrect:_ Adding hardware GPUs alters only parallel matrix execution velocity. It shortens clock runtime but does not shift the underlying mathematical path of the loss space optimization curve per epoch.
    
- _(iii) Is incorrect:_ Modifying the batch size by an insignificant margin of 2 samples introduces negligible adjustments to gradient variance calculations and cannot trigger a massive, discrete validation drop.
    
- _**(ii) Is correct:** Decay of the learning rate ($\eta$) is a standard optimization recipe. Early in training, a high learning rate allows rapid exploration across high-loss regions. However, the weights eventually begin bouncing chaotically around a valley basin due to steps that are too large. Dropping the learning rate shrinks the step sizes, allowing the weights to settle smoothly down into the local minimum, resulting in a sudden drop in training and validation error._
    

**Final Exam Answer:** **(ii) Reducing the learning rate.** This allows the optimizer to drop directly into a tighter local coordinate minimum that was previously overshot by large update steps.

### Example 2: Mathematical Analysis of Hinge Loss Gradients

**Problem Statement:** A linear classifier maps inputs to continuous scores via $h(\mathbf{x}) = \mathbf{w}^T \mathbf{x}$. The model is optimized using Stochastic Gradient Descent on a single data instance $(\mathbf{x}_i, y_i)$ under a Zero-Margin Hinge Loss objective:

$$\ell(\mathbf{w}) = \max(0, -y_i(\mathbf{w}^T \mathbf{x}_i))$$

Derive the exact parameter update rule for the weight vector $\mathbf{w}$ given a learning rate step size $\eta$.

#### Step-by-Step Analytical Derivation

1. Analyze the sub-gradient properties of the piecewise function based on the data margin condition:
    
    - **Case A (Correctly classified sample):** If $-y_i(\mathbf{w}^T \mathbf{x}_i) < 0 \implies y_i(\mathbf{w}^T \mathbf{x}_i) > 0$. The active branch of the function is $0$. The loss is perfectly flat, meaning the gradient is zero:
        
        $$\nabla_{\mathbf{w}} \ell = \mathbf{0}$$
        
    - **Case B (Misclassified sample):** If $-y_i(\mathbf{w}^T \mathbf{x}_i) > 0 \implies y_i(\mathbf{w}^T \mathbf{x}_i) < 0$. The active branch of the function is $-y_i \mathbf{w}^T \mathbf{x}_i$. Differentiating this linear component with respect to $\mathbf{w}$ gives:
        
        $$\nabla_{\mathbf{w}} \ell = -y_i \mathbf{x}_i$$
        
2. Formulate the standard Gradient Descent parameter adjustment rule ($\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla_{\mathbf{w}} \ell$):
    
    - For $y_i(\mathbf{w}^T \mathbf{x}_i) > 0$: $\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)}$ (No change).
        
    - For $y_i(\mathbf{w}^T \mathbf{x}_i) < 0$: $\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta(-y_i \mathbf{x}_i) = \mathbf{w}^{(t)} + \eta y_i \mathbf{x}_i$.
        

**Final Exam Answer:** The weight update rule under zero-margin hinge loss matches the classic **Perceptron Learning Rule**:

$$\mathbf{w}^{(t+1)} = \begin{cases} \mathbf{w}^{(t)} & \text{if } y_i(\mathbf{w}^T \mathbf{x}_i) > 0 \\ \mathbf{w}^{(t)} + \eta y_i \mathbf{x}_i & \text{if } y_i(\mathbf{w}^T \mathbf{x}_i) < 0 \end{cases}$$


---
## 🔗 Navigation
**Previous:** [[ ]] | **Next:** [[ ]]