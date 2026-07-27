---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Summary
date_added: 2026-06-21
---
# Topic 9 — Multiclass Classification & Softmax Regression (Class 05A, Past Exams)

**MOC:** [[Machine learning MOC]]
**Course:** [[English]]

# Study Guide: Topic 9 — Multiclass Classification & Softmax Regression (Class 05A, Past Exams)

## 1. Context: Moving from Binary to Multi-Category Decisions

Until now, we restricted our classification tasks to binary outcomes ($\mathcal{Y} = \{-1, +1\}$). In real-world applications, systems must routinely categorize data into one of $K$ mutually exclusive classes where $K \geq 3$ (e.g., classifying an image as a dog, cat, or bird).

The training data contains $n$ i.i.d. pairs:

$$\mathcal{D} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \dots, (\mathbf{x}_n, y_n)\}$$

where each input feature vector resides in $\mathbf{x}_i \in \mathbb{R}^d$ and labels are categorical class indices $y_i \in \{1, 2, \dots, K\}$.

## 2. Structural Approaches to Multiclass Problems

### A. One-vs-All / One-vs-Rest (OvR)

- **Methodology:** Train $K$ independent, completely separate binary linear classifiers. The $k$-th classifier is optimized to solve a binary task: _"Is the example from class $k$ ($+1$) or from any other class ($-1$)?_
    
- **Inference Rule:** Pass an input $\mathbf{x}$ through all $K$ classifiers and assign the final prediction to whichever classifier outputs the highest raw score:
    
    $$\hat{y} = \arg\max_{k \in \{1, \dots, K\}} \left( \mathbf{w}_k^T \mathbf{x} + b_k \right)$$
    
- **Limitation:** Scale calibration mismatch. Since the $K$ models are trained completely independently, their output scores are unnormalized relative to one another, making it difficult to interpret them reliably as true conditional probabilities.
    

### B. Multinomial Softmax Regression

Instead of building a patchwork of individual binary classifiers, **Softmax Regression** handles all $K$ classes simultaneously inside a single, unified probabilistic framework.

- **The Parameter Matrix:** We maintain $K$ separate weight vectors, which we stack horizontally to form a comprehensive weight matrix $W \in \mathbb{R}^{d \times K}$:
    
    $$W = \begin{bmatrix} \mid & \mid & & \mid \\ \mathbf{w}_1 & \mathbf{w}_2 & \dots & \mathbf{w}_K \\ \mid & \mid & & \mid \end{bmatrix}$$
    
- **Logit Computation:** Multiplying an input vector by the weight matrix computes a raw linear score vector $\mathbf{z} \in \mathbb{R}^K$ (called **logits**):
    
    $$\mathbf{z} = W^T \mathbf{x} \implies z_k = \mathbf{w}_k^T \mathbf{x}$$
    

## 3. The Softmax Function

To transform raw, unbounded logits into a valid probability distribution over the $K$ classes, we apply the **Softmax function**. For a specific class category $k$:

$$P(y = k \mid \mathbf{x}; W) = \text{Softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}} = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}}$$

### Core Mathematical Properties

1. **Normalization Boundary:** The output components are strictly positive ($e^{z_k} > 0$) and are guaranteed to sum up exactly to 1 across all classes: $\sum_{k=1}^K P(y=k \mid \mathbf{x}) = 1$.
    
2. **Max Approximation:** The exponentiation amplifies the largest raw values significantly while turning small or negative logit gaps into numbers close to zero.
    

## 4. Cross-Entropy Loss Optimization

To find the optimal parameter matrix $W$, we evaluate model performance using the multiclass **Cross-Entropy Loss**.

Let $\mathbf{1}\{y_i = k\}$ be an indicator function that equals $1$ if the true label for instance $i$ is exactly class $k$, and $0$ otherwise. For a training dataset of $n$ i.i.d. samples, the total objective function is:

$$L(W) = -\frac{1}{n}\sum_{i=1}^n \sum_{k=1}^K \mathbf{1}\{y_i = k\} \ln \left( \frac{e^{\mathbf{w}_k^T \mathbf{x}_i}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}_i}} \right)$$

### Rationale Behind Multi-Category Cross-Entropy

This loss function directly calculates the distance between the true label vector (a one-hot encoded vector with a $1$ at index $y_i$) and the model's predicted probability vector. Minimizing this objective dynamically pushes the model to maximize the predicted probability of the correct class while suppressing the scores of incorrect classes.

## 5. Python / NumPy Implementation Snippet

This block shows how to compute logits, evaluate safe softmax probabilities, and calculate the cross-entropy loss vectorwise:

```python
import numpy as np

def multiclass_softmax_loss(X, y_true, W):
    """
    X: Design matrix of shape (n, d)
    y_true: 1D array of true class labels (0 to K-1) with length n
    W: Weight matrix of shape (d, K)
    """
    n = X.shape[0]
    
    # Compute raw logits: shape (n, K)
    logits = np.dot(X, W)
    
    # Stability trick: subtract max logit per row to prevent numerical overflow
    logits_stabilized = logits - np.max(logits, axis=1, keepdims=True)
    
    # Compute softmax probabilities: shape (n, K)
    exp_logits = np.exp(logits_stabilized)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Extract the probability assigned to the correct class for each sample
    correct_class_probs = probs[np.arange(n), y_true]
    
    # Calculate cross entropy loss
    total_loss = -np.mean(np.log(correct_class_probs + 1e-15))
    
    return total_loss, probs
```

## 6. Solved Exam-Style Examples

### Example 1: Deriving the Softmax Gradient (From 2022 Moed B)

**Problem Statement:** Consider a single training instance $(\mathbf{x}, y)$ where the true class label index is $y$. Let $P_k = P(y = k \mid \mathbf{x}; W)$ be the probability predicted by the softmax model for class $k$.

Derive the exact analytical gradient of the cross-entropy loss with respect to the specific weight vector $\mathbf{w}_m$ belonging to class $m$.

#### Step-by-Step Analytical Derivation

1. Write down the objective loss equation for the individual data sample. The cross-entropy loss for a single sample is defined as $\mathcal{L} = -\ln(P_y)$. Substituting the softmax probability $P_y = \frac{e^{\mathbf{w}_y^T \mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}}$ and simplifying via logarithmic quotient rules ($\ln\left(\frac{a}{b}\right) = \ln(a) - \ln(b)$) gives:
    
    $$\mathcal{L} = -\ln\left( \frac{e^{\mathbf{w}_y^T \mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}} \right) = -\left[ \ln\left(e^{\mathbf{w}_y^T \mathbf{x}}\right) - \ln\left( \sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}} \right) \right]$$
    
    Since $\ln(e^u) = u$, distributing the negative sign yields the final loss equation:
    
    $$\mathcal{L} = -\mathbf{w}_y^T \mathbf{x} + \ln\left( \sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}} \right)$$
    
2. Differentiate this loss expression with respect to the targeted weight vector $\mathbf{w}_m$ by using the vector chain rule:
    
    $$\nabla_{\mathbf{w}_m} \mathcal{L} = -\nabla_{\mathbf{w}_m}\left(\mathbf{w}_y^T \mathbf{x}\right) + \nabla_{\mathbf{w}_m}\left[ \ln\left( \sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}} \right) \right]$$
    
3. Analyze the first term:
    
    $$\nabla_{\mathbf{w}_m}\left(\mathbf{w}_y^T \mathbf{x}\right) = \begin{aligned} \begin{cases} \mathbf{x} & \text{if } m = y \\ \mathbf{0} & \text{if } m \neq y \end{cases} \end{aligned} = \mathbf{1}\{y = m\}\mathbf{x}$$
    
4. Analyze the second logarithmic term using the chain rule:
    
    $$\nabla_{\mathbf{w}_m}\left[ \ln\left( \sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}} \right) \right] = \frac{1}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}} \cdot \nabla_{\mathbf{w}_m}\left( \sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}} \right)$$
    
5. Within the summation, the only exponential term that depends on $\mathbf{w}_m$ is $e^{\mathbf{w}_m^T \mathbf{x}}$. The derivatives of all other terms are zero:
    
    $$\nabla_{\mathbf{w}_m}\left( \sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}} \right) = e^{\mathbf{w}_m^T \mathbf{x}} \cdot \mathbf{x}$$
    
6. Substitute this back into the logarithmic derivative term:
    
    $$\frac{e^{\mathbf{w}_m^T \mathbf{x}} \cdot \mathbf{x}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}} = \left( \frac{e^{\mathbf{w}_m^T \mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x}}} \right) \mathbf{x} = P_m \mathbf{x}$$
    
7. Combine the terms back together to get the final gradient expression:
    
    $$\nabla_{\mathbf{w}_m} \mathcal{L} = -\mathbf{1}\{y = m\}\mathbf{x} + P_m \mathbf{x} = \left( P_m - \mathbf{1}\{y = m\} \right) \mathbf{x}$$
    

**Final Exam Answer:** The analytical gradient vector with respect to $\mathbf{w}_m$ is **$\nabla_{\mathbf{w}_m} \mathcal{L} = (P_m - \mathbf{1}\{y = m\})\mathbf{x}$**. This result shows that the gradient direction is driven directly by the prediction error: the difference between the model's predicted probability ($P_m$) and the target ground truth ($1$ or $0$).

### Example 2: Invariance of Softmax to Logit Shifting

**Problem Statement:** Prove that adding a constant scalar $C \in \mathbb{R}$ to every logit component $z_k$ does not change the final output probabilities calculated by the Softmax function.

#### Step-by-Step Analytical Derivation

1. Let the original logit values be $z_1, z_2, \dots, z_K$. Let the shifted logits be defined as $\tilde{z}_k = z_k + C$.
    
2. Write down the expression for the updated Softmax calculation under these shifted inputs:
    
    $$\tilde{P}_k = \frac{e^{\tilde{z}_k}}{\sum_{j=1}^K e^{\tilde{z}_j}} = \frac{e^{z_k + C}}{\sum_{j=1}^K e^{z_j + C}}$$
    
3. Factor out the constant exponential term using exponent algebra properties ($e^{a+b} = e^a \cdot e^b$):
    
    $$\tilde{P}_k = \frac{e^{z_k} \cdot e^C}{\sum_{j=1}^K \left( e^{z_j} \cdot e^C \right)}$$
    
4. Factor the constant term $e^C$ out of the denominator summation:
    
    $$\tilde{P}_k = \frac{e^{z_k} \cdot e^C}{e^C \cdot \left( \sum_{j=1}^K e^{z_j} \right)}$$
    
5. Cancel out the $e^C$ terms from the numerator and denominator:
    
    $$\tilde{P}_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}} = P_k$$
    

**Final Exam Answer:** This algebraic cancellation proves that **the Softmax function is completely invariant to constant logit shifting**. This property is why subtracting the maximum logit value from the logit vector is a safe, effective method to prevent numerical overflow in software implementations.

---
## 🔗 Navigation
**Previous:** [[ ]] | **Next:** [[ ]]