---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Summary
date_added: 2026-06-21
---
# Topic 8 — Linear Binary Classification & Logistic Regression (Class 04B, Recitation 4, Past Exams)

**MOC:** [[Machine learning MOC]]
**Course:** [[English]]

# Study Guide: Topic 8 — Linear Binary Classification & Logistic Regression (Class 04B, Recitation 4, Past Exams)

## 1. Context: Moving from Continuous Targets to Category Decisions

We now transition from predicting continuous targets (Regression) to making categorical decisions (Classification). In a **Binary Classification** setting, our target label space is discrete: $\mathcal{Y} = \{-1, +1\}$ (or $\{0, 1\}$).

Our goal is to learn a linear hyperplane classifier parameterized by a weight vector $\mathbf{w} \in \mathbb{R}^d$ and a bias scalar $b \in \mathbb{R}$. For a given input vector $\mathbf{x}$, the classifier computes a continuous score $z = \mathbf{w}^T \mathbf{x} + b$. The final discrete prediction is determined by taking the sign of this score:

$$\hat{y} = \text{sign}\left(\mathbf{w}^T \mathbf{x} + b\right)$$

## 2. Geometric Margin & "Correct Classification"

To evaluate how well a linear classifier is performing on a training example $(\mathbf{x}_i, y_i)$, we calculate its **functional margin**:

$$\text{Margin}_i = y_i \left(\mathbf{w}^T \mathbf{x}_i + b\right)$$

- **Correct Classification ($\text{Margin}_i > 0$):** The true label $y_i$ and the prediction score share the exact same sign.
    
- **Incorrect Classification ($\text{Margin}_i \leq 0$):** The model's score has a different sign than the true label, representing a classification error.
    

The absolute magnitude of the margin reflects the model's _confidence_ in its prediction. A large positive margin means the data point lies far inside the correct side of the decision boundary.

## 3. The Perceptron Algorithm

The Perceptron is one of the earliest iterative algorithms designed to find a separating hyperplane for binary classification.

### The Objective Function

The Perceptron focuses exclusively on misclassified points ($\mathcal{M} = \{i \mid y_i (\mathbf{w}^T \mathbf{x}_i) \leq 0\}$). It minimizes the Perceptron Loss, which sums the negative margins of these failed predictions:

$$L_{\text{Perceptron}}(\mathbf{w}) = \sum_{i \in \mathcal{M}} -y_i \left(\mathbf{w}^T \mathbf{x}_i\right)$$

### The Iterative Learning Rule (From 2023 Moed A, 2025 Moed B)

1. Initialize the weights to zero or small random values: $\mathbf{w} = \mathbf{0}$.
    
2. Iterate through each training sample $(\mathbf{x}_i, y_i)$.
    
3. Compute the prediction sign: $\hat{y}_i = \text{sign}(\mathbf{w}^T \mathbf{x}_i)$.
    
4. If the sample is misclassified ($y_i \neq \hat{y}_i$), update the weight vector by shifting it directly toward the target sample:
    
    $$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + \eta y_i \mathbf{x}_i$$
    
    where $\eta > 0$ is the learning rate.
    

### Convergence Guarantee

The Perceptron algorithm is guaranteed to converge to a perfect separating solution in a finite number of steps **if and only if the training data is strictly linearly separable**. If the data cannot be perfectly separated by a straight line, the updates will loop infinitely.

## 4. Logistic Regression & Probabilistic Modeling

When data is noisy or not perfectly separable, we need a model that provides continuous, probabilistic confidence scores rather than hard, abrupt classifications. **Logistic Regression** achieves this by mapping the raw linear score $z = \mathbf{w}^T \mathbf{x}$ into a valid probability range $[0, 1]$ using the **Sigmoid function** $\sigma(z)$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### The Probabilistic Setup

We model the conditional class probabilities for labels $y \in \{-1, +1\}$ as:

$$P(y = +1 \mid \mathbf{x}; \mathbf{w}) = \sigma(\mathbf{w}^T \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}}}$$

$$P(y = -1 \mid \mathbf{x}; \mathbf{w}) = \sigma(-\mathbf{w}^T \mathbf{x}) = \frac{1}{1 + e^{\mathbf{w}^T \mathbf{x}}}$$

Using the functional margin notation, these two expressions can be combined into a single compact probability equation:

$$P(y_i \mid \mathbf{x}_i; \mathbf{w}) = \sigma(y_i \mathbf{w}^T \mathbf{x}_i) = \frac{1}{1 + e^{-y_i \mathbf{w}^T \mathbf{x}_i}}$$

## 5. Maximum Likelihood & Cross-Entropy Loss (From 2023 Moed B)

To find the optimal weight vector $\mathbf{w}$, we maximize the joint conditional likelihood across $n$ independent training observations:

$$L(\mathbf{w}) = \prod_{i=1}^n P(y_i \mid \mathbf{x}_i; \mathbf{w}) = \prod_{i=1}^n \frac{1}{1 + e^{-y_i \mathbf{w}^T \mathbf{x}_i}}$$

Maximizing this likelihood is mathematically equivalent to minimizing its negative log-likelihood. This defines the **Cross-Entropy (or Logistic) Loss** function:

$$L_{\text{CE}}(\mathbf{w}) = -\sum_{i=1}^n \ln \left( \frac{1}{1 + e^{-y_i \mathbf{w}^T \mathbf{x}_i}} \right) = \sum_{i=1}^n \ln \left( 1 + e^{-y_i \mathbf{w}^T \mathbf{x}_i} \right)$$

### The Rationale behind Cross-Entropy Loss

- **Differentiable and Convex:** Unlike the zero-one step function, Cross-Entropy is smooth and continuous everywhere, making it highly effective for gradient descent optimization.
    
- **Asymptotic Penalty:** If a sample is correctly classified with high confidence ($\text{Margin}_i \rightarrow \infty$), the loss decays toward zero. Conversely, if the model misclassifies a point with high confidence ($\text{Margin}_i \rightarrow -\infty$), the loss grows linearly, providing a powerful gradient signal to correct the weights.
    

## 6. Python / NumPy Implementation Snippet

This snippet implements a clean binary logistic regression step using vector margins:

```python
import numpy as np

class LogisticRegressionSGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.w = None

    def fit_step(self, x_i, y_i):
        """
        Performs a single Stochastic Gradient Descent update step.
        x_i: 1D array of features. y_i: label (-1 or +1).
        """
        if self.w is None:
            self.w = np.zeros(x_i.shape[0])
            
        # Compute the margin: y_i * (w^T * x_i)
        margin = y_i * np.dot(self.w, x_i)
        
        # Gradient of log(1 + exp(-margin)) w.r.t w is:
        # (-y_i * x_i) * exp(-margin) / (1 + exp(-margin)) = -y_i * x_i * (1 - sigma(margin))
        sigma_neg_margin = 1.0 / (1.0 + np.exp(margin))
        gradient = -y_i * x_i * sigma_neg_margin
        
        # Update weights: w = w - lr * gradient
        self.w -= self.lr * gradient

    def predict_proba(self, x):
        score = np.dot(x, self.w)
        return 1.0 / (1.0 + np.exp(-score))
```

## 7. Solved Exam-Style Examples

### Example 1: Deriving the Gradient of Logistic Regression (From 2023 Moed B)

**Problem Statement:** Derive the exact gradient vector of the cross-entropy loss function for a single training instance $(\mathbf{x}_i, y_i)$ where $y_i \in \{-1, +1\}$.

#### Step-by-Step Analytical Derivation

1. Write down the objective function for a single sample:
    
    $$f(\mathbf{w}) = \ln\left(1 + e^{-y_i \mathbf{w}^T \mathbf{x}_i}\right)$$
    
2. Apply the chain rule for derivatives ($\frac{d}{dx}\ln(u) = \frac{1}{u}\frac{du}{dx}$):
    
    $$\nabla_{\mathbf{w}} f(\mathbf{w}) = \frac{1}{1 + e^{-y_i \mathbf{w}^T \mathbf{x}_i}} \cdot \nabla_{\mathbf{w}}\left(e^{-y_i \mathbf{w}^T \mathbf{x}_i}\right)$$
    
3. Compute the gradient of the internal exponential component:
    
    $$\nabla_{\mathbf{w}}\left(e^{-y_i \mathbf{w}^T \mathbf{x}_i}\right) = e^{-y_i \mathbf{w}^T \mathbf{x}_i} \cdot \nabla_{\mathbf{w}}\left(-y_i \mathbf{w}^T \mathbf{x}_i\right) = e^{-y_i \mathbf{w}^T \mathbf{x}_i} \left(-y_i \mathbf{x}_i\right)$$
    
4. Substitute this back into the expression:
    
    $$\nabla_{\mathbf{w}} f(\mathbf{w}) = \frac{-y_i \mathbf{x}_i e^{-y_i \mathbf{w}^T \mathbf{x}_i}}{1 + e^{-y_i \mathbf{w}^T \mathbf{x}_i}}$$
    
5. Divide both the numerator and the denominator by $e^{-y_i \mathbf{w}^T \mathbf{x}_i}$ to clean up the expression:
    
    $$\nabla_{\mathbf{w}} f(\mathbf{w}) = \frac{-y_i \mathbf{x}_i}{\frac{1}{e^{-y_i \mathbf{w}^T \mathbf{x}_i}} + 1} = \frac{-y_i \mathbf{x}_i}{e^{y_i \mathbf{w}^T \mathbf{x}_i} + 1}$$
    
6. Recognizing that $\frac{1}{1 + e^{z}} = 1 - \sigma(z)$, we can express the final gradient relative to the model's error:
    
    $$\nabla_{\mathbf{w}} f(\mathbf{w}) = -y_i \mathbf{x}_i \left( 1 - \sigma(y_i \mathbf{w}^T \mathbf{x}_i) \right)$$
    

**Final Exam Answer:** The analytical gradient vector with respect to $\mathbf{w}$ is **$\nabla_{\mathbf{w}} L = -\frac{y_i \mathbf{x}_i}{1 + e^{y_i \mathbf{w}^T \mathbf{x}_i}}$** (or equivalently, **$-y_i \mathbf{x}_i (1 - \sigma(y_i \mathbf{w}^T \mathbf{x}_i))$**).

### Example 2: Perceptron Update Mechanics (From 2025 Moed B)

**Problem Statement:** You initialize a Perceptron model with weights $\mathbf{w}^{(0)} = [0, 2]^T$. The learning rate is fixed at $\eta = 0.5$. Your next training sample is $\mathbf{x}_1 = [2, -2]^T$ with a true label of $y_1 = +1$. Compute the updated weight vector $\mathbf{w}^{(1)}$.

#### Step-by-Step Analytical Derivation

1. Calculate the current raw prediction score using a dot product:
    
    $$z = (\mathbf{w}^{(0)})^T \mathbf{x}_1 = (0)(2) + (2)(-2) = 0 - 4 = -4$$
    
2. Determine the classifier's predicted label:
    
    $$\hat{y}_1 = \text{sign}(-4) = -1$$
    
3. Compare the prediction with the true label:
    
    Since $\hat{y}_1 = -1$ and the true label is $y_1 = +1$, the model has made a classification error ($y_1 \neq \hat{y}_1$).
    
4. Apply the Perceptron weight update rule to correct the error:
    
    $$\mathbf{w}^{(1)} = \mathbf{w}^{(0)} + \eta y_1 \mathbf{x}_1$$
    
    $$\mathbf{w}^{(1)} = \begin{bmatrix} 0 \\ 2 \end{bmatrix} + 0.5 \cdot (+1) \cdot \begin{bmatrix} 2 \\ -2 \end{bmatrix}$$
    
    $$\mathbf{w}^{(1)} = \begin{bmatrix} 0 \\ 2 \end{bmatrix} + \begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$
    

**Final Exam Answer:** The updated weight vector after processing the misclassified sample is **$\mathbf{w}^{(1)} = [1, 1]^T$**.

---
## 🔗 Navigation
**Previous:** [[ ]] | **Next:** [[ ]]