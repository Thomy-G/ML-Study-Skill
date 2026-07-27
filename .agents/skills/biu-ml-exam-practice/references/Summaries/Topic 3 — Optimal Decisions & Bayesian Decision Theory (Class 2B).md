---
type: uni_general
course: "[[Machine learning]]"
status: 🟢 Active
order: Summary
date_added: 2026-06-21
---
# Topic 3 — Optimal Decisions & Bayesian Decision Theory (Class 2B)

**MOC:** [[Machine learning MOC]]
**Course:** [[Machine learning]]

# Study Guide: Topic 3 — Optimal Decisions & Bayesian Decision Theory (Class 2B)

## 1. What to Do When the Distribution $\mathcal{D} = P(\mathbf{x}, y)$ is Known?

In standard machine learning settings, the true underlying joint distribution $\mathcal{D}$ is completely unknown, and we must learn from empirical training samples. **Bayesian Decision Theory** explores the ideal, theoretical scenario where we possess **perfect, complete knowledge** of the joint probability distribution $P(\mathbf{x}, y)$.

When the distribution is fully known, the machine learning problem transforms from an _estimation_ task into a pure _optimization_ and _decision-making_ task. Our goal is to determine the absolute theoretical limit of performance—the optimal strategy that minimizes our expected loss.

## 2. Formal Framework & Components

To derive optimal decision boundaries, we define our setting with the following components:

- **State of Nature / True Class ($y \in \mathcal{Y}$):** The true, hidden categorical category of an instance. For binary classification, $\mathcal{Y} = \{c_1, c_2\}$ (or $\{0, 1\}$).
    
- **Observed Features ($\mathbf{x} \in \mathcal{X}$):** A continuous or discrete feature vector $\mathbf{x} \in \mathbb{R}^d$ that acts as measured evidence about the hidden state of nature.
    
- **Prior Probabilities ($P(y = c_k)$):** The baseline probability of encountering class $c_k$ _before_ observing any feature evidence. By total probability, $\sum_{k} P(y=c_k) = 1$.
    
- **Class-Conditional Densities ($P(\mathbf{x} \mid y = c_k)$):** The probability density function describing how features are distributed _given_ that the true underlying state is class $c_k$.
    
- **Posterior Probabilities ($P(y = c_k \mid \mathbf{x})$):** The updated probability that the true state is $c_k$ _after_ observing the feature vector $\mathbf{x}$. This is calculated rigorously using **Bayes' Rule**:
    
    $$P(y = c_k \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y = c_k) P(y = c_k)}{P(\mathbf{x})} = \frac{P(\mathbf{x} \mid y = c_k) P(y = c_k)}{\sum_{j} P(\mathbf{x} \mid y = c_j) P(y = c_j)}$$
    

## 3. Minimizing Expected Risk & The Bayes Optimal Classifier

### The Loss Function Matrix

Let $\alpha_i$ represent the action of predicting class $c_i$, and let $c_j$ represent the true underlying class. We define a generalized loss function $\lambda(\alpha_i \mid c_j)$, which specifies the scalar cost incurred for predicting class $c_i$ when the true class is $c_j$.

### Conditional Risk

Given a specific observed feature vector $\mathbf{x}$, the expected loss (or **conditional risk**) associated with taking prediction action $\alpha_i$ is computed by taking the expected value over the posterior class distributions:

$$R(\alpha_i \mid \mathbf{x}) = \sum_{j=1}^{K} \lambda(\alpha_i \mid c_j) P(y = c_j \mid \mathbf{x})$$

### The Bayes Decision Rule

To achieve global optimal performance, our strategy must minimize the conditional risk for every single observed input $\mathbf{x}$. The **Bayes Optimal Classifier** $h^*(\mathbf{x})$ is defined mathematically as:

$$h^*(\mathbf{x}) = \arg\min_{\alpha_i} R(\alpha_i \mid \mathbf{x})$$

### The Zero-One Loss Function (Classification Error)

Under the standard $0\text{-}1$ loss function, zero cost is assigned to correct classifications, and an equal unit cost of $1$ is assigned to any misclassification:

$$\lambda(\alpha_i \mid c_j) = \begin{cases} 0 & \text{if } i = j \\ 1 & \text{if } i \neq j \end{cases}$$

Substituting the $0\text{-}1$ loss function into the conditional risk equation yields:

$$R(\alpha_i \mid \mathbf{x}) = \sum_{j \neq i} 1 \cdot P(y = c_j \mid \mathbf{x}) = 1 - P(y = c_i \mid \mathbf{x})$$

To minimize this risk, we must select the class $\alpha_i$ that maximizes the posterior probability. Therefore, under $0\text{-}1$ loss, the Bayes Optimal Classifier simplifies to the **Maximum A Posteriori (MAP)** decision rule:

$$h^*(\mathbf{x}) = \arg\max_{c_k} P(y = c_k \mid \mathbf{x})$$

## 4. Analytical Decision Boundaries: Two-Class Gaussian Setting

Let us evaluate a binary classification problem ($\mathcal{Y} = \{c_1, c_2\}$) where the class-conditional feature distributions are modeled as multivariate Gaussian distributions:

$$P(\mathbf{x} \mid y = c_1) \sim \mathcal{N}(\boldsymbol{\mu}_1, \Sigma_1) \quad \text{and} \quad P(\mathbf{x} \mid y = c_2) \sim \mathcal{N}(\boldsymbol{\mu}_2, \Sigma_2)$$

The MAP decision rule chooses class $c_1$ if:

$$P(y = c_1 \mid \mathbf{x}) > P(y = c_2 \mid \mathbf{x}) \iff P(\mathbf{x} \mid y = c_1)P(y = c_1) > P(\mathbf{x} \mid y = c_2)P(y = c_2)$$

Taking the natural logarithm of both sides gives the equivalent log-odds decision rule. We classify an input as class $c_1$ if the following inequality holds:

$$\ln P(\mathbf{x} \mid y = c_1) + \ln P(y = c_1) > \ln P(\mathbf{x} \mid y = c_2) + \ln P(y = c_2)$$

### Case 1: Identical Covariance Structures ($\Sigma_1 = \Sigma_2 = \Sigma$)

When the two classes share an identical covariance structure, the non-linear quadratic terms in the Gaussian expansion cancel out perfectly. Expanding the log-densities results in a **Linear Discriminant Function**:

$$\mathbf{x}^T \Sigma^{-1}\boldsymbol{\mu}_1 - \frac{1}{2}\boldsymbol{\mu}_1^T \Sigma^{-1}\boldsymbol{\mu}_1 + \ln P(y = c_1) > \mathbf{x}^T \Sigma^{-1}\boldsymbol{\mu}_2 - \frac{1}{2}\boldsymbol{\mu}_2^T \Sigma^{-1}\boldsymbol{\mu}_2 + \ln P(y = c_2)$$

Grouping terms relative to the variable $\mathbf{x}$ isolates a strictly **linear decision boundary** equation ($\mathbf{w}^T \mathbf{x} + b = 0$):

$$\mathbf{w} = \Sigma^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$$

$$b = -\frac{1}{2}\boldsymbol{\mu}_1^T \Sigma^{-1}\boldsymbol{\mu}_1 + \frac{1}{2}\boldsymbol{\mu}_2^T \Sigma^{-1}\boldsymbol{\mu}_2 + \ln \frac{P(y = c_1)}{P(y = c_2)}$$

### Case 2: Arbitrary Covariance Structures ($\Sigma_1 \neq \Sigma_2$)

When the covariance matrices differ, the quadratic terms do not cancel. The resulting decision boundary becomes a hyper-quadratic surface (such as a parabola, ellipse, or hyperbola). This formulation is known as **Quadratic Discriminant Analysis (QDA)**:

$$-\frac{1}{2}\ln|\Sigma_1| - \frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_1)^T \Sigma_1^{-1}(\mathbf{x}-\boldsymbol{\mu}_1) + \ln P(y = c_1) > -\frac{1}{2}\ln|\Sigma_2| - \frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_2)^T \Sigma_2^{-1}(\mathbf{x}-\boldsymbol{\mu}_2) + \ln P(y = c_2)$$

## 5. Python / NumPy Simulation Snippet

This functional block illustrates how to construct an analytical MAP decision engine for binary Gaussian data sharing a common covariance matrix:

```python
import numpy as np

class BayesOptimalLinearClassifier:
    def __init__(self, mu1, mu2, sigma, prior_c1=0.5):
        self.mu1 = np.array(mu1).reshape(-1, 1)
        self.mu2 = np.array(mu2).reshape(-1, 1)
        self.sigma = np.array(sigma)
        self.prior_c1 = prior_c1
        self.prior_c2 = 1.0 - prior_c1
        
        # Precompute the linear hyperplane parameters
        self.sigma_inv = np.linalg.inv(self.sigma)
        
        # w = Sigma^-1 * (mu1 - mu2)
        self.w = np.dot(self.sigma_inv, (self.mu1 - self.mu2))
        
        # b = -0.5*mu1^T*Sigma^-1*mu1 + 0.5*mu2^T*Sigma^-1*mu2 + ln(P(c1)/P(c2))
        term1 = -0.5 * np.dot(self.mu1.T, np.dot(self.sigma_inv, self.mu1))
        term2 = 0.5 * np.dot(self.mu2.T, np.dot(self.sigma_inv, self.mu2))
        term3 = np.log(self.prior_c1 / self.prior_c2)
        self.b = (term1 + term2 + term3)[0, 0]

    def predict(self, x):
        """
        x is a 1D or 2D array of features. Shape: (d, 1)
        """
        x = np.array(x).reshape(-1, 1)
        # Check linear discriminant score: w^T * x + b
        score = np.dot(self.w.T, x)[0, 0] + self.b
        return 1 if score > 0 else 2
```

## 6. Solved Exam-Style Examples

### Example 1: Asymmetric Loss Boundaries in a Continuous 1D Space

**Problem Statement:** Consider a 1D continuous feature space with two classes, $c_1$ and $c_2$, that are uniformly distributed over distinct intervals:

$$P(x \mid y = c_1) \sim \mathcal{U}(0, 10) \quad \text{and} \quad P(x \mid y = c_2) \sim \mathcal{U}(5, 15)$$

The prior class fields are identical: $P(y = c_1) = P(y = c_2) = 0.5$.

Suppose the operational application imposes an **asymmetric loss structure**, where misclassifying an instance of class $c_1$ is far more costly than misclassifying class $c_2$:

$$\lambda(\alpha_1 \mid c_1) = 0, \quad \lambda(\alpha_2 \mid c_2) = 0$$

$$\lambda(\alpha_2 \mid c_1) = 4 \quad (\text{False Negative Cost})$$

$$\lambda(\alpha_1 \mid c_2) = 1 \quad (\text{False Positive Cost})$$

Derive the exact analytical decision boundaries that a Bayes optimal decision engine must enforce to minimize total expected risk.

#### Step-by-Step Analytical Derivation

1. Write out the explicit class-conditional probability density functions over their active intervals:
    
    $$P(x \mid y = c_1) = \begin{cases} \frac{1}{10} & \text{if } 0 \leq x \leq 10 \\ 0 & \text{otherwise} \end{cases}$$
    
    $$P(x \mid y = c_2) = \begin{cases} \frac{1}{10} & \text{if } 5 \leq x \leq 15 \\ 0 & \text{otherwise} \end{cases}$$
    
2. Analyze the decision space by partitions across the variable $x$:
    
    - For $0 \leq x < 5$: Only class $c_1$ exists ($P(x \mid y = c_2) = 0$). The only logical decision is to predict class $c_1$.
        
    - For $10 < x \leq 15$: Only class $c_2$ exists ($P(x \mid y = c_1) = 0$). The only logical decision is to predict class $c_2$.
        
3. Evaluate the overlapping conflict region where $5 \leq x \leq 10$. In this interval, both probability densities are active and equal to $\frac{1}{10}$.
    
4. Formulate the conditional risk functions for making each prediction inside the overlap zone:
    
    $$R(\alpha_1 \mid x) = \lambda(\alpha_1 \mid c_1)P(c_1 \mid x) + \lambda(\alpha_1 \mid c_2)P(c_2 \mid x) = 0 + 1 \cdot P(c_2 \mid x)$$
    
    $$R(\alpha_2 \mid x) = \lambda(\alpha_2 \mid c_1)P(c_1 \mid x) + \lambda(\alpha_2 \mid c_2)P(c_2 \mid x) = 4 \cdot P(c_1 \mid x) + 0$$
    
5. Using Bayes' rule, because the priors and conditional densities are identical inside this overlap zone, the posterior probabilities are equal:
    
    $$P(c_1 \mid x) = P(c_2 \mid x) = 0.5$$
    
6. Substitute these posteriors into the conditional risk expressions:
    
    $$R(\alpha_1 \mid x) = 1 \cdot 0.5 = 0.5$$
    
    $$R(\alpha_2 \mid x) = 4 \cdot 0.5 = 2.0$$
    
7. Compare the conditional risks:
    
    $$R(\alpha_1 \mid x) = 0.5 < 2.0 = R(\alpha_2 \mid x)$$
    
    Since predicting class $c_1$ yields lower conditional risk throughout the entire overlap region, the optimal strategy is to predict class $c_1$ for all $x \in [5, 10]$.
    

**Final Exam Answer:** The Bayes optimal classifier must assign predictions across the domain partitions as follows:

$$h^*(x) = \begin{cases} c_1 & \text{if } 0 \leq x \leq 10 \\ c_2 & \text{if } 10 < x \leq 15 \end{cases}$$

### Example 2: Finding the Threshold Point for 1D Gaussians

**Problem Statement:** Let two 1D classes share an identical variance scalar $\sigma^2 = 1$, with separated means $\mu_1 = 0$ and $\mu_2 = 4$. The class priors are skewed such that class $c_1$ is twice as likely as class $c_2$: $P(y = c_1) = \frac{2}{3}$ and $P(y = c_2) = \frac{1}{3}$.

Assuming a standard $0\text{-}1$ loss criteria, find the exact coordinate threshold $x^*$ where the decision boundary resides.

#### Step-by-Step Analytical Derivation

1. Set up the log-odds equality for 1D Gaussian distributions to find the crossing threshold $x^*$:
    
    $$-\frac{(x^* - \mu_1)^2}{2\sigma^2} + \ln P(c_1) = -\frac{(x^* - \mu_2)^2}{2\sigma^2} + \ln P(c_2)$$
    
2. Substitute the given parameters ($\mu_1 = 0, \mu_2 = 4, \sigma^2 = 1$):
    
    $$-\frac{(x^*)^2}{2} + \ln\left(\frac{2}{3}\right) = -\frac{(x^* - 4)^2}{2} + \ln\left(\frac{1}{3}\right)$$
    
3. Isolate the logarithmic prior terms onto one side:
    
    $$\ln\left(\frac{2/3}{1/3}\right) = \frac{(x^*)^2}{2} - \frac{(x^* - 4)^2}{2}$$
    
    $$\ln(2) = \frac{(x^*)^2 - ((x^*)^2 - 8x^* + 16)}{2}$$
    
4. Simplify the algebraic expression:
    
    $$\ln(2) = \frac{8x^* - 16}{2} \implies \ln(2) = 4x^* - 8$$
    
5. Solve for the threshold variable $x^*$:
    
    $$4x^* = 8 + \ln(2) \implies x^* = 2 + \frac{\ln(2)}{4} \approx 2 + 0.1733 = 2.1733$$
    

**Final Exam Answer:** The analytical decision boundary threshold coordinate is exactly **$x^* = 2 + \frac{\ln(2)}{4}$** (approximately **$2.1733$**). Because class $c_1$ has a larger prior probability, the decision boundary is shifted away from the midpoint ($x=2$) toward the right, expanding the decision region for class $c_1$.

---
## 🔗 Navigation
**Previous:** [[ ]] | **Next:** [[ ]]