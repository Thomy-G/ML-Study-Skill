---
type: uni_general
course: "[[Machine learning]]"
status: 🟢 Active
order: Summary
date_added: 2026-06-21
---
# Topic 2 — The Machine Learning Workflow & Data Splitting (Class 1A, Recitation 1)

**MOC:** [[Machine learning MOC]]
**Course:** [[Machine learning]]

# Study Guide: Topic 2 — The Machine Learning Workflow & Data Splitting (Class 1A, Recitation 1)

---

## 1. The Core Components of an ML Problem

An engineering problem requires three distinct, foundational components to be framed as a machine learning task:

1. 
**The Pattern / Relationship:** A systematic underlying relationship exists between the input data and target labels, but it cannot be easily formalized or captured via a deterministic, hand-coded rule-based algorithm.


2. **No Analytical Formula:** There is no known closed-form mathematical expression or direct physical simulation that maps inputs to outputs efficiently.
3. 
**The Availability of Data:** A historical corpus of data examples exists from which a statistical model can interactively learn or approximate this pattern.



---

## 2. Formal Mathematical Setting

### Data Spaces and Random Variables

* 
**Input Space ($\mathcal{X}$):** The domain of all possible input feature vectors, typically represented as a multi-dimensional real vector space $\mathcal{X} \subseteq \mathbb{R}^d$. An individual data instance is a column vector $\mathbf{x} = [x_1, x_2, \dots, x_d]^T$.


* **Output/Label Space ($\mathcal{Y}$):** The range of target values.
* For **Binary Classification**: $\mathcal{Y} = \{0, 1\}$ or $\mathcal{Y} = \{-1, +1\}$.
* For **Multiclass Classification**: $\mathcal{Y} = \{1, 2, \dots, k\}$.
* For **Regression**: $\mathcal{Y} = \mathbb{R}$.


* 
**Data Generating Distribution ($P(\mathbf{X}, Y)$):** We assume there exists an unknown, static joint probability distribution over the product space $\mathcal{X} \times \mathcal{Y}$. Our training and testing instances are sampled independent and identically distributed (i.i.d.) from this distribution.



### The Hypothesis Class ($\mathcal{H}$)

A hypothesis class $\mathcal{H}$ is a pre-selected set of functions mapping inputs to outputs:


$$\mathcal{H} = \{ f_\mathbf{w}: \mathcal{X} \rightarrow \mathcal{Y} \mid \mathbf{w} \in \Omega \}$$

where $\mathbf{w}$ represents the parameters or weights of the model, and $\Omega$ represents the parameter space. Learning consists of navigating $\mathcal{H}$ to find a specific hypothesis function $f \in \mathcal{H}$ that minimizes error.

---

## 3. Generalization, Overfitting, and Underfitting

The critical goal of machine learning is **generalization**: the ability of a model to make accurate predictions on novel, unseen data samples drawn from the same underlying distribution as the training data.

```
  High ↑
       │          Underfitting                    Overfitting
       │         ┌──────────────┐              ┌──────────────┐
       │         │ High Train   │              │ Low Train    │
  E    │         │ Error        │              │ Error        │
  r    │         │ High Test    │              │ High Test    │
  r    │         │ Error        │              │ Error        │
  o    │         └──────────────┘              └──────────────┘
  r    │
       │             /─\                                /
       │            /   \─── Validation / Test Error ──/
       │           /                                  /
       │          /──────────────────────────────────/
       │         /
       │        /───────────── Training Error ───────┐
       │       /                                     └───────
       └─────────────────────────────────────────────────────────►
                                                      Model Complexity

```

### Quantifying Errors

* **Empirical Risk / Training Error ($L_{\text{train}}$):** The average error evaluated directly over the observed training dataset $\mathcal{D}_{\text{train}}$ containing $n$ samples:

$$L_{\text{train}}(f) = \frac{1}{n} \sum_{i=1}^n \ell(f(\mathbf{x}_i), y_i)$$


where $\ell(\cdot, \cdot)$ is a foundational scalar loss function (e.g., $0\text{-}1$ loss for classification, squared error for regression).


* **True Risk / Generalization Error ($L_{\text{gen}}$):** The expected loss evaluated across the entire mathematical joint data distribution $P(\mathbf{X},Y)$:

$$L_{\text{gen}}(f) = \mathbb{E}_{(\mathbf{x}, y) \sim P} [\ell(f(\mathbf{x}), y)]$$



### Pathological Optimization States

* **Underfitting:** Occurs when the hypothesis class $\mathcal{H}$ is too restrictive, or training time is insufficient, to capture the underlying geometric structure of the data. Characteristics: **High training error** and **high test error**.
* **Overfitting:** Occurs when a model over-optimizes for the random noise, artifacts, or specific sample variations present in the training set rather than the true distribution properties. Characteristics: **Low training error** but **high test error**.

---

## 4. Rigorous Data Splitting Strategy

To measure generalizability safely without introducing severe optimistic evaluation biases, a dataset must be cleanly partitioned into non-overlapping subsets prior to modeling.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Complete Dataset                              │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
         ┌─────────────────────────┴────────────────────────┐
         ▼                                                  ▼
┌──────────────────────────────────────────────┐ ┌────────────────────────┐
│               Development Set                │ │        Test Set        │
└──────────────────────────────────────────────┘ └────────────────────────┘
         │                                                  │
   ┌─────┴─────────────┐                                    │
   ▼                   ▼                                    ▼
┌──────────────┐ ┌──────────────┐                     ┌──────────────────┐
│ Training Set │ │Validation Set│                     │ Final Evaluation │
└──────────────┘ └──────────────┘                     └──────────────────┘

```

### The Three Imperative Subsets

1. 
**Training Set (typically $60\%\text{–}80\%$):** The data optimization block used exclusively to calculate gradients and fit model structural parameters (weights $\mathbf{w}$ and biases).


2. **Validation Set (typically $10\%\text{–}20\%$):** Used to perform model selection, tune architectural hyperparameters (e.g., learning rate, regularization penalties, depth), and implement early stopping to prevent overfitting.
3. 
**Test Set (typically $10\%\text{–}20\%$):** Held back securely until the absolute final validation phase. **Crucial Rule:** The test set must never be exposed to the model during training, hyperparameter sweep selections, or feature engineering steps.



---

## 5. Python / NumPy Implementation Snippets

This comprehensive framework demonstrates how to compute an optimal random split from scratch using vector indexing arithmetic:

```python
import numpy as np

def train_val_test_split(X, y, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Splits feature matrix X and label array y into Train, Validation, and Test sets.
    """
    assert len(X) == len(y), "Features and labels must have matching sample dimensions"
    
    # Set seed for deterministic reproducibility across runs
    np.random.seed(seed)
    
    num_samples = len(X)
    shuffled_indices = np.random.permutation(num_samples)
    
    # Calculate index cut-off splits
    test_cutoff = int(num_samples * test_ratio)
    val_cutoff = test_cutoff + int(num_samples * val_ratio)
    
    # Slice indices arrays
    test_idx = shuffled_indices[:test_cutoff]
    val_idx = shuffled_indices[test_cutoff:val_cutoff]
    train_idx = shuffled_indices[val_cutoff:]
    
    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx]
    )

```

---

## 6. Solved Exam-Style Examples

### Example 1: Deducing Overfitting via Analytical Metric Signals

**Problem Statement:** You track the cross-entropy loss profile of a deep multi-layer perceptron over successive parameter epoch updates. At epoch 20, the training loss is $0.12$ and the validation loss is $0.15$. At epoch 100, the training loss drops significantly to $0.02$, but the validation loss climbs steadily back up to $0.45$.

Identify the generalization behavior of the model at epoch 100, and propose two explicit engineering remediations to counteract this failure mode.

#### Analytical Evaluation

1. **Diagnosis:** At epoch 100, the model exhibits a severe gap between a low training error ($0.02$) and a very high validation error ($0.45$). This is the classic signature of **overfitting**. The model has memorized high-frequency training variances instead of generalizing general conceptual patterns.


2. **Engineering Remediations:**
* *Early Stopping:* Terminate the parameter training loop dynamically at the historical inflection point where the validation metric stops improving and begins to degrade systematically (near epoch 20).
* 
*Regularization:* Introduce weight penalties (such as $L_2$ ridge weight decay) to structurally restrict the hypothesis capability scale of individual parameter components.





### Example 2: Mathematical Probability of Split Data Leakage

**Problem Statement:** A data scientist accidentally generates a synthetic dataset containing redundant, duplicated samples. Suppose your source dataset contains $n$ unique samples, but each sample is duplicated exactly once, resulting in a pool of $2n$ total items.

If you apply a standard random partition that uniformly assigns half of the samples ($n$ items) directly to a training partition and the remaining half ($n$ items) to a test validation partition, derive the mathematical expected probability that a specific unique sample present in your test partition is also simultaneously present in your training partition.

#### Step-by-Step Analytical Derivation

1. Consider a specific unique data entity, which consists of an indistinguishable twin pair: item $A_1$ and item $A_2$.
2. The target objective is to compute the conditional probability that a unique sample exists in both sets. This is equivalent to finding $1 - P(\text{Both items land in the same partition})$.
3. Let us calculate the total number of valid ways to partition the $2n$ items into two sets of size $n$:

$$\text{Total Configurations} = \binom{2n}{n}$$


4. Now, compute the number of configurations where *both* twins $A_1$ and $A_2$ are forced to reside inside the same set (e.g., both inside the Training partition). We choose the remaining $n-2$ slots from the leftover $2n-2$ samples:

$$\text{Configurations with both in Train} = \binom{2n-2}{n-2}$$


5. By symmetry, the number of ways both twins land together in the Test partition is identical: $\binom{2n-2}{n}$. Note that $\binom{2n-2}{n-2} = \binom{2n-2}{n}$.
6. Sum these together to get the probability that the twins are kept together in the same partition:

$$P(\text{Together}) = \frac{2 \cdot \binom{2n-2}{n}}{\binom{2n}{n}} = \frac{2 \cdot \frac{(2n-2)!}{n!(n-2)!}}{\frac{(2n)!}{n!n!}} = 2 \cdot \frac{(2n-2)!}{(2n)!} \cdot \frac{n!}{(n-2)!}$$


$$P(\text{Together}) = 2 \cdot \frac{1}{2n(2n-1)} \cdot n(n-1) = \frac{2n(n-1)}{2n(2n-1)} = \frac{n-1}{2n-1}$$


7. The probability that the duplicates are split across both training and test subsets (causing data leakage) is the complement:

$$P(\text{Separated / Leaked}) = 1 - P(\text{Together}) = 1 - \frac{n-1}{2n-1} = \frac{(2n-1) - (n-1)}{2n-1} = \frac{n}{2n-1}$$


8. For large datasets ($n \rightarrow \infty$):

$$\lim_{n \rightarrow \infty} \frac{n}{2n-1} = \frac{1}{2} = 50\%$$



**Final Exam Answer:** The analytical probability that a unique training sample leaks directly across the partition line into the test evaluation space is **$\frac{n}{2n-1}$** (which converges exactly to **$50\%$** as $n$ becomes large). This shows why deduplication must occur *before* data splitting.

---

## 5. Classification Performance Metrics & Evaluation

In classification tasks (especially imbalanced datasets), raw Accuracy ($\frac{\text{Correct}}{\text{Total}}$) is highly misleading. We evaluate models using a **Confusion Matrix** and derived rate metrics.

### 5.1 The Confusion Matrix

| | **Predicted Positive ($\hat{y}=1$)** | **Predicted Negative ($\hat{y}=0$)** |
|---|---|---|
| **Actual Positive ($y=1$)** | **True Positive (TP)** | **False Negative (FN)** *(Type II Error)* |
| **Actual Negative ($y=0$)** | **False Positive (FP)** *(Type I Error)* | **True Negative (TN)** |

---

### 5.2 Key Classification Metrics

1. **Accuracy:**
   $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
   *(Misleading on imbalanced data; e.g. predicting all negative on a 99% negative dataset yields 99% accuracy).*

2. **Precision (Positive Predictive Value):**  
   Out of all instances predicted as positive, how many were actually positive?
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
   *(Crucial when False Positives are costly, e.g., Spam Filtering).*

3. **Recall (Sensitivity / True Positive Rate - TPR):**  
   Out of all actual positive instances, how many did the model catch?
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
   *(Crucial when False Negatives are dangerous, e.g., Disease Detection).*

4. **$F_1$ Score (Harmonic Mean of Precision and Recall):**  
   Balances Precision and Recall into a single metric. Uses the **Harmonic Mean** (which severely penalizes extreme trade-off imbalances):
   $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$

5. **Generalized $F_\beta$ Score:**  
   Allows weighting Recall $\beta$ times as important as Precision:
   $$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}$$
   * $\beta = 1$: Standard $F_1$ Score (equal weight).
   * $\beta = 2$: $F_2$ Score (gives higher weight to Recall).
   * $\beta = 0.5$: $F_{0.5}$ Score (gives higher weight to Precision).

---

## 🔗 Navigation
**Previous:** [[Topic 1 — Introduction to ML & Mathematical Foundations]] | **Next:** [[Topic 3 — Optimal Decisions & Bayesian Decision Theory]]