---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Summary
date_added: 2026-06-21
---
# Topic 20 — Generative Classifiers & Non-Parametric Neighborhoods (Syllabus Item 7)

**MOC:** [[Machine learning MOC]]
**Course:** [[English]]

# Study Guide: Topic 20 — Generative Classifiers & Non-Parametric Neighborhoods (Syllabus Item 7)

## 1. Context: Generative vs. Discriminative Frameworks

Most classifiers we reviewed (Logistic Regression, Perceptron, SVM, MLPs) are **discriminative**: they do not model how the data features are generated; instead, they focus entirely on mapping features to labels by constructing an explicit decision boundary $P(y \mid \mathbf{x})$.

**Generative models** (like Naïve Bayes) take the opposite path: they learn how features are distributed for each class by modeling the class-conditional probability distribution $P(\mathbf{x} \mid y)$ along with the class prior $P(y)$. Once learned, they evaluate predictions on new instances using Bayes' Rule to infer the final posterior probability.

## 2. The Naïve Bayes Classifier

The fundamental challenge with modeling the joint class-conditional distribution $P(\mathbf{x} \mid y) = P(x_1, x_2, \dots, x_d \mid y)$ is that the number of parameters scales exponentially with feature dimension $d$, leading to immediate overfitting.

### The Naïve Independence Assumption

The **Naïve Bayes model** simplifies this challenge by making a strong conditional independence assumption: **given the true class label $y$, all features are assumed to be completely independent of one another.**

Mathematically, this conditional independence assumption simplifies the joint probability into a product of individual 1D feature probabilities:

$$P(\mathbf{x} \mid y = c) = P(x_1, \dots, x_d \mid y = c) = \prod_{j=1}^d P(x_j \mid y = c)$$

### The Decision Rule

To classify an incoming feature vector $\mathbf{x}$, the model applies the Maximum A Posteriori (MAP) decision rule:

$$\hat{y} = \arg\max_{c} P(y = c \mid \mathbf{x}) = \arg\max_{c} \frac{P(\mathbf{x} \mid y = c)P(y = c)}{P(\mathbf{x})}$$

Since the marginal evidence $P(\mathbf{x})$ is identical for all classes, it can be safely ignored during maximization. Substituting the independence product yields the final Naïve Bayes classification rule:

$$\hat{y} = \arg\max_{c} \left[ P(y = c) \prod_{j=1}^d P(x_j \mid y = c) \right]$$

## 3. $k$-Nearest Neighbors ($k$-NN)

In contrast to parametric models that compress training data into a fixed set of weights, **$k$-Nearest Neighbors ($k$-NN)** is a instance-based, non-parametric algorithm that memorizes the entire training dataset.

### Algorithmic Execution Protocol

Given a novel test point $\mathbf{x}^*$:

1. Calculate the distance (typically Euclidean distance) from $\mathbf{x}^*$ to every single instance $\mathbf{x}_i$ in the training dataset:
    
    $$d(\mathbf{x}^*, \mathbf{x}_i) = \|\mathbf{x}^* - \mathbf{x}_i\|$$
    
2. Sort the distances in ascending order and isolate the indices of the $k$ closest training points. Let this neighborhood subset be $\mathcal{N}_k(\mathbf{x}^*)$.
    
3. **Classification Strategy:** Cast a majority vote across the labels in the neighborhood. Assign the test point to whichever class appears most frequently:
    
    $$\hat{y}^* = \arg\max_{c} \sum_{i \in \mathcal{N}_k(\mathbf{x}^*)} \mathbb{I}\{y_i = c\}$$
    

### The Impact of Hyperparameter $k$

- **Small $k$ (e.g., $k=1$):** Extremely complex decision boundaries that fit the training data perfectly. The model has **low bias** but **high variance**, making it highly sensitive to random noise or outliers.
    
- **Large $k$ (e.g., $k \rightarrow n$):** The decision boundary becomes smooth and flat, eventually predicting the global majority class everywhere. The model has **high bias** but **low variance**.
    

## 4. Python Implementation Snippet: $k$-NN Classification Engine

```python
import numpy as np

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # Non-parametric step: simply memorize the complete training corpus
        self.X_train = X
        self.y_train = y

    def predict_single(self, x_test):
        # 1. Compute Euclidean distance from the test point to all training rows
        distances = np.linalg.norm(self.X_train - x_test, axis=1)
        
        # 2. Extract indices of the k smallest distances
        k_nearest_indices = np.argsort(distances)[:self.k]
        
        # 3. Pull corresponding labels and perform majority vote
        neighbor_labels = self.y_train[k_nearest_indices]
        unique_classes, counts = np.unique(neighbor_labels, return_counts=True)
        
        return unique_classes[np.argmax(counts)]
```


---
## 🔗 Navigation
**Previous:** [[ ]] | **Next:** [[ ]]