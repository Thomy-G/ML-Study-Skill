---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Summary
date_added: 2026-06-21
---
# Topic 10 — Decision Trees & Information Gain (Class 06B)

**MOC:** [[Machine learning MOC]]
**Course:** [[English]]

# Study Guide: Topic 10 — Decision Trees & Information Gain (Class 06B)

## 1. Context: Non-parametric Discriminative Learning

Until now, our classification models restricted decision boundaries to linear surfaces (hyperplanes) or pre-mapped hyper-quadratic manifolds. **Decision Trees** represent a non-parametric, hierarchical discriminative learning framework that partitions the feature space into axis-aligned rectangular regions.

Instead of computing weighted continuous combinations of features simultaneously, a decision tree makes sequence-based, discrete categorical inquiries about individual attributes. The leaf nodes contain the final categorical class predictions.

## 2. Measuring Disorder: Entropy and Impurity

To construct an optimal tree top-down, we need a mathematical metric to evaluate the quality of a data split. We define metrics that quantify the "disorder" or **impurity** of a set of samples.

Let $S$ be a dataset containing examples from $K$ discrete classes. Let $p_k$ be the empirical probability (proportion) of instances belonging to class $k$ within $S$:

$$p_k = \frac{|\{\mathbf{x} \in S \mid y = k\}|}{|S|}$$

### A. Shannon Entropy

Entropy measures the average informational uncertainty inherent to the distribution of labels in the sample set $S$:

$$H(S) = -\sum_{k=1}^K p_k \log_2(p_k)$$

- If a node is completely **pure** (all examples belong to one single class, $p_1 = 1$), then $H(S) = -1 \log_2(1) = 0$.
    
- If a node is completely **impure** (examples are uniformly split across $K$ classes, $p_k = \frac{1}{K}$), entropy reaches its absolute maximum: $H(S) = \log_2(K)$.
    

### B. Gini Impurity

An alternative metric often utilized in tree algorithms (such as CART) to measure the probability of misclassifying a randomly chosen element from the set:

$$I_G(S) = 1 - \sum_{k=1}^K p_k^2$$

## 3. Information Gain (The ID3 Feature Selection Heuristic)

When choosing which feature $A$ to branch on at a given node, the **ID3 algorithm** calculates the **Information Gain**. This metric measures the expected reduction in entropy achieved by partitioning the dataset $S$ according to the distinct values of attribute $A$.

Let $A$ be an attribute that takes on values $v \in \text{Values}(A)$. Let $S_v$ be the subset of $S$ for which attribute $A$ has value $v$:

$$S_v = \{\mathbf{x} \in S \mid x_A = v\}$$

The Information Gain $\text{Gain}(S, A)$ is defined as:

$$\text{Gain}(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$

The algorithm evaluates all available candidate features and selects the attribute that maximizes Information Gain:

$$A^* = \arg\max_{A} \text{Gain}(S, A)$$

## 4. The ID3 Tree-Building Algorithm (TDIDT)

ID3 relies on **Top-Down Induction of Decision Trees (TDIDT)**, a recursive greedy search strategy through the feature space.

### Core Pseudocode Framework

```
Function BuildTree(Data, Features):
    1. Create a Root node for the tree.
    2. If all instances in Data belong to the same class c:
           Return Root as a leaf node labeled with class c.
    3. If Features list is empty:
           Return Root as a leaf node labeled with the majority class in Data.
    4. Move through all candidate attributes and calculate Gain(Data, A).
    5. Let A_best be the attribute that maximizes Information Gain.
    6. Set the decision attribute of Root to A_best.
    7. For each distinct value v of A_best:
           Let Data_v be the subset of Data where A_best == v.
           If Data_v is empty:
               Add a leaf child node to Root labeled with the majority class in Data.
           Else:
               Add the subtree returned by BuildTree(Data_v, Features \ {A_best}) as a child branch.
    8. Return Root.
```

## 5. Overfitting and Remediations

Because decision trees can recursively split until every single training example resides in its own isolated pure leaf node, unconstrained trees have exceptionally high variance and overfit training samples severely.

### Structural Strategies to Limit Complexity: Pruning

- **Pre-Pruning (Early Stopping):** Terminates tree growth early during construction if a stopping criterion is met:
  * Node sample count drops below a minimum threshold (e.g., $n < 5$).
  * Tree depth reaches a maximum predefined limit $D_{\text{max}}$.
  * Maximum Information Gain $\text{IG}(S, A)$ drops below a threshold $\epsilon$.

- **Post-Pruning (Cost-Complexity Pruning):** Grows a full unpruned tree $T_0$ to zero training error, then prunes back subtrees using a mathematical cost-complexity objective:

  $$\mathcal{R}_\alpha(T) = R(T) + \alpha |T|$$

  *   $R(T) = \sum_{m=1}^{|T|} r_m(T)$ is the total misclassification rate / empirical risk of tree $T$ on training data.
  *   $|T|$ is the number of terminal leaf nodes in subtree $T$.
  *   $\alpha \ge 0$ is the **complexity parameter** (regularization hyperparameter):
      *   If $\alpha = 0$, no leaves are penalized, returning the full unpruned overfitted tree.
      *   As $\alpha \rightarrow \infty$, leaf capacity is infinitely penalized, collapsing the tree down to a single root node.
  *   *Algorithm:* Sequentially collapse internal nodes that produce the smallest increase in $R(T)$ per removed leaf, generating a sequence of nested subtrees $T_0 \supset T_1 \supset T_2 \dots$, and select the optimal tree $T_\alpha$ using cross-validation.
    

## 6. Python / NumPy Implementation Snippet

This functional block demonstrates how to calculate base entropy and isolate the best categorical attribute based on Information Gain metrics:

```python
import numpy as np

def compute_entropy(labels):
    """
    Computes Shannon Entropy for a 1D array of categorical labels.
    """
    unique_elements, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    # H(S) = -sum(p_k * log2(p_k))
    return -np.sum(probabilities * np.log2(probabilities))

def compute_information_gain(data_features, labels, attribute_index):
    """
    Calculates Information Gain achieved by splitting on a specific attribute column.
    """
    total_entropy = compute_entropy(labels)
    n_samples = len(labels)
    
    # Isolate distinct feature values for the chosen attribute
    feature_column = data_features[:, attribute_index]
    values, counts = np.unique(feature_column, return_counts=True)
    
    # Compute weighted conditional entropy sum
    conditional_entropy = 0.0
    for val, count in zip(values, counts):
        subset_labels = labels[feature_column == val]
        weight = count / n_samples
        conditional_entropy += weight * compute_entropy(subset_labels)
        
    return total_entropy - conditional_entropy
```

## 7. Solved Exam-Style Examples

### Example 1: Executing an ID3 Split Trace

**Problem Statement:** You are given a small training dataset consisting of 14 samples evaluating whether to play tennis based on an outdoor category feature named `Outlook` (which takes values `Sunny`, `Overcast`, or `Rain`). The overall prior distribution of the target class label `Play` is 9 `Yes` and 5 `No`.

The categorical distribution for the `Outlook` feature subsets is structured as follows:

- `Sunny`: 2 `Yes`, 3 `No`
    
- `Overcast`: 4 `Yes`, 0 `No`
    
- `Rain`: 3 `Yes`, 2 `No`
    

Calculate the exact analytical Information Gain value for the `Outlook` feature attribute.

#### Step-by-Step Analytical Derivation

1. Compute the base root node entropy $H(S)$ prior to any splitting:
    
    $$H(S) = -\left(\frac{9}{14}\log_2\left(\frac{9}{14}\right) + \frac{5}{14}\log_2\left(\frac{5}{14}\right)\right) \approx 0.9403$$
    
2. Compute the internal entropy for each separate branch value partition:
    
    - **Sunny ($S_{\text{Sunny}}$):** 5 samples total.
        
        $$H(S_{\text{Sunny}}) = -\left(\frac{2}{5}\log_2\left(\frac{2}{5}\right) + \frac{3}{5}\log_2\left(\frac{3}{5}\right)\right) \approx 0.9710$$
        
    - **Overcast ($S_{\text{Overcast}}$):** 4 samples total. This branch is completely pure.
        
        $$H(S_{\text{Overcast}}) = -\left(\frac{4}{4}\log_2\left(\frac{4}{4}\right) + 0\right) = 0$$
        
    - **Rain ($S_{\text{Rain}}$):** 5 samples total.
        
        $$H(S_{\text{Rain}}) = -\left(\frac{3}{5}\log_2\left(\frac{3}{5}\right) + \frac{2}{5}\log_2\left(\frac{2}{5}\right)\right) \approx 0.9710$$
        
3. Calculate the total weighted conditional entropy of the split:
    
    $$\sum_{v} \frac{|S_v|}{|S|} H(S_v) = \frac{5}{14}H(S_{\text{Sunny}}) + \frac{4}{14}H(S_{\text{Overcast}}) + \frac{5}{14}H(S_{\text{Rain}})$$
    
    $$= \frac{5}{14}(0.9710) + \frac{4}{14}(0) + \frac{5}{14}(0.9710) = \frac{10}{14}(0.9710) \approx 0.6936$$
    
4. Subtract the conditional entropy from the base entropy to find the Information Gain:
    
    $$\text{Gain}(S, \text{Outlook}) = 0.9403 - 0.6936 = 0.2467$$
    

**Final Exam Answer:** The analytical Information Gain achieved by splitting on the `Outlook` attribute is approximately **$0.2467$ bits**.

### Example 2: Expressive Limits of Decision Trees (The XOR Bound)

**Problem Statement:** Prove that a standard decision tree cannot classify the 2D binary XOR function with a depth of 1, but can achieve perfect separation at a depth of 2.

#### Step-by-Step Analytical Derivation

1. The XOR function inputs are $A \in \{0,1\}$ and $B \in \{0,1\}$. The target labels are $y = 1$ if $A \neq B$, and $y = 0$ if $A = B$. The dataset contains 4 points: $(0,0) \rightarrow 0$, $(1,1) \rightarrow 0$, $(1,0) \rightarrow 1$, $(0,1) \rightarrow 1$. The baseline base entropy is $H(S) = 1.0$.
    
2. Attempt a split at depth 1 using attribute $A$:
    
    - For $A = 0$, the points are $(0,0)$ [label 0] and $(0,1)$ [label 1]. The label distribution is exactly balanced (1 zero, 1 one), so $H(S_{A=0}) = 1.0$.
        
    - For $A = 1$, the points are $(1,1)$ [label 0] and $(1,0)$ [label 1]. The label distribution is also balanced, so $H(S_{A=1}) = 1.0$.
        
3. Compute Information Gain for attribute $A$:
    
    $$\text{Gain}(S, A) = 1.0 - \left(\frac{2}{4}(1.0) + \frac{2}{4}(1.0)\right) = 0.0$$
    
4. By symmetry, choosing attribute $B$ at depth 1 yields identical results: $\text{Gain}(S, B) = 0.0$. Since the Information Gain is zero for both attributes, a single split (depth 1) provides no predictive power.
    
5. Now consider a depth 2 tree. Branch on attribute $A$ at the root node. This creates two internal child nodes:
    
    - Left child node ($A = 0$): Contains points $(0,0)$ [label 0] and $(0,1)$ [label 1]. Now branch this node on attribute $B$:
        
        - Sub-branch $B = 0 \rightarrow$ Point $(0,0)$, label is purely $0$ (Entropy = 0).
            
        - Sub-branch $B = 1 \rightarrow$ Point $(0,1)$, label is purely $1$ (Entropy = 0).
            
    - Right child node ($A = 1$): Contains points $(1,0)$ [label 1] and $(1,1)$ [label 0]. Branch this node on attribute $B$:
        
        - Sub-branch $B = 0 \rightarrow$ Point $(1,0)$, label is purely $1$ (Entropy = 0).
            
        - Sub-branch $B = 1 \rightarrow$ Point $(1,1)$, label is purely $0$ (Entropy = 0).
            
6. Because every leaf node at depth 2 is perfectly pure, the total entropy drops to 0, achieving perfect classification accuracy.
    

**Final Exam Answer:** A depth-1 decision tree cannot solve the XOR problem because the individual linear projection of any single feature yields an Information Gain of **0**. A depth-2 tree resolves the problem by evaluating the attributes sequentially, capturing the non-linear interaction between features.

---
## 🔗 Navigation
**Previous:** [[ ]] | **Next:** [[ ]]