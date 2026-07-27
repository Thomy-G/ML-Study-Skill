---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Summary
date_added: 2026-06-21
---
# Topic 17 — PAC Learning for Infinite Spaces & VC Dimension (Syllabus Item 3)

**MOC:** [[Machine learning MOC]]
**Course:** [[English]]

Let us bridge the most critical theoretical and algorithmic gaps remaining in your syllabus checklist. We will tackle **VC Dimension and Infinite Hypothesis Spaces (Syllabus Item 3)** and **Support Vector Machines (SVMs) & Maximum-Margin Classification (Syllabus Items 5 & 6)**.

# Study Guide: Topic 17 — PAC Learning for Infinite Spaces & VC Dimension (Syllabus Item 3)

## 1. Context: Moving Beyond Finite Hypothesis Classes

In Topic 5, we derived sample complexity bounds under the strict assumption that our hypothesis space $\mathcal{H}$ was finite ($|\mathcal{H}| < \infty$). However, most practical machine learning models utilize continuous parameters (e.g., linear classifiers $\mathbf{w}^T\mathbf{x} + b \geq 0$), which means the size of the hypothesis space is uncountably infinite ($|\mathcal{H}| = \infty$).

To prove whether infinite spaces are PAC learnable and to derive their sample complexity bounds, we need a geometric metric that measures the _expressivity_ or capacity of an infinite class. This metric is the **Vapnik-Chervonenkis (VC) Dimension**.

## 2. Shattering and the VC Dimension

### Dichotomy

Let $S = \{\mathbf{x}_1, \dots, \mathbf{x}_k\}$ be a set of $k$ unlabeled data points. A _dichotomy_ is a specific assignment of binary labels ($\{0, 1\}$ or $\{-1, +1\}$) to the points in $S$. For $k$ points, there are exactly $2^k$ possible unique label assignments.

### Shattering

A hypothesis class $\mathcal{H}$ **shatters** a finite set of points $S$ if, for _every possible_ dichotomy of binary labels assigned to $S$, there exists a hypothesis $h \in \mathcal{H}$ that perfectly classifies those points without making a single mistake. That is, $\mathcal{H}$ restricted to $S$ can generate all $2^k$ label configurations.

### VC Dimension Definition

The **VC Dimension** of a hypothesis class $\mathcal{H}$, denoted as $\text{VCDim}(\mathcal{H})$, is the **maximum size $k$** of a set of data points $S$ that can be completely shattered by $\mathcal{H}$.

- To prove $\text{VCDim}(\mathcal{H}) \geq k$: You must show that there _exists_ at least one configuration of $k$ points that $\mathcal{H}$ can shatter.
    
- To prove $\text{VCDim}(\mathcal{H}) < k+1$: You must prove that _no matter how_ you pick $k+1$ points in the input space, $\mathcal{H}$ can never shatter them (i.e., there is always at least one dichotomy that causes the class to fail).
    

## 3. Generalization Bound & Sample Complexity

The fundamental theorem of PAC learning states that a hypothesis class $\mathcal{H}$ is PAC learnable if and only if its VC dimension is finite ($\text{VCDim}(\mathcal{H}) = d < \infty$).

Under the Agnostic PAC learning framework, if $\text{VCDim}(\mathcal{H}) = d$, then with a confidence of at least $1-\delta$, the generalization error of the empirical risk minimizer ($\hat{h}$) is bounded by:

$$Err_{\mathcal{D}}(\hat{h}) \leq Err_{S}(\hat{h}) + \mathcal{O}\left( \sqrt{\frac{d \ln(n/d) + \ln(1/\delta)}{n}} \right)$$

This yields the fundamental **Sample Complexity** requirement for infinite spaces:

$$n \geq \mathcal{O}\left( \frac{1}{\epsilon^2} \left( d \ln\left(\frac{1}{\epsilon}\right) + \ln\left(\frac{1}{\delta}\right) \right) \right)$$

## 4. Solved Exam Proof: VC Dimension of 1D Intervals

**Problem Statement:** Let the input space be the real continuous line $\mathcal{X} = \mathbb{R}$. Let the hypothesis class be the set of all closed 1D intervals: $\mathcal{H} = \{ h_{a,b}(x) = \mathbb{I}\{x \in [a, b]\} \mid a, b \in \mathbb{R}, a \leq b \}$. Prove that $\text{VCDim}(\mathcal{H}) = 2$.

### Step-by-Step Analytical Derivation

1. **Prove $\text{VCDim}(\mathcal{H}) \geq 2$:** We must find 2 points that can be shattered. Let $S = \{1, 3\}$. We test all $2^2 = 4$ dichotomies:
    
    - Labels $(0,0)$: Set interval $[5, 6]$. Neither point is inside. Correct.
        
    - Labels $(1,0)$: Set interval $[0, 2]$. Point 1 is inside, 3 is outside. Correct.
        
    - Labels $(0,1)$: Set interval $[2, 4]$. Point 1 is outside, 3 is inside. Correct.
        
    - Labels $(1,1)$: Set interval $[0, 4]$. Both points are inside. Correct.
        
        Since all 4 dichotomies can be realized, $\mathcal{H}$ shatters 2 points, so $\text{VCDim}(\mathcal{H}) \geq 2$.
        
2. **Prove $\text{VCDim}(\mathcal{H}) < 3$:** Let us take any 3 arbitrary points on the real line sorted in ascending order: $x_1 < x_2 < x_3$.
    
    Consider the specific dichotomy where the outer points are positive and the middle point is negative: $y_1 = 1, y_2 = 0, y_3 = 1$.
    
    To correctly classify $x_1$ and $x_3$ as positive, our closed interval $[a, b]$ must satisfy $a \leq x_1$ and $b \geq x_3$. This means the interval covers the entire span from $x_1$ to $x_3$. Because $x_1 < x_2 < x_3$, the intermediate point $x_2$ is mathematically forced to lie inside $[a, b]$, meaning $h_{a,b}(x_2)$ _must_ equal $1$.
    
    Therefore, it is physically impossible to label $x_2$ as $0$ while keeping $x_1$ and $x_3$ as $1$. This configuration can never be realized.
    

**Final Exam Answer:** Since a subset of 2 points can be shattered but no configuration of 3 points can ever be shattered, **$\text{VCDim}(\mathcal{H}) = 2$**.



---
## 🔗 Navigation
**Previous:** [[ ]] | **Next:** [[ ]]