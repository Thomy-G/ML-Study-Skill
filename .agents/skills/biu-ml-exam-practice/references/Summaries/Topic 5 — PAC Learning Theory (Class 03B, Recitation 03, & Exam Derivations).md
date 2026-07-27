---
type: uni_general
course: "[[Machine learning]]"
status: 🟢 Active
order: Summary
date_added: 2026-06-21
---
# Topic 5 — PAC Learning Theory (Class 03B, Recitation 03, & Exam Derivations)

**MOC:** [[Machine learning MOC]]
**Course:** [[Machine learning]]

# Study Guide: Topic 5 — PAC Learning Theory (Class 03B, Recitation 03, & Exam Derivations)

## 1. Theoretical Foundation: The Core Learning Question

Until now, we have derived empirical algorithms based on minimizing error over a finite dataset. **Probably Approximately Correct (PAC) Learning Theory** addresses the fundamental mathematical limits of machine learning:

- _Can we guarantee that a model trained on a finite sample will generalize well to unseen data?_
    
- _How many training samples ($n$) are mathematically necessary to provide this guarantee?_
    

### The Fundamental Quantities

1. **True Generalization Error / Risk ($Err_{\mathcal{D}}(h)$):** The expected $0\text{-}1$ loss of a hypothesis $h$ over the entire true, unknown joint distribution $\mathcal{D}$:
    
    $$Err_{\mathcal{D}}(h) = \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}} [\mathbb{I}\{h(\mathbf{x}) \neq y\}] = P_{(\mathbf{x}, y) \sim \mathcal{D}}(h(\mathbf{x}) \neq y)$$
    
2. **Empirical Error / Training Risk ($Err_{S}(h)$):** The average error computed directly over an observed data sample $S = \{(\mathbf{x}_1, y_1), \dots, \mathbf{x}_n, y_n\}$ containing $n$ i.i.d. instances:
    
    $$Err_{S}(h) = \frac{1}{n}\sum_{i=1}^n \mathbb{I}\{h(\mathbf{x}_i) \neq y_i\}$$
    

## 2. Mathematical Definition of PAC Learnability

A hypothesis class $\mathcal{H}$ is **PAC Learnable** if there exists an algorithm $A$ and a polynomial function $poly(1/\epsilon, 1/\delta, d)$ such that for any distribution $\mathcal{D}$, any target concept $c \in \mathcal{H}$, and any choice of parameters $\epsilon, \delta \in (0, 0.5)$:

Given a sample size $n \geq poly(1/\epsilon, 1/\delta, d)$, the algorithm outputs a hypothesis $\hat{h}$ satisfying:

$$P\Big( Err_{\mathcal{D}}(\hat{h}) \leq \epsilon \Big) \geq 1 - \delta$$

### Decoupling the Parameters

- **$\epsilon$ (Accuracy Parameter):** The maximum tolerable generalization error gap (**"Approximately Correct"**).
    
- **$\delta$ (Confidence Parameter):** The probability that the algorithm fails to find a good hypothesis due to a highly unrepresentative, misleading training sample (**"Probably"**).
    

## 3. Case 1: Finite, Realizable Hypothesis Class

### Assumptions

- **Finite $\mathcal{H}$:** The total number of unique candidate functions in our class is finite: $|\mathcal{H}| < \infty$.
    
- **Realizable Case:** The absolute true data-generating function $h^*$ lives inside our hypothesis class ($h^* \in \mathcal{H}$). This guarantees that the Empirical Risk Minimization (ERM) solution $\hat{h}$ achieves zero empirical error: $Err_{S}(\hat{h}) = 0$.
    

### Generalization Boundary Derivation

We want to bound the probability that a bad hypothesis (one where $Err_{\mathcal{D}}(h) > \epsilon$) manages to fool us by achieving an empirical error of $Err_{S}(h) = 0$ over a sample $S$ of size $n$.

1. For a specific bad hypothesis $h_{\text{bad}}$, the probability that it correctly classifies a single i.i.d. instance is at most $1 - \epsilon$.
    
2. The probability that $h_{\text{bad}}$ correctly classifies all $n$ independent training samples is:
    
    $$P(Err_{S}(h_{\text{bad}}) = 0) \leq (1 - \epsilon)^n$$
    
3. Using the inequality $1 - \epsilon \leq e^{-\epsilon}$:
    
    $$P(Err_{S}(h_{\text{bad}}) = 0) \leq e^{-n\epsilon}$$
    
4. To account for _any_ bad hypothesis within our class, we apply the **Union Bound** across all possible bad hypotheses (at most $|\mathcal{H}|$):
    
    $$P\big(\exists h \in \mathcal{H} \text{ s.t. } Err_{\mathcal{D}}(h) > \epsilon \text{ AND } Err_{S}(h) = 0\big) \leq |\mathcal{H}| e^{-n\epsilon}$$
    
5. To guarantee that our failure probability is at most $\delta$, we set this upper bound equal to $\delta$ and solve for $n$:
    
    $$|\mathcal{H}| e^{-n\epsilon} \leq \delta \implies e^{-n\epsilon} \leq \frac{\delta}{|\mathcal{H}|}$$
    
    $$-n\epsilon \leq \ln\left(\frac{\delta}{|\mathcal{H}|}\right) \implies n\epsilon \geq \ln\left(\frac{|\mathcal{H}|}{\delta}\right)$$
    
    $$n \geq \frac{1}{\epsilon}\ln\left(\frac{|\mathcal{H}|}{\delta}\right)$$
    

## 4. Case 2: Finite, Agnostic Hypothesis Class

### Assumptions

- **Agnostic Case:** The true target labeling function does _not_ necessarily belong to $\mathcal{H}$ ($h^* \notin \mathcal{H}$). The best possible empirical error may be strictly greater than zero ($Err_{S}(\hat{h}) > 0$).
    

### Hoeffding's Inequality & Generalization

Agnostic learning relies on **Hoeffding's Inequality**, which limits how far the empirical mean of independent random variables can deviate from their true mathematical expectation. For any single hypothesis $h \in \mathcal{H}$:

$$P\big(|Err_{S}(h) - Err_{\mathcal{D}}(h)| > \epsilon\big) \leq 2e^{-2n\epsilon^2}$$

Applying the Union Bound across the entire finite hypothesis class $\mathcal{H}$:

$$P\big(\exists h \in \mathcal{H} \text{ s.t. } |Err_{S}(h) - Err_{\mathcal{D}}(h)| > \epsilon\big) \leq 2|\mathcal{H}|e^{-2n\epsilon^2}$$

Setting this failure bound to $\delta$ isolates the required **Sample Complexity** formula for the agnostic case:

$$n \geq \frac{1}{2}\left(\frac{1}{\epsilon^2}\right)\ln\left(\frac{2|\mathcal{H}|}{\delta}\right)$$

## 5. Python Evaluation Simulation Snippet

This snippet validates PAC sample bounds under both the realizable and agnostic settings:

```python
import numpy as np

def calculate_pac_sample_size(hypothesis_space_size, epsilon, delta, setting="realizable"):
    """
    Computes the sample size required under PAC learning guarantees.
    """
    H_size = float(hypothesis_space_size)
    
    if setting == "realizable":
        # n >= (1 / epsilon) * ln(|H| / delta)
        n = (1.0 / epsilon) * np.log(H_size / delta)
    elif setting == "agnostic":
        # n >= (1 / (2 * epsilon^2)) * ln(2 * |H| / delta)
        n = (1.0 / (2 * (epsilon ** 2))) * np.log((2.0 * H_size) / delta)
    else:
        raise ValueError("Setting must be either 'realizable' or 'agnostic'")
        
    return int(np.ceil(n))
```

## 6. Solved Exam-Style Examples

### Example 1: Boolean Conjunction Over d Features

**Problem Statement:** Consider an input space $\mathcal{X} = \{0, 1\}^d$ consisting of $d$-dimensional boolean vectors. The hypothesis space $\mathcal{H}$ consists of all possible pure conjunctions of literals (e.g., $h(\mathbf{x}) = x_1 \land \neg x_3 \land x_4$).

Assuming the setting is completely **realizable**, derive the explicit formula for the sample size $n$ needed to ensure that an ERM classifier achieves a generalization error of at most $\epsilon=0.05$ with a confidence of at least $95\%$ ($\delta=0.05$).

#### Step-by-Step Analytical Derivation

1. **Determine the size of the hypothesis space $|\mathcal{H}|$:** Each feature $x_i$ can appear in a conjunction in one of three ways: as a positive literal ($x_i$), as a negated literal ($\neg x_i$), or it can be completely omitted from the expression. Therefore, for $d$ independent features, the total number of unique conjunctions is:
    
    $$|\mathcal{H}| = 3^d$$
    
2. **State the realizable sample complexity equation:**
    
    $$n \geq \frac{1}{\epsilon}\ln\left(\frac{|\mathcal{H}|}{\delta}\right)$$
    
3. **Substitute the value of $|\mathcal{H}|$ into the equation:**
    
    $$n \geq \frac{1}{\epsilon}\ln\left(\frac{3^d}{\delta}\right) = \frac{1}{\epsilon}\Big( d\ln(3) - \ln(\delta) \Big)$$
    
4. **Insert the numerical criteria ($\epsilon = 0.05$, $\delta = 0.05$):**
    
    $$n \geq \frac{1}{0.05}\Big( d\ln(3) - \ln(0.05) \Big) = 20 \cdot \Big( 1.0986d - (-2.9957) \Big)$$
    
    $$n \geq 21.97d + 59.91$$
    

**Final Exam Answer:** The exact sample size required is **$n \geq \frac{1}{\epsilon}\left(d\ln(3) + \ln\left(\frac{1}{\delta}\right)\right)$**. For the given parameters, this simplifies to approximately **$n \geq \lceil 21.97d + 59.91 \rceil$** samples, showing that sample complexity scales linearly with feature dimensionality $d$.

### Example 2: Uniform Boundary Estimation (Continuous Domain)

**Problem Statement:** Let our input domain be the continuous real line $\mathcal{X} = [0, L]$. The true target concept is an unknown threshold interval $[0, \theta^*]$, where any point $x \leq \theta^*$ is labeled $1$ and any point $x > \theta^*$ is labeled $0$. Our hypothesis space is defined as $\mathcal{H} = \{[0, \theta] \mid 0 \leq \theta \leq L\}$. Assume a uniform data distribution over the domain: $x \sim \mathcal{U}(0, L)$.

Given a training sample $S$, the learning algorithm selects the tightest possible envelope fitting the positive examples: $\hat{\theta} = \max_{x_i \in S, y_i = 1} x_i$.

Prove that if the sample size satisfies $n \geq \frac{1}{\epsilon}\ln\left(\frac{1}{\delta}\right)$, the generalization error will be bounded by $Err_{\mathcal{D}}(\hat{h}) \leq \epsilon$ with a confidence of at least $1-\delta$.

#### Step-by-Step Analytical Derivation

1. **Analyze the error region:** Because $\hat{\theta}$ is defined as the maximum positive training sample, it can never exceed the true parameter $\theta^*$ ($\hat{\theta} \leq \theta^*$). The generalization error occurs only when a new sample falls into the unmapped gap between $\hat{\theta}$ and $\theta^*$. The size of this error zone is exactly $\theta^* - \hat{\theta}$.
    
2. **Define the failure condition:** A generalization failure ($Err_{\mathcal{D}}(\hat{h}) > \epsilon$) happens if and only if the length of this unmapped gap is greater than $\epsilon \cdot L$. This means that during training, _not a single sample_ landed inside the interval $(\theta^* - \epsilon L, \theta^*]$.
    
3. **Calculate the probability of failure for a single point:** For a uniformly distributed variable $x \sim \mathcal{U}(0, L)$, the probability that a single sample misses this interval is:
    
    $$P(\text{Miss}) = 1 - \frac{\text{Length of Interval}}{\text{Total Length}} = 1 - \frac{\epsilon L}{L} = 1 - \epsilon$$
    
4. **Calculate the probability that all $n$ independent samples miss the interval:**
    
    $$P(Err_{\mathcal{D}}(\hat{h}) > \epsilon) = (1 - \epsilon)^n \leq e^{-n\epsilon}$$
    
5. **Enforce the confidence bound $\delta$:**
    
    $$e^{-n\epsilon} \leq \delta \implies -n\epsilon \leq \ln(\delta) \implies n \geq \frac{1}{\epsilon}\ln\left(\frac{1}{\delta}\right)$$
    

**Final Exam Answer:** We have proven that the generalization error satisfies $P(Err_{\mathcal{D}}(\hat{h}) > \epsilon) \leq \delta$ provided that **$n \geq \frac{1}{\epsilon}\ln\left(\frac{1}{\delta}\right)$**.

---
## 🔗 Navigation
**Previous:** [[ ]] | **Next:** [[ ]]