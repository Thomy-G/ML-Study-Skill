---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Summary
date_added: 2026-06-23
---
# Topic 24 — Advanced Mathematical Proofs & Derivations

**MOC:** [[Machine learning MOC]]
**Course:** [[English]]

This guide compiles supplementary advanced mathematical derivations and machine learning topics: **Dropout Regularization**, **Word Embeddings**, and **Constrained Optimization (Lagrangian Duality & Maximum Entropy)**.

---

## 1. Regularization Primitives: Dropout Mechanics

### 1.1 Context: The Overfitting Problem in Deep Networks
Deep Neural Networks (like deep MLPs, ConvNets, and Transformers) contain millions of trainable parameters. Because of this massive expressive capacity, these models are highly susceptible to severe overfitting (**high variance**). The hidden units can easily develop complex **co-adaptations**—a pathological state where a neuron only generates useful representations when specific other neurons are active simultaneously, essentially memorizing noisy training patterns instead of learning generalizable features.

While $L_2$ regularization (Weight Decay) restricts parameter magnitudes, **Dropout** is a powerful regularization technique specifically designed to break up these mutual neuron dependencies by structurally modifying the network topology during training.

### 1.2 Mathematical Formalization of Dropout
During the training phase, Dropout randomly deactivates (drops out) a subset of neurons in a layer with a pre-configured probability $p$ (e.g., $p=0.5$).

#### A. The Training Phase Forward Pass
Let $\mathbf{h} \in \mathbb{R}^K$ be the activation vector output of a hidden layer. For each training sample and each training forward pass:

1. Generate a random mask vector $\mathbf{m} \in \{0, 1\}^K$, where each element is an independent Bernoulli random variable:
   $$m_j \sim \text{Bernoulli}(1 - p)$$

2. Apply the mask element-wise to the hidden layer activations using the Hadamard product ($\odot$):
   $$\tilde{\mathbf{h}} = \mathbf{h} \odot \mathbf{m}$$

   > 💡 **Mathematical Note: The Hadamard Product ($\odot$)**  
   > The **Hadamard product** (also called element-wise multiplication) multiplies corresponding entries of two matrices or vectors of identical dimensions: $(\mathbf{h} \odot \mathbf{m})_j = h_j \cdot m_j$. Unlike standard matrix multiplication ($W \mathbf{x}$), the Hadamard product requires identical shapes, is commutative ($\mathbf{a} \odot \mathbf{b} = \mathbf{b} \odot \mathbf{a}$), and is executed via the `*` operator in NumPy/PyTorch.

3. Pass the regularized vector $\tilde{\mathbf{h}}$ to the next layer in the computational graph. Because a random half of the hidden units are silenced at each step, neurons are forced to learn robust features that are useful independently of their neighbors.

#### B. The Inference (Test) Phase Forward Pass
During inference, we want our predictions to be deterministic and stable, so **no units are dropped** ($\mathbf{m} = \mathbf{1}$). However, because all neurons are now active simultaneously, the total expected input sum flowing into the next layer would be higher than it was during training.

To maintain numerical scale alignment between phases, we must scale the static weights or activations down proportionally by the survival probability ($1-p$):
$$\mathbf{h}_{\text{inference}} = (1 - p) \cdot \mathbf{h}$$

#### C. Inverted Dropout (Modern PyTorch Implementation)
To avoid performing extra multiplications at inference time, modern frameworks use **Inverted Dropout**. This method scales the activations *up* during the training phase instead:
$$\tilde{\mathbf{h}}_{\text{training}} = \frac{\mathbf{h} \odot \mathbf{m}}{1 - p}$$

This ensures that at test time, the activations can be evaluated normally without any modification.

---

## 2. Word Embeddings & Semantic Vector Spaces

### 2.1 Context: Representing Discrete Tokens as Continuous Vectors
Machine learning architectures are fundamentally mathematical vector engines—they can only compute gradients and optimize loss surfaces over continuous real numbers. They cannot natively process raw, discrete categorical strings (words).

The simplest representation, **One-Hot Encoding**, maps each word to a sparse vector of vocabulary size $V$, where a single index contains a $1$ and all other elements are $0$. However, One-Hot vectors suffer from two major flaws:
1. **Catastrophic Dimensionality:** If a vocabulary contains 100,000 unique tokens, every individual word vector requires 100,000 dimensions ($W \in \mathbb{R}^{100,000}$), creating an immense parameter footprint.
2. **Semantic Orthogonality:** The inner dot product of any two distinct one-hot vectors is always exactly zero ($\mathbf{w}_{\text{cat}}^T \mathbf{w}_{\text{dog}} = 0$). This means the representation completely fails to capture semantic relationships; the word *"cat"* is mathematically just as distant from *"dog"* as it is from *"refrigerator"*.

**Word Embeddings** resolve this by learning a dense, low-dimensional vector space mapping $\mathbf{w} \in \mathbb{R}^d$ (where $d \ll V$, typically $d=300$) where the geometric proximity between vectors directly reflects their semantic similarity.

### 2.2 The Distributional Hypothesis & Word2Vec
Word embedding frameworks rely on the **Distributional Hypothesis**: *Words that appear in similar textual contexts tend to share similar semantic meanings.*

#### The Skip-Gram Architecture
The **Skip-Gram model** (a core component of Word2Vec) formalizes this by turning an unlabeled text corpus into a self-supervised prediction task. Given a target center word $w_t$, the network is trained to maximize the conditional probability of predicting its surrounding context words $w_{t+j}$ within a sliding window of size $C$.

#### The Probabilistic Formulation
For each word type in our vocabulary, the model learns two separate vector representations: an input vector $\mathbf{v}_w$ (when the word acts as a center word) and an output vector $\mathbf{u}_w$ (when it acts as a context word). The conditional probability of observing context word $c$ given center word $w$ is modeled using the Softmax function:
$$P(c \mid w) = \frac{\exp(\mathbf{u}_c^T \mathbf{v}_w)}{\sum_{j=1}^V \exp(\mathbf{u}_j^T \mathbf{v}_w)}$$

The total objective function maximizes the log-probability across the entire text corpus:
$$\max \sum_{t=1}^T \sum_{-C \leq j \leq C, j \neq 0} \ln P(w_{t+j} \mid w_t)$$

### 2.3 Geometric Vector Properties (Linear Subspace Analogies)
A powerful emergent property of dense word embedding spaces is that semantic relationships translate directly into linear algebraic vector displacements.

For example, the vector difference between the word vector for *"man"* and *"woman"* aligns nearly perfectly with the vector difference between *"king"* and *"queen"*:
$$\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$$

This vector alignment allows semantic relationships (such as gender transitions, verb tenses, or country-capital pairings) to be navigated using simple, unsupervised linear vector arithmetic.

---

## 3. Constrained Optimization, Lagrangian Duality & Maximum Entropy

### 3.1 Mathematical Formulation of the Lagrangian Function
In constrained optimization, we aim to minimize an objective function $f(\mathbf{x})$ subject to inequality constraints $g_i(\mathbf{x}) \le 0$ for $i = 1, \dots, m$ and equality constraints $h_j(\mathbf{x}) = 0$ for $j = 1, \dots, p$.

The **Lagrangian function** combines the objective and the constraints into a single unconstrained function using **Lagrange multipliers** $\boldsymbol{\lambda} \succeq 0$ (for inequality constraints) and $\boldsymbol{\nu}$ (for equality constraints):
$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f(\mathbf{x}) + \sum_{i=1}^m \lambda_i g_i(\mathbf{x}) + \sum_{j=1}^p \nu_j h_j(\mathbf{x})$$

Instead of applying an un-differentiable step-penalty (0 inside the allowed region, $\infty$ outside), the multipliers apply a smooth, dynamic penalty scaled by the degree of constraint violation.

### 3.2 Primal vs. Dual Formulation
* **Primal Problem:** The original constrained optimization problem:
  $$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{subject to } g_i(\mathbf{x}) \le 0, \: h_j(\mathbf{x}) = 0$$
* **Lagrangian Dual Function:** Defined as the infimum of the Lagrangian over $\mathbf{x}$:
  $$g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$$
  The dual function yields a lower bound on the optimal primal value $p^*$: $g(\boldsymbol{\lambda}, \boldsymbol{\nu}) \le p^*$ for all $\boldsymbol{\lambda} \succeq 0$.
* **Dual Optimization Problem:** We maximize the dual function to get the tightest lower bound:
  $$\max_{\boldsymbol{\lambda} \succeq 0, \boldsymbol{\nu}} g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \max_{\boldsymbol{\lambda} \succeq 0, \boldsymbol{\nu}} \min_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$$
  Under **strong duality** (which holds under convexity conditions like Slater's condition), the optimal value of the dual problem $d^*$ equals the optimal value of the primal problem $p^*$ ($d^* = p^*$).

### 3.3 The 4 Karush-Kuhn-Tucker (KKT) Conditions
For any constrained optimization problem with differentiable functions where strong duality holds, any primal-dual optimal point $(\mathbf{x}^*, \boldsymbol{\lambda}^*, \boldsymbol{\nu}^*)$ must satisfy the 4 KKT conditions:

1. **Stationarity:** The gradient of the Lagrangian with respect to $\mathbf{x}$ vanishes at the optimum:
   $$\nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}^*, \boldsymbol{\lambda}^*, \boldsymbol{\nu}^*) = \nabla f(\mathbf{x}^*) + \sum_{i=1}^m \lambda_i^* \nabla g_i(\mathbf{x}^*) + \sum_{j=1}^p \nu_j^* \nabla h_j(\mathbf{x}^*) = \mathbf{0}$$
2. **Primal Feasibility:** The optimal point satisfies all original constraints:
   $$g_i(\mathbf{x}^*) \le 0 \quad \forall i=1,\dots,m, \qquad h_j(\mathbf{x}^*) = 0 \quad \forall j=1,\dots,p$$
3. **Dual Feasibility:** Inequality multipliers are non-negative:
   $$\lambda_i^* \ge 0 \quad \forall i=1,\dots,m$$
4. **Complementary Slackness:** The product of each inequality multiplier and its constraint function is zero:
   $$\lambda_i^* g_i(\mathbf{x}^*) = 0 \quad \forall i=1,\dots,m$$
   *Implication:* If a constraint is strictly inactive ($g_i(\mathbf{x}^*) < 0$), its multiplier must be zero ($\lambda_i^* = 0$). If $\lambda_i^* > 0$, the constraint must be active ($g_i(\mathbf{x}^*) = 0$).

### 3.4 Application: The Principle of Maximum Entropy
The **Principle of Maximum Entropy** states that when choosing among probability distributions subject to known constraints, the least biased distribution is the one that maximizes Shannon Entropy:
$$H(\mathbf{p}) = -\sum_{i=1}^n p_i \ln(p_i)$$

#### Derivation for an Unbiased Discrete Distribution
Suppose we have an $n$-outcome discrete event (e.g., an $n$-sided die) with no prior information other than the probability normalization constraint:
$$\sum_{i=1}^n p_i = 1 \implies \left( \sum_{i=1}^n p_i \right) - 1 = 0$$

We formulate the Lagrangian function with multiplier $\lambda_0$:
$$\mathcal{L}(\mathbf{p}, \lambda_0) = -\sum_{i=1}^n p_i \ln(p_i) - \lambda_0 \left( \sum_{i=1}^n p_i - 1 \right)$$

To maximize entropy, we compute partial derivatives w.r.t. $p_i$ and set them to zero:
$$\frac{\partial \mathcal{L}}{\partial p_i} = -\left( \ln(p_i) + p_i \cdot \frac{1}{p_i} \right) - \lambda_0 = -\ln(p_i) - 1 - \lambda_0 = 0$$

Solving for $p_i$:
$$\ln(p_i) = -(1 + \lambda_0) \implies p_i = e^{-(1 + \lambda_0)}$$

Since $e^{-(1 + \lambda_0)}$ is a constant independent of $i$, all probabilities $p_i$ are identical ($p_i = C$). Applying the constraint $\sum_{i=1}^n p_i = 1$:
$$\sum_{i=1}^n C = n C = 1 \implies C = \frac{1}{n} \implies p_i = \frac{1}{n} \quad \forall i$$

This proves mathematically that without prior information, the Maximum Entropy distribution is the **Uniform Distribution**.

---

## 4. Final Curriculum Verification

Cross-referencing the syllabus checklist against our completed study guide modules:
1. **Syllabus 1 (Intro, ERM, Bias-Variance):** Covered in Topics 1, 2, and 6.
2. **Syllabus 2 (Bayes Theory, Gaussians, Parameter Estimation):** Covered in Topics 3 and 4.
3. **Syllabus 3 (PAC Theory, Finite/Infinite Spaces, VC Dim):** Covered in Topics 5 and 17.
4. **Syllabus 4 (Regression, Logistic, Perceptron, GD/SGD):** Covered in Topics 7 and 8.
5. **Syllabus 5 & 6 (SVMs, Max-Margin Optimization, Kernels, CV, ROC):** Covered in Topics 7, 11, and 18.
6. **Syllabus 7 (Trees, Boosting, Naïve Bayes, $k$-NN):** Covered in Topics 10, 11, and 20.
7. **Syllabus 8 & 9 (MLPs, Backprop, Softmax, CNNs, Dropout, Embeddings):** Covered in Topics 9, 14, 21, 22, 23, and 24.
8. **Syllabus 10 & 11 & 12 (Clustering, $K$-means, EM, PCA, Autoencoders, SSL):** Covered in Topics 12, 13, 15, 19, and 22.
9. **Syllabus 13 (Sequences, RNNs, LSTMs, Transformers, Attention):** Covered in Topics 16, 21, and 23.

---
## 🔗 Navigation
**Previous:** [[Topic 25 — Algorithmic Fairness & Bias in Machine Learning (Class 12)]] | **Next:** [[Machine learning MOC]]