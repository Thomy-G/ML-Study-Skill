# ✍️ 2023 Moed B — Answers Template | דף תשובות — מועד ב׳ 2023

**Student Name / שם הסטודנט/ית:**  
**Date / תאריך:**  

---

## 🔹 Part A: True/False Questions / חלק א': שאלות נכון/לא נכון

### Question 1: Overfitting Reduction / שאלה 1
* **a.** [Yes / No] — *Reasoning / נימוק:*
* **b.** [Yes / No] — *Reasoning / נימוק:*
* **c.** [Yes / No] — *Reasoning / נימוק:*
* **d.** [Yes / No] — *Reasoning / נימוק:*

### Question 2: Core Principles / שאלה 2
* **a.** [Correct / Wrong] — *Reasoning / נימוק:*
* **b.** [Correct / Wrong] — *Reasoning / נימוק:*
* **c.** [Correct / Wrong] — *Reasoning / נימוק:*
* **d.** [Correct / Wrong] — *Reasoning / נימוק:*

---

## 🔹 Part B: Open Questions / חלק ב': שאלות פתוחות

### Question 3: Linear Classifiers & MAP Logistic Regression / שאלה 3
* **a.** Binary logistic regression formulation, loss function, & rationale:
For logistic regression optimizing the likelihood is the same as minimizing the log likelihood, therefore we write it as such:
$$
\displaylines{
let: \\
y_{i} \in \Set{ -1,1 } \text{ be a label for the true data}\\
\hat{y} \in \Set{ -1,1 } \text{ predictions made by the model}\\
x_{i}\in \mathbb{R}^{d} \text{ the data that comes with the label as } (x_{i}, y_{i})\\
w \text{The weight vector}\\
\text{The negative log likelyhood is equivalent to the probability of getting a wrong answer}\\
\mathcal{L}_{NLL} = -\ln(L(x))
}
$$

* **b.** SGD derivation, gradient simplification, & learning rate selection:
 $$
 \displaylines{
 \nabla_{\mathbf{w}} \sum_{i=1}^n \ln \left( 1 + e^{-y_i \mathbf{w}^T \mathbf{x}_i} \right) = \sum_{i=1}^n \nabla_{\mathbf{w}} \ln \left( 1 + e^{-y_i \mathbf{w}^T \mathbf{x}_i} \right) = \sum_{i=1}^n \frac{1}{1 + e^{-y_i \mathbf{w}^T \mathbf{x}_i}} \cdot \nabla_{\mathbf{w}} \left( 1 + e^{-y_i \mathbf{w}^T \mathbf{x}_i} \right) =\\
 \sum_{i=1}^n \frac{1}{1 + e^{-y_i \mathbf{w}^T \mathbf{x}_i}} \cdot \left( -y_i \mathbf{x}_i e^{-y_i \mathbf{w}^T \mathbf{x}_i} \right) = \sum_{i=1}^n \frac{-y_i \mathbf{x}_i e^{-y_i \mathbf{w}^T \mathbf{x}_i}}{1 + e^{-y_i \mathbf{w}^T \mathbf{x}_i}} =\\
 \sum_{i=1}^n \frac{-y_i \mathbf{x}_i e^{-y_i \mathbf{w}^T \mathbf{x}_i} \cdot e^{y_i \mathbf{w}^T \mathbf{x}_i}}{\left( 1 + e^{-y_i \mathbf{w}^T \mathbf{x}_i} \right) \cdot e^{y_i \mathbf{w}^T \mathbf{x}_i}} = \sum_{i=1}^n \frac{-y_i \mathbf{x}_i}{e^{y_i \mathbf{w}^T \mathbf{x}_i} + 1} =\\
 -\sum_{i=1}^n y_i \mathbf{x}_i \cdot \frac{1}{1 + e^{y_i \mathbf{w}^T \mathbf{x}_i}} = -\sum_{i=1}^n y_i \mathbf{x}_i \left( 1 - \sigma(y_i \mathbf{w}^T \mathbf{x}_i) \right) =\\
 
 }
 $$
* **c.** MAP formulation with Gaussian prior $\mathcal{N}(0,1)$, new SGD update, & $L_2$ regularization effect:

#### 1. Bayesian MAP Log-Posterior Formulation (WHY & Derivation)
* **Why start with Bayes' Theorem?** Standard Maximum Likelihood (MLE) maximizes $P(\mathcal{D} \mid \mathbf{w})$, which can overfit noisy training data. Maximum A Posteriori (MAP) incorporates prior knowledge $P_0(\mathbf{w})$ to find the parameter vector $\mathbf{w}$ maximizing the posterior $P(\mathbf{w} \mid \mathcal{D})$.
* **Bayes' Rule**:
  $$P(\mathbf{w} \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \mathbf{w}) P_0(\mathbf{w})}{P(\mathcal{D})}$$
* **Why take the negative logarithm ($-\ln$)?**
  1. Taking the log turns products of probabilities ($\prod P_i$) into sums ($\sum \ln P_i$), avoiding floating-point underflow and simplifying differentiation.
  2. Taking the negative converts posterior maximization ($\max P$) into loss minimization ($\min [-\ln P]$) for SGD optimization.
  3. Constant term $\ln P(\mathcal{D})$ is independent of $\mathbf{w}$, so its gradient $\nabla_{\mathbf{w}} \ln P(\mathcal{D}) = \mathbf{0}$. We drop it as a constant.
$$\mathcal{L}_{\text{MAP}}(\mathbf{w}) = -\ln P(\mathbf{w} \mid \mathcal{D}) = -\ln P(\mathcal{D} \mid \mathbf{w}) - \ln P_0(\mathbf{w}) = \mathcal{L}_{\text{NLL}}(\mathbf{w}) - \ln P_0(\mathbf{w})$$

#### 2. Deriving the Gaussian Prior Term (WHY & Derivation)
* **Why product of Gaussians?** Each component $w_j \sim \mathcal{N}(0, 1)$ is assumed independent, so the joint prior is the product of individual 1D Gaussians:
$$P_0(w_j) = \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{w_j^2}{2} \right)$$
$$P_0(\mathbf{w}) = \prod_{j=1}^d P_0(w_j) = \left( \frac{1}{\sqrt{2\pi}} \right)^d \exp\left( -\frac{1}{2} \sum_{j=1}^d w_j^2 \right) = \left( \frac{1}{\sqrt{2\pi}} \right)^d \exp\left( -\frac{1}{2} \|\mathbf{w}\|^2 \right)$$
* **Taking the negative logarithm**:
$$-\ln P_0(\mathbf{w}) = -\ln \left[ \left( \frac{1}{\sqrt{2\pi}} \right)^d \exp\left( -\frac{1}{2} \|\mathbf{w}\|^2 \right) \right] = \frac{d}{2}\ln(2\pi) + \frac{1}{2} \|\mathbf{w}\|^2$$
* **Why drop constants?** $\frac{d}{2}\ln(2\pi)$ does not depend on $\mathbf{w}$, so its derivative is 0.
* **Core Insight**: The MAP objective simplifies to:
$$\mathcal{L}_{\text{MAP}}(\mathbf{w}) = \sum_{i=1}^n \ln \left( 1 + e^{-y_i \mathbf{w}^T \mathbf{x}_i} \right) + \frac{1}{2} \|\mathbf{w}\|^2$$
This proves that assigning a standard Gaussian prior $\mathcal{N}(0,1)$ over weights is **mathematically identical to $L_2$ weight decay regularization**!

#### 3. Gradient & New SGD Update Step (WHY & Derivation)
* **Why differentiate?** Gradient descent requires the directional gradient vector pointing toward steepest loss increase, so we step in the opposite direction ($-\nabla \mathcal{L}$).
* **Linearity of differentiation**:
$$\nabla_{\mathbf{w}} \mathcal{L}_{i, \text{MAP}}(\mathbf{w}) = \nabla_{\mathbf{w}} \mathcal{L}_{i, \text{NLL}}(\mathbf{w}) + \nabla_{\mathbf{w}} \left( \frac{1}{2} \|\mathbf{w}\|^2 \right)$$
Since $\nabla_{\mathbf{w}} \left( \frac{1}{2} \sum w_j^2 \right) = \mathbf{w}$:
$$\nabla_{\mathbf{w}} \mathcal{L}_{i, \text{MAP}}(\mathbf{w}) = -y_i \mathbf{x}_i \left( 1 - \sigma(y_i \mathbf{w}^T \mathbf{x}_i) \right) + \mathbf{w}$$
* **Deriving the SGD weight update step**:
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla_{\mathbf{w}} \mathcal{L}_{i, \text{MAP}}(\mathbf{w}_t)$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \left[ -y_i \mathbf{x}_i \left( 1 - \sigma(y_i \mathbf{w}_t^T \mathbf{x}_i) \right) + \mathbf{w}_t \right]$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{w}_t + \eta y_i \mathbf{x}_i \left( 1 - \sigma(y_i \mathbf{w}_t^T \mathbf{x}_i) \right)$$
$$\mathbf{w}_{t+1} = (1 - \eta) \mathbf{w}_t + \eta y_i \mathbf{x}_i \left( 1 - \sigma(y_i \mathbf{w}_t^T \mathbf{x}_i) \right)$$

#### 4. Effect of the Prior on Solution $\mathbf{w}$ (3 Core Practical Effects)
1. **Weight Shrinkage ($L_2$ Regularization)**: At every step, multiplying $\mathbf{w}_t$ by factor $(1 - \eta) < 1$ shrinks weight magnitudes toward zero, penalizing unnecessarily large parameters.
2. **Prevents Overconfidence (Smoother Probabilities)**: Unregularized large weights cause logits $\mathbf{w}^T \mathbf{x}$ to saturate sigmoid outputs to extreme values ($1.0$ or $0.0$) near the decision boundary. Keeping $\|\mathbf{w}\|$ small produces smooth, well-calibrated probabilities and prevents overfitting.
3. **Ensures Unique Finite Minimum**: On linearly separable data, standard NLL loss drives $\|\mathbf{w}\| \to \infty$ to make loss approach zero. Adding $\frac{1}{2}\|\mathbf{w}\|^2$ introduces strict convexity ($H \succ 0$), guaranteeing a **unique finite global minimum** $\mathbf{w}^*$.

---

### Question 4: Large Margin & 1-NN Network Equivalence / שאלה 4
* **a.** Kernel SVM prediction formula via $K$ vs $\Phi$ & complexity analysis:
  ```text
  
  ```
* **b.** 2-layer network structure & Kernel SVM equivalence proof:

$$
\displaylines{
\text{The first layer has } h_{i}(x) = y_{i}K(x,x_{i})\\
\text{the weights are } a_{1},\dots,a_{n}\\
\text{for all neuron in the second layer we got that we have a sign activation so that }\\
\text{the output equals}\\
sign\left( \sum_{i=1}^{n} a_{i}\cdot y_{i}K(x,x_{i}) \right)
}
$$
  
* **c.** RBF kernel parameter selection for 1-NN classification equivalence:
$$
\displaylines{
\text{For this to be equivalent to 1-NN we need to make each neuron must cast a vote and we will get the }\\
\text{nearest one to be the classifier}\\
\forall i \in [1,n]: a_{i} = 1\\

}
$$

---

### Question 5: Deep Networks & Multi-Output Architecture / שאלה 5
* **a.** 1-hidden-layer MLP forward pass mathematical formula:
$$
\displaylines{
\text{Suppose bias vector b}_{1} \in \mathbb{R}^{k}\\
y_{1}= W_{1}x + b_{1}
}
$$
* **b.** Chain rule derivatives $\frac{\partial l}{\partial \mathbf{w}_2}, \frac{\partial l}{\partial W_1}$ & SGD updates:

#### 1. Forward Pass Summary
$$\mathbf{z}^{(1)} = W_1^T \mathbf{x} \in \mathbb{R}^k \quad \implies z_j^{(1)} = \sum_{m=1}^d W_{1, m, j} x_m$$
$$\mathbf{h} = g_1(\mathbf{z}^{(1)}) \in \mathbb{R}^k \quad \implies h_j = g_1(z_j^{(1)})$$
$$\hat{y} = \mathbf{w}_2^T \mathbf{h} = \sum_{j=1}^k w_{2, j} h_j \in \mathbb{R}$$
$$l = \frac{1}{2} (\hat{y} - y)^2$$

---

#### 2. Derivation of $\frac{\partial l}{\partial \mathbf{w}_2}$ (Output Layer Weight Gradient)
Using the chain rule for each weight component $w_{2, j}$ ($j = 1, \dots, k$):
$$\frac{\partial l}{\partial w_{2, j}} = \frac{\partial l}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_{2, j}}$$

* **Step A: Output Error Signal ($\delta_{\text{out}}$)**:
  $$\frac{\partial l}{\partial \hat{y}} = \frac{\partial}{\partial \hat{y}} \left[ \frac{1}{2} (\hat{y} - y)^2 \right] = (\hat{y} - y) \equiv \delta_{\text{out}} \in \mathbb{R}$$

* **Step B: Partial Derivative of $\hat{y}$ w.r.t. $w_{2, j}$**:
  $$\frac{\partial \hat{y}}{\partial w_{2, j}} = \frac{\partial}{\partial w_{2, j}} \left( \sum_{i=1}^k w_{2, i} h_i \right) = h_j$$

* **Step C: Combine**:
  $$\frac{\partial l}{\partial w_{2, j}} = (\hat{y} - y) h_j$$
  In vector form:
  $$\frac{\partial l}{\partial \mathbf{w}_2} = (\hat{y} - y) \mathbf{h} = \delta_{\text{out}} \mathbf{h} \in \mathbb{R}^k$$

---

#### 3. Derivation of $\frac{\partial l}{\partial W_1}$ (Hidden Layer Weight Matrix Gradient)
We derive $\frac{\partial l}{\partial W_{1, m, j}}$, where $W_{1, m, j}$ connects input feature $x_m$ ($m=1,\dots,d$) to hidden neuron $j$ ($j=1,\dots,k$).

By multivariate chain rule:
$$\frac{\partial l}{\partial W_{1, m, j}} = \frac{\partial l}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h_j} \cdot \frac{\partial h_j}{\partial z_j^{(1)}} \cdot \frac{\partial z_j^{(1)}}{\partial W_{1, m, j}}$$

Let's evaluate each factor step-by-step:

* **Factor 1 ($\frac{\partial l}{\partial \hat{y}}$)**: Output error = $(\hat{y} - y)$
* **Factor 2 ($\frac{\partial \hat{y}}{\partial h_j}$)**: Effect of hidden neuron $j$ on output = $w_{2, j}$
* **Factor 3 ($\frac{\partial h_j}{\partial z_j^{(1)}}$)**: Derivative of activation function = $g_1'(z_j^{(1)})$
* **Factor 4 ($\frac{\partial z_j^{(1)}}{\partial W_{1, m, j}}$)**: Pre-activation derivative w.r.t weight = $x_m$

Combining Factors 1, 2, and 3 gives the backpropagated error signal for hidden neuron $j$:
$$\delta_j^{(1)} \equiv \frac{\partial l}{\partial z_j^{(1)}} = (\hat{y} - y) w_{2, j} g_1'(z_j^{(1)})$$
In vector notation:
$$\boldsymbol{\delta}^{(1)} = (\hat{y} - y) \mathbf{w}_2 \odot g_1'(\mathbf{z}^{(1)}) = \delta_{\text{out}} \mathbf{w}_2 \odot g_1'(\mathbf{z}^{(1)}) \in \mathbb{R}^k$$

Multiplying by Factor 4 ($x_m$):
$$\frac{\partial l}{\partial W_{1, m, j}} = x_m \delta_j^{(1)} = x_m (\hat{y} - y) w_{2, j} g_1'(z_j^{(1)})$$

Taking the outer product of input vector $\mathbf{x} \in \mathbb{R}^d$ and hidden error vector $\boldsymbol{\delta}^{(1)} \in \mathbb{R}^k$:
$$\frac{\partial l}{\partial W_1} = \mathbf{x} (\boldsymbol{\delta}^{(1)})^T = \mathbf{x} \left[ (\hat{y} - y) \mathbf{w}_2 \odot g_1'(\mathbf{z}^{(1)}) \right]^T \in \mathbb{R}^{d \times k}$$

---

#### 4. SGD Weight Update Steps
Using learning rate $\eta > 0$:

1. **Second Layer Weights ($\mathbf{w}_2$) Update**:
   $$\mathbf{w}_2^{(t+1)} = \mathbf{w}_2^{(t)} - \eta \frac{\partial l}{\partial \mathbf{w}_2} = \mathbf{w}_2^{(t)} - \eta (\hat{y} - y) \mathbf{h}$$

2. **First Layer Weight Matrix ($W_1$) Update**:
   $$W_1^{(t+1)} = W_1^{(t)} - \eta \frac{\partial l}{\partial W_1} = W_1^{(t)} - \eta \mathbf{x} \left[ (\hat{y} - y) \mathbf{w}_2^{(t)} \odot g_1'(W_1^{(t)T} \mathbf{x}) \right]^T$$
* **c.** Multi-task architecture (regression $y_1$ + classification $y_2$), joint loss, & backprop update:

The MLP architecture will have 2 different neurons in the hidden layer per input that will have their own loss function, for the continuous output it will have regression for loss using MSE and for the binary calification it will have hinge loss, finally each neuron in the hidden layer will connect to the binary and linear output respectively and have loss based on that output

---

### Question 6: Unsupervised Learning & PCA / שאלה 6
* **a.** PCA dimensionality reduction pseudocode ($d \to r$):
```python
  
```
* **b.** Reconstruction error minimization proof $\mathbf{z} = V^T \mathbf{x}$:

* **c.** PCA pre-processing trade-offs for classification & $d=2, r=1$ geometric counter-example:
  ```text
  
  ```
