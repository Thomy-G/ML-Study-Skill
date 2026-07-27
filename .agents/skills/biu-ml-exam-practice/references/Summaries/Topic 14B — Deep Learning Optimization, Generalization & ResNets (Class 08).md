---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Summary
date_added: 2026-07-21
---
# Topic 14B — Deep Learning Optimization, Generalization & ResNets (Class 08)

**MOC:** [[Machine learning MOC]]
**Course:** [[English]]

# Study Guide: Topic 14B — Deep Learning Optimization, Generalization & ResNets (Class 08)

## 1. Context: The Depth Bottleneck in Deep Neural Networks

As we stack more layers in feedforward architectures (like MLPs or ConvNets) to increase expressivity, we expect the representation capacity to grow, lowering the approximation error. However, in practice, training extremely deep networks (e.g., standard feedforward networks with 50+ layers) yields a paradoxical result: **both the training error and test error worsen** compared to shallower networks. 

This is not an overfitting issue (as the training error itself is high); it is a fundamental **optimization bottleneck**.

### Why Deeper Networks Fail to Optimize at Initialization
1. **Vanishing and Exploding Gradients:** Deep networks compose many functions. By the calculus Chain Rule, calculating the gradient at early layers requires multiplying many weight matrices together. If the eigenvalues of the weights are slightly smaller than 1, the gradient decays exponentially to zero (vanishing). If they are larger than 1, it grows exponentially (exploding).
2. **Noisy and Meaningless Initial Activations:** If weights are initialized randomly, the activations of early layers are mostly random noise. Over many layers, this noise propagates, and when passed through ReLU activations, a large fraction of the activations collapse to exactly zero. 
3. **Noisy Early Gradients:** When the noisy error signal backpropagates back to the early layers, it becomes highly corrupted. The gradients at the beginning of the network take an exceptionally long time to convey any meaningful direction for optimization.

---

## 2. Residual Networks (ResNets) & Skip Connections

To solve the optimization bottleneck in very deep networks, He et al. (2015) introduced **Residual Learning** and **ResNets**.

### Mathematical Formulation of Residual Blocks
Instead of expecting a stack of layers to directly fit an underlying mapping $H(\mathbf{x})$, we configure the layers to fit a **residual mapping** $F(\mathbf{x}) := H(\mathbf{x}) - \mathbf{x}$. 

The original mapping is reconstructed by adding the input $\mathbf{x}$ back to the output of the layers:

$$H(\mathbf{x}) = F(\mathbf{x}) + \mathbf{x}$$

```
                Input (x)
             ┌──────┴──────┐
             │             │ (Identity Shortcut)
             ▼             │
        ┌─────────┐        │
        │ Weight  │        │
        └────┬────┘        │
             ▼             │
          [ReLU]           │
             ▼             │
        ┌─────────┐        │
        │ Weight  │        │
        └────┬────┘        │
             ▼             │
             └────►(+)◄────┘ (Element-wise Addition)
                   │
                   ▼
                [ReLU]
                   │
                   ▼
               Output H(x)
```

### Why Skip Connections Solve the Gradient Problem
The shortcut connection $\mathbf{x}$ performs a parameter-free identity mapping. During backpropagation, the gradient of the loss $\mathcal{L}$ with respect to the input $\mathbf{x}$ of the residual block is:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial H(\mathbf{x})} \cdot \frac{\partial H(\mathbf{x})}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial H(\mathbf{x})} \left( \frac{\partial F(\mathbf{x})}{\partial \mathbf{x}} + I \right)$$

The addition of the identity matrix $I$ acts as a **gradient highway**. Even if the weights in the residual layers $F(\mathbf{x})$ are small or poorly initialized (causing $\frac{\partial F(\mathbf{x})}{\partial \mathbf{x}} \approx 0$), the gradient flows back to previous layers completely unimpeded through the $+ I$ term. This enables training networks with hundreds or thousands of layers.

---

## 3. Improving Optimization: Initialization & Normalization

### Weight Initialization Strategies
Properly scaling weights at initialization is crucial to preserve the variance of activations in the forward pass and the variance of gradients in the backward pass.

*   **Xavier (Glorot) Initialization:** Designed for linear or symmetric activations (like Sigmoid/Tanh). Weights are sampled from:
    
    $$W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right) \quad \text{or} \quad W \sim \text{Uniform}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$$
    
*   **Kaiming (He) Initialization:** Designed specifically for ReLU activations (which zero out negative values, halving the variance). The weights are scaled by an extra factor of $2$:
    
    $$W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$$

---

### Modern Optimization Algorithms: Momentum, RMSProp & Adam

While standard Stochastic Gradient Descent (SGD) updates parameters via $\boldsymbol{\theta}_{t} = \boldsymbol{\theta}_{t-1} - \eta \mathbf{g}_t$ (where $\mathbf{g}_t = \nabla_{\boldsymbol{\theta}} \mathcal{L}$), standard SGD struggles on ill-conditioned loss surfaces, ravines, and noisy mini-batch gradients.

#### 1. SGD with Momentum (Polyak, 1964)
Momentum accumulates a velocity vector $\mathbf{v}_t$ in the direction of persistent loss reduction, accelerating descent and dampening oscillations:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \eta \mathbf{g}_t, \qquad \boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \mathbf{v}_t$$

*(where $\beta \in [0, 1)$ is the momentum decay hyperparameter, typically $\beta = 0.9$).*

#### 2. RMSProp (Root Mean Squared Propagation - Tieleman & Hinton, 2012)
RMSProp adapts the learning rate per parameter by dividing the gradient by an Exponential Moving Average (EMA) of squared gradients:

$$\mathbf{s}_t = \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2, \qquad \boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t$$

*(where $\mathbf{g}_t^2$ is element-wise square, $\beta_2 \approx 0.99$, and $\epsilon \approx 10^{-8}$ prevents division by zero).*

#### 3. Adam Optimizer (Adaptive Moment Estimation - Kingma & Ba, 2014)
**Adam** combines the principles of **Momentum** (1st moment estimate) and **RMSProp** (2nd moment estimate).

##### The 5 Step-by-Step Equations:

1. **First Moment Vector (Exponential Moving Average of Gradients):**
   $$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t$$

2. **Second Moment Vector (Exponential Moving Average of Squared Gradients):**
   $$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2$$

3. **First-Moment Bias Correction:**
   $$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}$$

4. **Second-Moment Bias Correction:**
   $$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$

5. **Parameter Update Rule:**
   $$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \odot \hat{\mathbf{m}}_t$$

##### Why Bias Correction is Required:
At initial timesteps ($t=1, 2$), $\mathbf{m}_0 = \mathbf{0}$ and $\mathbf{v}_0 = \mathbf{0}$, causing uncorrected estimates $\mathbf{m}_t$ and $\mathbf{v}_t$ to be severely biased toward zero. Dividing by $(1 - \beta^t)$ corrects this initialization bias early in training. Standard defaults: $\eta = 0.001, \beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$.

---

#### 4. AdaGrad (Adaptive Gradient Algorithm - Duchi et al., 2011)
AdaGrad adapts the learning rate to feature frequency by accumulating the sum of squared gradients from the start of training:

$$\mathbf{G}_t = \mathbf{G}_{t-1} + \mathbf{g}_t^2 = \sum_{\tau=1}^t \mathbf{g}_\tau^2, \qquad \boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \frac{\eta}{\sqrt{\mathbf{G}_t} + \epsilon} \odot \mathbf{g}_t$$

*   **Key Advantage:** Performs larger updates for rare/infrequent features and smaller updates for frequent features (ideal for sparse data like text/NLP).
*   **Core Limitation:** Because $\mathbf{G}_t$ accumulates positive terms monotonically, the denominator grows continuously, causing the effective learning rate to decay to zero ($\frac{\eta}{\sqrt{\mathbf{G}_t}} \rightarrow 0$) and prematurely stopping learning in deep networks. *(RMSProp resolved this by replacing the sum with an exponential moving average).*

---

#### 5. AdamW (Decoupled Weight Decay - Loshchilov & Hutter, 2017)
In standard SGD, adding an $L_2$ regularization penalty $\frac{\lambda}{2}\|\boldsymbol{\theta}\|^2$ to the loss function is mathematically identical to applying weight decay. 

However, in Adam, adding $L_2$ regularization to the gradient ($\mathbf{g}_t \leftarrow \mathbf{g}_t + \lambda \boldsymbol{\theta}_{t-1}$) forces the regularization term to pass through the gradient moments $\mathbf{m}_t$ and $\mathbf{v}_t$. As a result, the decay term is divided by $\sqrt{\hat{\mathbf{v}}_t}$, meaning **weights with larger gradients receive less weight decay than weights with small gradients**.

**AdamW** fixes this by **decoupling weight decay** from the adaptive gradient moments, applying weight decay directly to the parameter update step:

$$\boldsymbol{\theta}_t = (1 - \eta \lambda) \boldsymbol{\theta}_{t-1} - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \odot \hat{\mathbf{m}}_t$$

*(where $\lambda$ is the explicit weight decay rate). This simple decoupling significantly improves generalization performance and is the standard default optimizer used to train modern **Large Language Models (LLMs) and Vision Transformers**.*

---

### Batch Normalization (BatchNorm)
Batch Normalization (Ioffe & Szegedy, 2015) stabilizes the distribution of activations across layers throughout training, mitigating **Internal Covariate Shift** and smoothing the loss landscape.

#### The 4 Step-by-Step Equations
For a mini-batch $\mathcal{B} = \{\mathbf{x}_1, \dots, \mathbf{x}_m\}$ of $m$ feature vectors:

1. **Mini-Batch Mean:**
   $$\boldsymbol{\mu}_{\mathcal{B}} = \frac{1}{m}\sum_{i=1}^m \mathbf{x}_i$$

2. **Mini-Batch Variance:**
   $$\boldsymbol{\sigma}^2_{\mathcal{B}} = \frac{1}{m}\sum_{i=1}^m (\mathbf{x}_i - \boldsymbol{\mu}_{\mathcal{B}})^2$$

3. **Standardized Zero-Mean Normalization:**
   $$\hat{\mathbf{x}}_i = \frac{\mathbf{x}_i - \boldsymbol{\mu}_{\mathcal{B}}}{\sqrt{\boldsymbol{\sigma}^2_{\mathcal{B}} + \epsilon}}$$
   *(where $\epsilon > 0$ is a small constant for numerical stability).*

4. **Learnable Scale & Shift Transformation:**
   $$\mathbf{y}_i = \boldsymbol{\gamma} \odot \hat{\mathbf{x}}_i + \boldsymbol{\beta}$$
   *(where $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$ are trainable parameters learned via backpropagation, allowing the network to recover the identity transformation if optimal).*

#### Training Phase vs. Inference Phase Behavior
* **Training Phase:** Mini-batch mean $\boldsymbol{\mu}_{\mathcal{B}}$ and variance $\boldsymbol{\sigma}^2_{\mathcal{B}}$ are calculated dynamically for every batch. Simultaneously, running estimates are tracked using Exponential Moving Averages (EMA):
  $$\boldsymbol{\mu}_{\text{run}} \leftarrow (1 - \alpha)\boldsymbol{\mu}_{\text{run}} + \alpha \boldsymbol{\mu}_{\mathcal{B}}, \qquad \boldsymbol{\sigma}^2_{\text{run}} \leftarrow (1 - \alpha)\boldsymbol{\sigma}^2_{\text{run}} + \alpha \boldsymbol{\sigma}^2_{\mathcal{B}}$$
* **Inference (Test) Phase:** Mini-batches may be of size 1. BatchNorm operates deterministically using the fixed running statistics accumulated during training:
  $$\hat{\mathbf{x}}_{\text{test}} = \frac{\mathbf{x} - \boldsymbol{\mu}_{\text{run}}}{\sqrt{\boldsymbol{\sigma}^2_{\text{run}} + \epsilon}}, \qquad \mathbf{y}_{\text{test}} = \boldsymbol{\gamma} \odot \hat{\mathbf{x}}_{\text{test}} + \boldsymbol{\beta}$$

#### Core Advantages & Properties
1. **Improves Gradient Flow:** Prevents activations from saturating flat regions of non-linear functions (like Sigmoid/Tanh).
2. **Allows Higher Learning Rates:** Stabilizes parameter updates, allowing training with significantly larger learning rates $\eta$.
3. **Implicit Regularization:** Batch-dependent mean and variance introduce stochastic noise during training, reducing overfitting (similar to Dropout).

---

## 4. Generalization & The Double Descent Phenomenon

### Explicit vs. Implicit Regularization
*   **Explicit (Weight Decay):** Adding an $L_2$ penalty to the loss function $\mathcal{L}_{\text{total}} = \mathcal{L} + \frac{\lambda}{2}\|\mathbf{w}\|^2$. In gradient descent, this corresponds to:
    
    $$\mathbf{w}_{t} = (1 - \eta \lambda)\mathbf{w}_{t-1} - \eta \nabla \mathcal{L}(\mathbf{w}_{t-1})$$
    
*   **Implicit (Data Augmentation):** Modifying training instances (e.g., image flips, rotations) to enforce invariance. 
    *   **MixUp:** A technique that blends random pairs of images and their labels:
        
        $$\tilde{\mathbf{x}} = \lambda \mathbf{x}_i + (1-\lambda)\mathbf{x}_j, \quad \tilde{\mathbf{y}} = \lambda \mathbf{y}_i + (1-\lambda)\mathbf{y}_j \quad \text{for } \lambda \sim \text{Beta}(\alpha, \alpha)$$

### Rethinking Generalization: Memorization capacity
Deep neural networks have so much capacity that they can easily fit entirely random labels or random noise pixels (achieving 100% training accuracy). Yet, when trained on real labels, they still generalize well. This challenges classical learning theory.

### The Double Descent Curve
Classical statistical learning theory states that increasing model complexity beyond the "interpolation threshold" (where parameters = samples, allowing 0 training error) leads to overfitting and high test error.

Modern deep learning observes a **Double Descent** curve:

```
Test Error
  ▲
  │     / \           Double Descent Curve
  │    /   \
  │   /     \ ◄─── Interpolation Threshold
  │  /       \
  │ /         \───────────► Gen. improves again in over-parameterized region
  └──────────────────────────────► Model Complexity (# Parameters)
   ◄─ Under-param. ─►◄─ Over-param. ─►
```

Once the model enters the **over-parameterized region** (parameters $\gg$ samples), the test error decreases again.

### Why Over-Parameterized Networks Generalize Well
1.  **Implicit Regularization of SGD:** Stochastic Gradient Descent naturally favors minimum-norm solutions, biasing optimization toward smooth, low-complexity functions.
2.  **Benign Overfitting:** In over-parameterized regions, the network can learn a function that interpolates noisy training points by adding localized spikes of noise that do not affect the global smooth classification signal.
3.  **Spectral Bias:** Neural networks learn simple, low-frequency (smooth) components of target functions first before learning high-frequency details.
4.  **Lottery Ticket Hypothesis:** A randomly-initialized dense network contains sparse subnetworks ("winning tickets") that, when trained in isolation, can match the accuracy of the original network in the same number of training steps.

---
## 🔗 Navigation
**Previous:** [[Topic 14 — Neural Networks & Deep Learning (MLPs, ConvNets & Backpropagation) (Classes 07A & 07B, Recitation 07)]] | **Next:** [[Topic 15 — Self-Supervised Learning & Autoencoders (Class 10, Recitation 10)]]
