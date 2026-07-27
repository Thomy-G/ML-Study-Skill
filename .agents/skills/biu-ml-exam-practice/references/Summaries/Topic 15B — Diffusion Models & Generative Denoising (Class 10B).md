---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Summary
date_added: 2026-07-21
---
# Topic 15B — Diffusion Models & Generative Denoising (Class 10B)

**MOC:** [[Machine learning MOC]]
**Course:** [[English]]

# Study Guide: Topic 15B — Diffusion Models & Generative Denoising (Class 10B)

## 1. Context: From Denoising Autoencoders to Diffusion Models

Standard Autoencoders compress data through a low-dimensional bottleneck to avoid learning a trivial identity mapping. However, another powerful way to force a model to learn true structure without bottleneck compression is through a **Denoising Autoencoder (DAE)**.

### Denoising Autoencoder Paradigm
1.  **Corruption Function:** We corrupt a clean data sample $\mathbf{x}$ from the data distribution $\mathcal{D}$ using a corruption function $c(\mathbf{x}) \rightarrow \tilde{\mathbf{x}}$ (e.g., adding Gaussian noise, applying dropout/masking, or adding blur).
2.  **Encoder-Decoder mapping:** The corrupted input $\tilde{\mathbf{x}}$ is mapped to a latent code $\mathbf{a} = f_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})$ and then reconstructed as $\hat{\mathbf{x}} = g_{\boldsymbol{\phi}}(\mathbf{a})$.
3.  **Objective Function (MSE):** The network parameters are optimized using a squared error objective to reconstruct the original *clean* image $\mathbf{x}$, not the corrupted one:
    
    $$\min_{\boldsymbol{\theta}, \boldsymbol{\phi}} \mathbb{E}_{\mathbf{x} \sim \mathcal{D}} \left[ \|\mathbf{x} - g_{\boldsymbol{\phi}}(f_{\boldsymbol{\theta}}(c(\mathbf{x})))\|^2 \right]$$

To minimize this loss, the model cannot simply memorize or copy. It must learn the underlying manifold structure of the data to separate the true signal from the injected noise.

### The Diffusion Evolution
**Denoising Diffusion Probabilistic Models (DDPM)** build on this exact denoising concept but apply it **iteratively**. Instead of adding noise in a single step and reconstructing it, diffusion models decompose the task:
*   **Forward Direction (Fixed):** Slowly and iteratively add small amounts of Gaussian noise over $T$ steps until the data collapses into pure white noise.
*   **Backward Direction (Learned):** Train a deep neural network to iteratively predict and remove the noise at each step, generating clean data from pure noise.

---

## 2. Mathematical Formulation of Diffusion

### The Forward Diffusion Process (Noising)
Given a clean image $\mathbf{x}_0$, we add Gaussian noise iteratively over $T$ steps. At each step $t$, the distribution of $\mathbf{x}_t$ given $\mathbf{x}_{t-1}$ is defined by:

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\left(\sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t I\right)$$

where $\beta_1, \dots, \beta_T$ are pre-defined variance schedules (hyperparameters).

#### The Closed-Form Single-Step Sampling Trick
Using the properties of Gaussian distributions, we can express the distribution of $\mathbf{x}_t$ directly at any arbitrary step $t$ given the initial clean image $\mathbf{x}_0$ without computing all intermediate steps. 

Let $\alpha_t := 1 - \beta_t$, and define the cumulative product $\bar{\alpha}_t := \prod_{i=1}^t \alpha_i$:

$$\mathbf{x}_t = \sqrt{\alpha_t} \mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1}$$

Recursively expanding this yields the **Direct Sampling Equation**:

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}, \quad \text{where } \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$$

Therefore:

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)I\right)$$

---

## 3. Training and Sampling Algorithms

### The Backward Process (Denoising)
Since the true reverse transition $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ is mathematically intractable, we learn a parameterized Gaussian transition $p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$. Instead of predicting the clean image directly, the deep network $\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)$ (typically a U-Net architecture with self/cross-attention) is trained to **predict the added noise vector $\boldsymbol{\epsilon}$** at timestep $t$.

### Algorithm 1: Training Loop
```
Repeat until convergence:
  1. Sample clean data: x_0 ~ q(x_0)
  2. Sample random timestep: t ~ Uniform({1, ..., T})
  3. Sample random Gaussian noise: ε ~ N(0, I)
  4. Perform a gradient descent step on the objective:
     Minimize w.r.t θ:  || ε - ε_θ( \sqrt{α_bar_t} x_0 + \sqrt{1 - α_bar_t} ε, t ) ||^2
```

### Algorithm 2: Sampling / Inference Loop (Generation)
To generate a new image, we start with pure random noise $\mathbf{x}_T$ and iteratively denoise step-by-step back to $\mathbf{x}_0$:

```
1. Sample initial pure noise: x_T ~ N(0, I)
2. For t = T, T-1, ..., 1:
     Sample z ~ N(0, I) if t > 1, else z = 0
     Compute denoised state:
       x_(t-1) = [ 1 / \sqrt{α_t} ] * ( x_t - [ (1 - α_t) / \sqrt{1 - α_bar_t} ] * ε_θ(x_t, t) ) + σ_t * z
3. Return x_0
```

---

## 4. Advanced Architectures & Guidance ("System 2" Generation)

### Stable Diffusion Architecture
Stable Diffusion performs the noising/denoising processes inside a compressed **Latent Space** rather than high-dimensional pixel space to reduce computational complexity. It uses three main components:
1.  **Variational Autoencoder (VAE):** Encodes pixels to latents and decodes latents back to pixels.
2.  **U-Net Denoising Network:** Predicts latent noise, utilizing cross-attention modules.
3.  **Conditioning Encoder (e.g., CLIP):** Encodes prompt text, feeding embeddings into the U-Net via cross-attention.

### "System 2" Inference-Time Guidance
Rather than generating freely, we can guide the generation process at inference time to enforce specific spatial or semantic constraints (e.g., "a frog *below* a bowl").

*   **Attention Map Manipulation:** We can extract the U-Net's internal cross-attention maps (which link text words to spatial image regions).
*   **Optimization-Based Guidance:** We construct a relation loss function $\mathcal{L}_{\text{guide}}$ representing the desired constraint. During each denoising step $t$, we compute the gradient of the guidance loss with respect to the latent state $\mathbf{z}_t$ and update it:
    
    $$\mathbf{z}_t \leftarrow \mathbf{z}_t - \eta \nabla_{\mathbf{z}_t}\mathcal{L}_{\text{guide}}$$
    
    This guides the diffusion trajectory to satisfy the constraints without retraining the model.

### Video Semantic Layer Decomposition (OmniMatteZero)
Diffusion models can also be adapted to isolate objects in video sequences. By combining subject tracking with guided latent blending, models can segment a foreground subject and automatically isolate its corresponding environmental interactions (such as casting shadows or creating water ripples).

---
## 🔗 Navigation
**Previous:** [[Topic 15 — Self-Supervised Learning & Autoencoders (Class 10, Recitation 10)]] | **Next:** [[Topic 16 — Advanced Deep Learning - Sequenced Learning & Recurrent Neural Networks (Class 11)]]
