---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Summary
date_added: 2026-06-21
---
# Topic 15 — Self-Supervised Learning & Autoencoders (Class 10, Recitation 10)

**MOC:** [[Machine learning MOC]]
**Course:** [[English]]

# Study Guide: Topic 15 — Self-Supervised Learning & Autoencoders (Class 10, Recitation 10)

## 1. Context: Learning Representations Without Human Labels

Collecting huge amounts of high-quality human-annotated training labels is exceptionally expensive, time-consuming, and often logistically impractical. However, massive streams of raw, _unlabeled_ data (images, text corpora) are abundant and easy to collect.

**Self-Supervised Learning (SSL)** bridges this gap: it takes an unlabeled dataset and artificially designs a synthetic pretext prediction task (e.g., predicting if an image has been rotated, predicting a missing word in a sentence, or reconstructing the input itself). By training a model to solve this artificial task, the network is forced to learn robust, low-dimensional **feature representations** that capture the core underlying structure of the data. These learned features can then be reused for downstream supervised tasks with very few human labels.

## 2. Autoencoder Architecture

An **Autoencoder** is a neural network trained to solve a core self-supervised pretext task: reconstruct its own input data. To prevent the model from simply learning a trivial identity mapping (copying inputs directly to outputs), the network architecture passes the data through a low-dimensional bottleneck.

```
Input (x)              Bottleneck (z)            Reconstructed (x_hat)
 ┌───────┐                 ┌───┐                 ┌───────┐
 │       ├───┐             │   │             ┌───┤       │
 │       │   │   ┌───┐     │   │     ┌───┐   │   │       │
 │       │   └──►│ENC├────►│   ├────►│DEC├───┘   │       │
 │       │   ┌───┘   │     │   │     └───┘   └──►│       │
 │       │   │             │   │                 │       │
 └───────┘   ┘             └───┘                 └───────┘
  High Dim                  Latent                High Dim
 (e.g., d)                Space (r)               Space (d)
```

### The Architectural Components

1. **The Encoder ($f_{\boldsymbol{\theta}}$):** Maps the high-dimensional input vector $\mathbf{x} \in \mathbb{R}^d$ down to a compact, low-dimensional hidden latent space code $\mathbf{z} \in \mathbb{R}^r$ (where $r \ll d$):
    
    $$\mathbf{z} = f_{\boldsymbol{\theta}}(\mathbf{x}) = g\left(W_e \mathbf{x} + \mathbf{b}_e\right)$$
    
2. **The Decoder ($g_{\boldsymbol{\phi}}$):** Takes the compressed latent code $\mathbf{z}$ and attempts to reconstruct the original high-dimensional input vector:
    
    $$\hat{\mathbf{x}} = g_{\boldsymbol{\phi}}(\mathbf{z}) = g\left(W_d \mathbf{z} + \mathbf{b}_d\right)$$
    

### The Reconstruction Loss Objective & The $L_2$ "Blurriness Problem"

We train autoencoders by minimizing reconstruction error (typically Mean Squared Error / $L_2$ loss):

$$\min_{\boldsymbol{\theta}, \boldsymbol{\phi}} \frac{1}{n}\sum_{i=1}^n \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2 = \min_{\boldsymbol{\theta}, \boldsymbol{\phi}} \frac{1}{n}\sum_{i=1}^n \|\mathbf{x}_i - g_{\boldsymbol{\phi}}\left(f_{\boldsymbol{\theta}}(\mathbf{x}_i)\right)\|^2$$

#### ⚠️ The $L_2$ Averaging Problem (Why Pixel Reconstruction Produces Blurry Images)
In complex generative or prediction tasks (e.g. predicting the next frame in a video or predicting missing image regions), the future or missing context is inherently uncertain with multiple plausible outcomes.

Because $L_2$ loss penalizes squared distance to *all* possible true outcomes, minimizing $L_2$ loss under uncertainty forces the network to learn the **mathematical average of all plausible futures**. This results in **blurry, indistinct images** rather than sharp, semantic understandings of objects.

---

### 2.1 Modern SSL Paradigms: Contrastive Learning & Masked Autoencoders

To avoid the $L_2$ blurriness problem, modern Self-Supervised Learning shifts away from raw pixel reconstruction:

1. **Contrastive SSL (SimCLR - Chen et al., 2020):**
   * Instead of predicting pixels, SimCLR creates two augmented views $(\tilde{\mathbf{x}}_i, \tilde{\mathbf{x}}_j)$ of the same image (positive pair) and passes them through a rich encoder $f(\cdot)$ (ResNet) and a non-linear projection head $g(\cdot)$ to obtain representations $\mathbf{z}_i, \mathbf{z}_j$.
   * It optimizes the **NT-Xent (Normalized Temperature-scaled Cross Entropy)** loss to pull positive pairs together while pushing negative samples (other images in the batch) apart:
     $$\mathcal{L}_{i,j} = -\log \frac{\exp\left(\text{sim}(\mathbf{z}_i, \mathbf{z}_j)/\tau\right)}{\sum_{k=1}^{2N} \mathbb{I}_{[k \neq i]} \exp\left(\text{sim}(\mathbf{z}_i, \mathbf{z}_k)/\tau\right)}$$

2. **Masked Autoencoders (MAE - He et al., 2021):**
   * MAE masks out a huge fraction (**75% to 90%**) of random image patches.
   * An encoder processes *only* the remaining visible patches, forcing it to learn strong visual representations. A lightweight decoder then attempts to reconstruct the missing masked patches. After pre-training, the decoder is discarded and the encoder is fine-tuned for downstream tasks.

## 3. Mathematical Equivalence: Linear Autoencoders & PCA

A classic theoretical exam question requires analyzing a **Linear Autoencoder** (an autoencoder with no non-linear activation functions) trained using the Mean Squared Error reconstruction loss.

### The Theorem

If the hidden bottleneck layer is strictly linear, the $r$-dimensional subspace learned by a linear autoencoder is mathematically equivalent to the subspace spanned by the top $r$ principal components discovered by **Principal Component Analysis (PCA)**.

### Crucial Architectural Differences

- **PCA** enforces a strict mathematical structure: its projection matrix must consist of mutually orthogonal columns sorted by variance magnitude.
    
- **A Linear Autoencoder** will learn the exact same global geometric subspace, but its weight vectors are not forced to be orthogonal or sorted. The network weights can be rotated or scaled in infinitely many ways within that subspace while achieving the exact same minimum reconstruction loss.
    

## 4. Metric Representation Learning: Triplet Loss

Another powerful approach to self-supervised representation learning is forcing the network to group semantically similar items together in the feature space while pushing dissimilar items far away. This is optimized using **Triplet Loss**.

### The Triplet Structure

We construct training combinations containing three distinct instances:

1. **Anchor ($\mathbf{x}_a$):** A base reference sample.
    
2. **Positive ($\mathbf{x}_p$):** A sample semantically similar to the anchor (e.g., another image of the same class).
    
3. **Negative ($\mathbf{x}_n$):** A sample structurally different from the anchor.
    

### The Objective Function

Let $f(\cdot)$ be the embedding network projection function. We optimize the network parameters using a Hinge-style margin loss criteria:

$$\mathcal{L}(a, p, n) = \max\left(0, \|f(\mathbf{x}_a) - f(\mathbf{x}_p)\|^2 - \|f(\mathbf{x}_a) - f(\mathbf{x}_n)\|^2 + \alpha \right)$$

where $\alpha > 0$ is a constant safety margin parameter.

This loss function is zero only when the squared distance between the anchor and the negative sample exceeds the distance between the anchor and the positive sample by at least the margin value $\alpha$.

---
## 🔗 Navigation
**Previous:** [[ ]] | **Next:** [[ ]]