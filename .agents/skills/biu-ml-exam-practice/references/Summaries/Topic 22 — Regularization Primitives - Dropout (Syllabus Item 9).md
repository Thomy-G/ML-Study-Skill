---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Summary
date_added: 2026-06-22
---
 Study Guide - Topic 22 — Regularization Primitives - Dropout (Syllabus Item 9)

**MOC:** [[Machine learning MOC]]
**Course:** [[English]]

## 1. Context: The Overfitting Problem in Deep Networks

Deep Neural Networks (like deep MLPs, ConvNets, and Transformers) contain millions of trainable parameters. Because of this massive expressive capacity, these models are highly susceptible to severe overfitting (**high variance**). The hidden units can easily develop complex **co-adaptations**—a pathological state where a neuron only generates useful representations when specific other neurons are active simultaneously, essentially memorizing noisy training patterns instead of learning generalizable features.

While $L_2$ regularization (Weight Decay) restricts parameter magnitudes, **Dropout** is a powerful regularization technique specifically designed to break up these mutual neuron dependencies by structurally modifying the network topology during training.

## 2. Mathematical Formalization of Dropout

During the training phase, Dropout randomly deactivates (drops out) a subset of neurons in a layer with a pre-configured probability $p$ (e.g., $p=0.5$).

### A. The Training Phase Forward Pass

Let $\mathbf{h} \in \mathbb{R}^K$ be the activation vector output of a hidden layer. For each training sample and each training forward pass:

1. Generate a random mask vector $\mathbf{m} \in \{0, 1\}^K$, where each element is an independent Bernoulli random variable:
    
    $$m_j \sim \text{Bernoulli}(1 - p)$$
    
2. Apply the mask element-wise to the hidden layer activations using the Hadamard product ($\odot$):
    
    $$\tilde{\mathbf{h}} = \mathbf{h} \odot \mathbf{m}$$

    > 💡 **Mathematical Note: The Hadamard Product ($\odot$)**  
    > The **Hadamard product** (also called element-wise multiplication) multiplies corresponding entries of two matrices or vectors of identical dimensions: $(\mathbf{h} \odot \mathbf{m})_j = h_j \cdot m_j$. Unlike standard matrix multiplication ($W \mathbf{x}$), the Hadamard product requires identical shapes, is commutative ($\mathbf{a} \odot \mathbf{b} = \mathbf{b} \odot \mathbf{a}$), and is executed via the `*` operator in NumPy/PyTorch.

3. Pass the regularized vector $\tilde{\mathbf{h}}$ to the next layer in the computational graph. Because a random half of the hidden units are silenced at each step, neurons are forced to learn robust features that are useful independently of their neighbors.
    

### B. The Inference (Test) Phase Forward Pass

During inference, we want our predictions to be deterministic and stable, so **no units are dropped** ($\mathbf{m} = \mathbf{1}$). However, because all neurons are now active simultaneously, the total expected input sum flowing into the next layer would be higher than it was during training.

To maintain numerical scale alignment between phases, we must scale the static weights or activations down proportionally by the survival probability ($1-p$):

$$\mathbf{h}_{\text{inference}} = (1 - p) \cdot \mathbf{h}$$

### C. Inverted Dropout (Modern PyTorch Implementation)

To avoid performing extra multiplications at inference time, modern frameworks use **Inverted Dropout**. This method scales the activations _up_ during the training phase instead:

$$\tilde{\mathbf{h}}_{\text{training}} = \frac{\mathbf{h} \odot \mathbf{m}}{1 - p}$$

This ensures that at test time, the activations can be evaluated normally without any modification.




---
## 🔗 Navigation
**Previous:** [[ ]] | **Next:** [[ ]]