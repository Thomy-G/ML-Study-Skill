---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Summary
date_added: 2026-06-21
---
# Topic 21 — Structural Attention Frameworks - Transformers (Syllabus Item 13)

**MOC:** [[Machine learning MOC]]
**Course:** [[English]]


# Study Guide: Topic 21 — Structural Attention Frameworks & Transformers (Syllabus Item 13)

## 1. Context: Overcoming the Sequential Bottleneck of RNNs

In Topic 16, we explored Recurrent Neural Networks (RNNs and LSTMs) for handling variable-length sequential data. While highly effective at capturing temporal structures, RNNs suffer from a major computational limitation: **they must process tokens sequentially step-by-step.** Because the hidden state at temporal step $t$ requires the computed hidden state from step $t-1$, training cannot be parallelized across the sequence length. This creates a massive computational bottleneck when scaling up to large datasets.

The **Transformer architecture** eliminates this sequential bottleneck entirely by replacing recurrence loops with parallel **Self-Attention mechanisms**. This architecture processes all tokens in a sequence simultaneously, maximizing hardware acceleration on modern GPU clusters.

## 2. Representation Shift: Word Embeddings vs. Contextual Embeddings

Before diving into attention math, it is crucial to understand how representations transform through a Transformer block:

- **Static Word Embeddings (Input Space):** Raw text tokens are initially mapped to continuous vectors using a static lookup table (e.g., Word2Vec styles). In this initial space, a word like _"bank"_ has the exact same static vector representation whether it appears in _"river bank"_ or _"investment bank"_.
    
- **Contextual Embeddings (Transformer Output Space):** After passing through the self-attention layers, the static embeddings are dynamically mixed. The final output vectors are fully contextualized—meaning the token vector for _"bank"_ changes dynamically based on the surrounding semantic neighborhood of the entire input sequence.
    

## 3. Scaled Dot-Product Self-Attention

The core engine of a Transformer is the Self-Attention mechanism. It allows a model to calculate the semantic relationship between any two words in a sequence, regardless of their physical distance.


```
       Query (Q)    Key (K)
           │          │
           ▼          ▼
     ┌──────────────────────┐
     │ Pairwise Dot Product │
     └──────────┬───────────┘
                │
                ▼ [Divide by sqrt(d_k)]
     ┌──────────────────────┐
     │  Scale Normalization │
     └──────────┬───────────┘
                │
                ▼
     ┌──────────────────────┐
     │       Softmax        ├────► Attention Weight Matrix (A)
     └──────────┬───────────┘
                │                    Value (V)
                └─────────┬──────────────┘
                          ▼
                    ┌──────────┐
                    │  Output  │
                    └──────────┘
```


Given an input sequence representation matrix $H \in \mathbb{R}^{L \times d}$ (where $L$ is sequence length and $d$ is embedding dimension), we project each token embedding into three separate vector representations using learned weight matrices ($W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$):

1. **Queries ($Q = H W_Q \in \mathbb{R}^{L \times d_k}$):** Represents the current token actively searching for relevant context.
    
2. **Keys ($K = H W_K \in \mathbb{R}^{L \times d_k}$):** Acts as a descriptor index vector that matches against incoming queries.
    
3. **Values ($V = H W_V \in \mathbb{R}^{L \times d_v}$):** Contains the actual feature content to be extracted once a match is found.
    

### The Attention Equation

The contextual representations for the entire sequence are computed simultaneously using a single vectorized matrix expression:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$

### Rationale Behind the Components

- **$Q K^T$ Product Matrix:** Computes the raw similarity score between every pair of tokens in the sequence, producing a geometric compatibility attention weight matrix of shape $(L \times L)$.
    
- **Scaling Factor $\frac{1}{\sqrt{d_k}}$:** Prevents the dot products from growing excessively large in high dimensions. Without this scaling, large dot products would push the Softmax function into flat regions with vanishingly small gradients, destabilizing gradient descent during backpropagation.
    
- **Softmax Activation:** Normalizes the similarity scores into a valid probability distribution per row, determining how much focus (attention weight) each token should place on every other word in the sequence.
    

## 4. The Complete Transformer Block Components

To build a fully deep sequence model, individual Scaled Dot-Product layers are augmented with several essential architectural components inside a unified block:

### A. Multi-Head Attention

Instead of performing attention once over the full dimension, the model projects $Q, K, V$ into $h$ smaller subspaces (heads) and runs attention in parallel. The outputs from each head are then concatenated and projected back to the original dimension:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

This enables the network to track different types of contextual relationships simultaneously (e.g., tracking a verb's subject with one head, while tracking prepositional modifiers with another).

### B. Positional Encoding

Because self-attention processes all tokens simultaneously using pure matrix multiplication, it is inherently invariant to word order (treating sequences like an unordered bag of words). To preserve sequential structure, we add a deterministic or learned wave vector—the **Positional Encoding**—directly to the input token embeddings before the first layer:

$$H_{\text{initial}} = X_{\text{word\_embeddings}} + P_{\text{positional\_encodings}}$$

### C. Position-Wise Feed-Forward Networks (FFN)

Directly after the attention weights mix the token context, each position is processed independently and identically by a small, fully connected Feed-Forward Network layer. This introduces additional non-linear capacity to the model:

$$\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x}W_1 + \mathbf{b}_1)W_2 + \mathbf{b}_2$$

### D. Regularization & Stabilization: Dropout and LayerNorm

To train ultra-deep architectures without encountering vanishing gradients or severe overfitting, Transformers heavily rely on two deep learning primitives:

1. **Residual (Skip) Connections:** Every sub-layer (Attention and FFN) is wrapped inside a skip connection ($x + \text{SubLayer}(x)$) allowing raw gradients to flow backwards completely unhindered.
    
2. **Layer Normalization (LayerNorm):** Applied immediately after the residual summation. It normalizes features across the channels of an individual sample, smoothing the optimization landscape:
    
    $$\mathbf{y} = \text{LayerNorm}(\mathbf{x} + \text{SubLayer}(\mathbf{x}))$$
    
3. **Dropout:** Applied to the output of each sub-layer (right before the residual addition) and to the attention weights themselves, randomly deactivating neurons to prevent complex co-adaptations and overfitting (**Syllabus Item 9**).
    

## 5. Python / PyTorch Implementation Snippet

This block shows how to define a complete, non-linear Transformer encoder block incorporating all architectural layers:

```python
import torch
import torch.nn as nn

class CompleteTransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(CompleteTransformerEncoderBlock, self).__init__()
        # 1. Multi-Head Attention Mechanism
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        # 2. Position-wise Feed Forward Network Layer
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, embed_dim)
        )
        
        # 3. Stabilization and Regularization Blocks
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Forward through Multihead Self-Attention with Residual & LayerNorm
        attn_out, _ = self.attention(x, x, x) # Q=x, K=x, V=x
        attn_out = self.dropout1(attn_out)
        x = self.layernorm1(x + attn_out) # Residual add + Normalization
        
        # Forward through Position-wise FFN with Residual & LayerNorm
        ffn_out = self.ffn(x)
        ffn_out = self.dropout2(ffn_out)
        out = self.layernorm2(x + ffn_out) # Residual add + Normalization
        return out
```

## 6. Solved Exam-Style Examples

### Example 1: Computational Complexity Analysis (From Past Exams)

**Problem Statement:** Analyze the computational time complexity of a single Scaled Dot-Product Self-Attention layer processing an input sequence of length $L$ with an embedding dimension of $d$. Compare its scalability profile to a standard Recurrent Neural Network (RNN) step.

#### Step-by-Step Analytical Derivation

1. **Compute $Q, K, V$ Matrices:** Multiplying the input matrix $H \in \mathbb{R}^{L \times d}$ by the weight matrices $W \in \mathbb{R}^{d \times d}$ requires:
    
    $$\mathcal{O}(L \cdot d^2) \quad \text{operations for each matrix (Total: } 3 \cdot L \cdot d^2)$$
    
2. **Compute the Similarity Matrix $Q K^T$:** Multiplying a matrix of shape $(L \times d)$ by a matrix of shape $(d \times L)$ results in an $(L \times L)$ matrix. The complexity of this matrix operation scales as:
    
    $$\mathcal{O}(L^2 \cdot d)$$
    
3. **Apply Softmax and Multiply by $V$:** Multiplying the normalized $(L \times L)$ attention matrix by the $(L \times d)$ values matrix requires:
    
    $$\mathcal{O}(L^2 \cdot d) \quad \text{operations.}$$
    
4. **Total Complexity:** Combining these steps yields a total time complexity of **$\mathcal{O}(L \cdot d^2 + L^2 \cdot d)$**.
    

- **Comparison with RNNs:** An RNN processes sequence tokens one by one sequentially. For a sequence of length $L$, it executes $L$ sequential hidden updates, each requiring a matrix-vector multiplication scaling as $\mathcal{O}(d^2)$. This results in a time complexity of **$\mathcal{O}(L \cdot d^2)$**.
    
- **The Scalability Tradeoff:** When the sequence length is small ($L \ll d$), the Transformer is highly efficient and trains much faster than an RNN due to full GPU parallelization. However, because the attention matrix maps every token to every other token, the complexity scales quadratically with sequence length ($\mathcal{O}(L^2)$). For exceptionally long sequences ($L \gg d$), the quadratic footprint can become a massive computational bottleneck.
    

### Example 2: Permutation Invariance Proof

**Problem Statement:** Mathematically prove that without Positional Encodings, the output of a Scaled Dot-Product Self-Attention layer is completely **invariant to permutations** of the input sequence.

#### Step-by-Step Analytical Derivation

1. Let $P \in \mathbb{R}^{L \times L}$ be an arbitrary permutation matrix. A permutation matrix is strictly orthogonal, meaning $P^T P = P P^T = I$.
    
2. Applying this permutation matrix to our input sequence matrix $H$ reorders its rows: $\tilde{H} = P H$.
    
3. Compute the permuted Query, Key, and Value matrices:
    
    $$\tilde{Q} = \tilde{H}W_Q = PHW_Q = PQ$$
    
    $$\tilde{K} = \tilde{H}W_K = PHW_K = PK$$
    
    $$\tilde{V} = \tilde{H}W_V = PHW_V = PV$$
    
4. Substitute these permuted matrices into the Self-Attention equation:
    
    $$\text{Attention}(\tilde{Q}, \tilde{K}, \tilde{V}) = \text{Softmax}\left( \frac{\tilde{Q}\tilde{K}^T}{\sqrt{d_k}} \right)\tilde{V} = \text{Softmax}\left( \frac{(PQ)(PK)^T}{\sqrt{d_k}} \right)(PV)$$
    
5. Expand the transposed product term inside the Softmax function ($(PK)^T = K^T P^T$):
    
    $$\text{Softmax}\left( \frac{P Q K^T P^T}{\sqrt{d_k}} \right) P V$$
    
6. Because the Softmax function operates independently across each row, permuting the rows beforehand with $P$ and then applying Softmax is mathematically equivalent to applying Softmax first and then permuting the final rows: $\text{Softmax}(P M P^T) = P \text{Softmax}(M) P^T$.
    
7. Substitute this relation back into the expression:
    
    $$\left( P \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) P^T \right) P V$$
    
8. Simplify the equation using the orthogonality identity ($P^T P = I$):
    
    $$P \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) I V = P \left[ \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V \right] = P \cdot \text{Attention}(Q, K, V)$$
    

**Final Exam Answer:** This proves that permuting the input sequence by $P$ simply permutes the exact same output rows by $P$, without changing or mixing the internal contextual mappings. Without an explicit Positional Encoding to break this directional symmetry, **a Transformer processes a sequence as an unordered bag of words**, ignoring temporal structures.

---
## 🔗 Navigation
**Previous:** [[ ]] | **Next:** [[ ]]