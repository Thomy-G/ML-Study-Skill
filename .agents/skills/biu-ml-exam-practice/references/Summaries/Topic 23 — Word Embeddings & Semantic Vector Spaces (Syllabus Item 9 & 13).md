---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Summary
date_added: 2026-06-22
---
# Topic 23 — Word Embeddings & Semantic Vector Spaces (Syllabus Item 9 & 13)

**MOC:** [[Machine learning MOC]]
**Course:** [[English]]

# Study Guide: Topic 23 — Word Embeddings & Semantic Vector Spaces (Syllabus Item 9 & 13)

## 1. Context: Representing Discrete Tokens as Continuous Vectors

Machine learning architectures are fundamentally mathematical vector engines—they can only compute gradients and optimize loss surfaces over continuous real numbers. They cannot natively process raw, discrete categorical strings (words).

The simplest representation, **One-Hot Encoding**, maps each word to a sparse vector of vocabulary size $V$, where a single index contains a $1$ and all other elements are $0$. However, One-Hot vectors suffer from two major flaws:

1. **Catastrophic Dimensionality:** If a vocabulary contains 100,000 unique tokens, every individual word vector requires 100,000 dimensions ($W \in \mathbb{R}^{100,000}$), creating an immense parameter footprint.
    
2. **Semantic Orthogonality:** The inner dot product of any two distinct one-hot vectors is always exactly zero ($\mathbf{w}_{\text{cat}}^T \mathbf{w}_{\text{dog}} = 0$). This means the representation completely fails to capture semantic relationships; the word _"cat"_ is mathematically just as distant from _"dog"_ as it is from _"refrigerator"_.
    

**Word Embeddings** resolve this by learning a dense, low-dimensional vector space mapping $\mathbf{w} \in \mathbb{R}^d$ (where $d \ll V$, typically $d=300$) where the geometric proximity between vectors directly reflects their semantic similarity.

## 2. The Distributional Hypothesis & Word2Vec

Word embedding frameworks rely on the **Distributional Hypothesis**: _Words that appear in similar textual contexts tend to share similar semantic meanings._

### The Skip-Gram Architecture

The **Skip-Gram model** (a core component of Word2Vec) formalizes this by turning an unlabeled text corpus into a self-supervised prediction task. Given a target center word $w_t$, the network is trained to maximize the conditional probability of predicting its surrounding context words $w_{t+j}$ within a sliding window of size $C$.

### The Probabilistic Formulation

For each word type in our vocabulary, the model learns two separate vector representations: an input vector $\mathbf{v}_w$ (when the word acts as a center word) and an output vector $\mathbf{u}_w$ (when it acts as a context word). The conditional probability of observing context word $c$ given center word $w$ is modeled using the Softmax function:

$$P(c \mid w) = \frac{\exp(\mathbf{u}_c^T \mathbf{v}_w)}{\sum_{j=1}^V \exp(\mathbf{u}_j^T \mathbf{v}_w)}$$

The total objective function maximizes the log-probability across the entire text corpus:

$$\max \sum_{t=1}^T \sum_{-C \leq j \leq C, j \neq 0} \ln P(w_{t+j} \mid w_t)$$

## 3. Geometric Vector Properties (Linear Subspace Analogies)

A powerful emergent property of dense word embedding spaces is that semantic relationships translate directly into linear algebraic vector displacements.

For example, the vector difference between the word vector for _"man"_ and _"woman"_ aligns nearly perfectly with the vector difference between _"king"_ and _"queen"_:

$$\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$$

This vector alignment allows semantic relationships (such as gender transitions, verb tenses, or country-capital pairings) to be navigated using simple, unsupervised linear vector arithmetic.


---
## 🔗 Navigation
**Previous:** [[ ]] | **Next:** [[ ]]