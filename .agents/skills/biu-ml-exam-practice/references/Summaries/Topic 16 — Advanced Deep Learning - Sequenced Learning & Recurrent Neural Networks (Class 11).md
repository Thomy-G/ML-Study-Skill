---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Summary
date_added: 2026-06-21
---
# Topic 16 — Advanced Deep Learning - Sequenced Learning & Recurrent Neural Networks (Class 11)

**MOC:** [[Machine learning MOC]]
**Course:** [[English]]

# Study Guide: Topic 16 — Advanced Deep Learning: Sequenced Learning & Recurrent Neural Networks (Class 11)

## 1. Context: Moving Beyond Static Structural Dimensions

Until now, all parametric architectures explored (Ordinary MLPs, Convolutional Networks) processed inputs of a **static, predefined size**. They assumed that data instances were statistically independent of one another.

However, many critical real-world phenomena consist of **sequential streams** where inputs have variable lengths and exhibit rich temporal or contextual dependencies over time (e.g., natural language processing, audio signals, time-series forecasting, or genetic strands).

A standard feedforward network fails on sequence data for two main structural reasons:

1. **Fixed-Dimension Constraints:** An MLP requires a fixed-size input layer, making it incapable of natively adapting to sequences of varying lengths.
    
2. **Context Amnesia:** Feedforward layers do not maintain an internal memory state across sequential steps. They process a word at position $t$ with completely unshared weight connections relative to position $t+1$.
    

## 2. Recurrent Neural Networks (RNNs)

A **Recurrent Neural Network (RNN)** handles sequential dependencies by introducing feedback loops into its hidden layers. As it steps through a sequence, it maintains a dynamic internal vector—the **hidden state** $\mathbf{h}_t$—which acts as the network's working memory, summarizing information from all past time steps.

```
Unfolded Temporal Computation Graph:

  x_(t-1)         x_t           x_(t+1)
     │             │               │
     ▼ [W_xh]      ▼ [W_xh]        ▼ [W_xh]
 ┌───┴───┐     ┌───┴───┐       ┌───┴───┐
─┤h_(t-1)├────►│  h_t  ├──────►│h_(t+1)├─►
 └───────┘     └───┴───┘       └───┬───┘
    [W_hh]       │ [W_hh]          │ [W_hy]
                 ▼ [W_hy]          ▼
                y_hat_t        y_hat_(t+1)
```

### The Recurrent Execution Equations

Let $\mathbf{x}_t \in \mathbb{R}^d$ be the input feature vector at time step $t$. Let $\mathbf{h}_t \in \mathbb{R}^K$ be the hidden memory state vector. The system shares three main weight matrices ($W_{xh}, W_{hh}, W_{hy}$) across all time steps:

1. **Hidden State Update:** The current hidden state is computed by combining the current input vector with the _previous_ hidden state vector, passed through a non-linear activation function (typically $\tanh$):
    
    $$\mathbf{z}_t = W_{xh}\mathbf{x}_t + W_{hh}\mathbf{h}_{t-1} + \mathbf{b}_h$$
    
    $$\mathbf{h}_t = \tanh\left(\mathbf{z}_t\right)$$
    
2. **Output Logit Layer:** Compute the sequence prediction vector $\hat{\mathbf{y}}_t$ at the current step:
    
    $$\hat{\mathbf{y}}_t = \text{Softmax}\left(W_{hy}\mathbf{h}_t + \mathbf{b}_y\right)$$
    

## 3. Training Sequences: Backpropagation Through Time (BPTT)

To optimize the shared parameter matrices ($W_{xh}, W_{hh}, W_{hy}$), we compute gradients by unfolding the recurrent network computationally across all temporal steps, a technique called **Backpropagation Through Time (BPTT)**.

### The Vanishing & Exploding Gradient Problem

The greatest challenge when training standard RNNs over long sequences stems from the recursive definition of the hidden state gradient. When calculating the derivative of the loss at step $T$ with respect to the initial weight matrix $W_{hh}$, the chain rule requires multiplying the internal hidden gradients sequentially:

$$\frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_T} \prod_{t=2}^T \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_T} \prod_{t=2}^T \text{diag}\Big(1 - \tanh^2(\mathbf{z}_t)\Big) W_{hh}^T$$

- **Vanishing Gradients:** If the largest eigenvalue of $W_{hh}$ is less than 1 ($\lambda_{\max} < 1$), multiplying it repeatedly causes the gradient vector to decay exponentially toward zero as the sequence length grows. The network completely forgets long-term historical dependencies.
    
- **Exploding Gradients:** If $\lambda_{\max} > 1$, the gradient values grow exponentially, causing parameter updates to oscillate wildly and destabilize training. _Remediation: Gradient Clipping._
    

## 4. Gated Architectures: Long Short-Term Memory (LSTM)

To solve the vanishing gradient problem and capture long-term dependencies, advanced architectures introduce **gated memory channels**. The most famous of these is the **Long Short-Term Memory (LSTM)** network.

LSTMs add an isolated, continuous highway called the **Cell State** ($\mathbf{C}_t$). Information flows down this highway with minimal linear interactions, preventing gradients from vanishing. The flow of information is regulated by three distinct, learned gating mechanisms:

1. **Forget Gate ($\mathbf{f}_t$):** Controls what proportion of the past cell state history to discard:
    
    $$\mathbf{f}_t = \sigma\left(W_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f\right)$$
    
2. **Input Gate ($\mathbf{i}_t$):** Controls what new information from the current step should be written into the cell state:
    
    $$\mathbf{i}_t = \sigma\left(W_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i\right)$$
    
    $$\tilde{\mathbf{C}}_t = \tanh\left(W_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c\right)$$
    
3. **Cell State Update ($\mathbf{C}_t$):** Combines the gated past memory with the gated new input:
    
    $$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$$
    
4. **Output Gate ($\mathbf{o}_t$):** Controls what information from the updated cell state should be exposed as the final hidden state $\mathbf{h}_t$:
    
    $$\mathbf{o}_t = \sigma\left(W_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o\right)$$
    
    $$\mathbf{h}_t = \mathbf{o}_t \odot \tanh\left(\mathbf{C}_t\right)$$
    

## 5. Python / PyTorch Implementation Snippet

This functional block illustrates how to implement a recurrent LSTM layer for sequence classification using PyTorch:

```python
import torch
import torch.nn as nn

class SequenceClassifierLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SequenceClassifierLSTM, self).__init__()
        # PyTorch handles the complete gating arithmetic internally
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (Batch_Size, Sequence_Length, Input_Dim)
        # lstm_out shape: (Batch_Size, Sequence_Length, Hidden_Dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Pull the final hidden state at the last step of the sequence
        last_hidden_state = h_n[-1] # shape: (Batch_Size, Hidden_Dim)
        
        logits = self.fc(last_hidden_state)
        return logits
```

## 6. Solved Exam-Style Examples

### Example 1: Trace Analysis of RNN Parameter Shaking

**Problem Statement:** Consider a basic RNN processing a sequence of length $T=3$. The weights are scalars: $W_{xh} = 1.0, W_{hh} = 2.0, W_{hy} = 1.5$. All bias scalars are zero, and the network uses a strictly linear activation function ($g(z)=z$).

If the input sequence is $x_1 = 1, x_2 = 0, x_3 = -1$, and the initial hidden state is $\mathbf{h}_0 = 0$, compute the exact analytical value of the final output prediction $\hat{y}_3$.

#### Step-by-Step Analytical Derivation

1. **Time step $t=1$:**
    
    $$z_1 = W_{xh}x_1 + W_{hh}h_0 = (1.0)(1) + (2.0)(0) = 1.0$$
    
    $$h_1 = g(z_1) = 1.0$$
    
2. **Time step $t=2$:**
    
    $$z_2 = W_{xh}x_2 + W_{hh}h_1 = (1.0)(0) + (2.0)(1.0) = 2.0$$
    
    $$h_2 = g(z_2) = 2.0$$
    
3. **Time step $t=3$:**
    
    $$z_3 = W_{xh}x_3 + W_{hh}h_2 = (1.0)(-1) + (2.0)(2.0) = -1.0 + 4.0 = 3.0$$
    
    $$h_3 = g(z_3) = 3.0$$
    
4. **Compute final linear output prediction:**
    
    $$\hat{y}_3 = W_{hy}h_3 = (1.5)(3.0) = 4.5$$
    

**Final Exam Answer:** The exact analytical value of the final sequence prediction output is **$\hat{y}_3 = 4.5$**.

### Example 2: Structural Contrast of ConvNets vs. RNNs (From Past Exams)

**Problem Statement:** Contrast how a 1D Convolutional Neural Network (ConvNet) and a Recurrent Neural Network (RNN) extract patterns from a 1D sequence of features. Explain how their structural assumptions impact their receptive field ranges and parameter scaling behaviors.

#### Analytical Evaluation

- **Receptive Field Span:** * A single 1D convolutional layer captures local dependencies bounded strictly by its window kernel width ($k$). To capture long-range interactions, multiple layers must be stacked hierarchically, increasing depth.
    
    - An RNN inherently processes the sequence iteratively, allowing its hidden state to pass information down across an unconstrained sequence length, creating an unbounded theoretical receptive field within a single layer.
        
- **Parameter Scaling & Computational Flow:**
    
    - Both models use parameter sharing. ConvNets slide a kernel across the input space, allowing steps to be computed completely in parallel. This makes them highly optimized for hardware acceleration (GPUs).
        
    - RNNs share weights across temporal steps sequentially ($h_{t-1} \rightarrow h_t$). Because step $t$ cannot be computed until step $t-1$ finishes, training cannot be easily parallelized across sequence paths, leading to higher computational training bottlenecks for long sequences.
        


---
## 🔗 Navigation
**Previous:** [[ ]] | **Next:** [[ ]]