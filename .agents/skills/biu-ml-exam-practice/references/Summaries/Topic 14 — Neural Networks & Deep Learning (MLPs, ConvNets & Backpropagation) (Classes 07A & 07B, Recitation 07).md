---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Summary
date_added: 2026-06-21
---
# Topic 14 — Neural Networks & Deep Learning (MLPs, ConvNets & Backpropagation) (Classes 07A & 07B, Recitation 07)

**MOC:** [[Machine learning MOC]]
**Course:** [[English]]

# Study Guide: Topic 14 — Neural Networks & Deep Learning (MLPs, ConvNets & Backpropagation) (Classes 07A & 07B, Recitation 07)

## 1. Context: Moving Beyond Linear Separability

Linear classifiers (like the Perceptron or Logistic Regression) can only find a single straight decision boundary (hyperplane). However, real-world data is often highly non-linear, meaning classes overlap or wrap around one another in ways that a straight line cannot separate.

While **Kernel methods** solve this by using a fixed mapping to project data into a higher-dimensional space, **Deep Neural Networks** solve this by _learning_ non-linear hierarchical representations directly from the raw data.

## 2. Multi-Layer Perceptrons (MLPs)

An MLP is a fully connected feedforward artificial neural network consisting of an input layer, one or more hidden layers, and an output layer.

```
Inputs (x)          Hidden Layer (h)          Output (y_hat)
   ┌───┐                 ┌───┐
   │ x1├───┐         ┌───┤ h1├─┐
   └───┘   │ ┌───┐   │   └───┘ │   ┌───┐
           ├─┤MUL├───┤         ├───┤   │
   ┌───┐   │ └───┘   │   ┌───┐ │   │   │
   │ x2├───┘         └───┤ h2├─┼───┤MAX├───► Prediction
   └───┘                 └───┘ │   │   │
                               │   │   │
   ┌───┐                 ┌───┐ │   └───┘
   │ 1 ├─────────────────┤ h3├─┘
 (Bias)                  └───┘
```

### What is a Hidden Layer?
A **hidden layer** is an intermediate layer of artificial neurons (hidden units) positioned between the input layer (which receives raw features $\mathbf{x}$) and the output layer (which produces final predictions $\hat{y}$). They are called *"hidden"* because their internal activation states are computed internally by the model rather than being directly observed in the training dataset.

* **Feature Space Transformation**: Linear models can only draw straight decision hyperplanes. A hidden layer projects raw features $\mathbf{x} \in \mathbb{R}^d$ into an internal representation $\mathbf{h} \in \mathbb{R}^K$, warping the feature space so complex non-linear class boundaries become linearly separable.
* **Hierarchical Abstraction**: In deep networks, lower hidden layers learn low-level primitive features (e.g. edges, corners), while deeper hidden layers combine them into high-level conceptual abstractions (e.g. shapes, objects).
* **Universal Approximation Theorem**: A single hidden layer containing a sufficient number of non-linear hidden units can approximate any continuous function to arbitrary precision.

### The Layer Equations (Forward Pass)

For a network with a single hidden layer containing $K$ hidden units, an input vector $\mathbf{x} \in \mathbb{R}^d$, a hidden weight matrix $W^{(1)} \in \mathbb{R}^{K \times d}$, and hidden bias vector $\mathbf{b}^{(1)} \in \mathbb{R}^K$:

1. **Linear Logit Activation:** Compute the raw linear combination for the hidden layer:
    
    $$\mathbf{z}^{(1)} = W^{(1)}\mathbf{x} + \mathbf{b}^{(1)}$$
    
2. **Non-Linear Activation Function:** Pass the logits through an element-wise activation function $g(\cdot)$ to form the hidden feature representations $\mathbf{h} \in \mathbb{R}^K$:
    
    $$\mathbf{h} = g\left(\mathbf{z}^{(1)}\right)$$
    
    _Without a non-linear activation function, stacking multiple layers would mathematically collapse back into a single simple linear transformation._
    
3. **Output Computation:** Compute the final prediction $\hat{y}$ using output weights $\mathbf{w}^{(2)} \in \mathbb{R}^K$ and output bias $b^{(2)}$:
    
    $$\hat{y} = (\mathbf{w}^{(2)})^T \mathbf{h} + b^{(2)}$$
    

### Core Activation Functions

1. **Sigmoid:** $\sigma(z) = \frac{1}{1 + e^{-z}}$ (Maps outputs to $(0,1)$; prone to vanishing gradients when $z \gg 0$ or $z \ll 0$).
2. **ReLU (Rectified Linear Unit):** $\text{ReLU}(z) = \max(0, z)$ (Computationally highly efficient; resolves vanishing gradients for positive inputs).

#### Deep Dive: ReLU Mechanics, Limitations & Variants

* **Sub-gradient Non-Differentiability at $z = 0$:** Strictly speaking, $\text{ReLU}(z)$ is not differentiable at $z=0$. By convention in automatic differentiation frameworks (PyTorch/TensorFlow), a sub-gradient is used:
  $$\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$$

* **💀 The "Dying ReLU" Problem (Dead Neurons):** If a large gradient update causes a neuron's weights to adjust such that $z = \mathbf{w}^T\mathbf{x} + b < 0$ for all dataset instances:
  * The activation is permanently $0$ ($\text{ReLU}(z) = 0$).
  * The derivative is permanently $0$ ($\text{ReLU}'(z) = 0$).
  * No gradient can ever flow back through this neuron during backpropagation ($\boldsymbol{\delta} = \mathbf{0}$). Its weights will **never update again**, rendering the neuron permanently "dead".

* **Relieving the Dying ReLU: Activation Variants:**
  1. **Leaky ReLU:** Introduces a small positive slope $\alpha \approx 0.01$ for negative inputs:
     $$\text{Leaky\_ReLU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha z & \text{if } z \le 0 \end{cases}$$
     *Derivative for $z \le 0$ is $\alpha > 0$, ensuring gradients never drop to zero.*
  2. **Parametric ReLU (PReLU):** Treats $\alpha$ as a learnable parameter updated dynamically via backpropagation.
  3. **ELU (Exponential Linear Unit):** $f(z) = z$ if $z > 0$, else $\alpha(e^z - 1)$ if $z \le 0$. Smoothly approaches $-\alpha$, bringing mean activations closer to zero.
  4. **GELU (Gaussian Error Linear Unit):** Weighting inputs by the Gaussian CDF. Standard activation function used in modern **Transformers (GPT, BERT)**.
    

## 3. Backpropagation (Reverse-Mode Automatic Differentiation)

**Backpropagation** is the foundational algorithm used to train neural networks. It computes the exact analytical gradient of a scalar Loss function $\mathcal{L}$ with respect to every weight matrix $W^{(l)}$ and bias vector $\mathbf{b}^{(l)}$ across all layers of the network by applying the multivariable calculus **Chain Rule** backward from the output layer to the input layer.

### 3.1 Intuition: Forward Pass vs. Backward Pass

1. **Forward Pass (Prediction):** Raw inputs $\mathbf{x}$ are passed forward through the network layer by layer to compute intermediate activations $\mathbf{z}^{(l)}, \mathbf{a}^{(l)}$ and eventually the prediction $\hat{\mathbf{y}}$, which is evaluated by the loss function $\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y})$.
2. **Backward Pass (Credit Assignment):** The algorithm measures the final output error and propagates error signals $\boldsymbol{\delta}^{(l)}$ backward through the computational graph. Each layer receives error gradients from the subsequent layer, computes its local parameter gradients, and passes the scaled error further back to earlier layers.

---

### 3.2 The 4 Fundamental Equations of Backpropagation

Let $\mathbf{z}^{(l)} = W^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$ be the raw logits at layer $l$, where $\mathbf{a}^{(l)} = g(\mathbf{z}^{(l)})$ is the activation vector, and $g'(\cdot)$ is the derivative of the activation function. 

We define the **error vector** at layer $l$ as $\boldsymbol{\delta}^{(l)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}}$.

1. **Output Layer Error ($\boldsymbol{\delta}^{(L)}$):**
   $$\boldsymbol{\delta}^{(L)} = \nabla_{\mathbf{a}^{(L)}} \mathcal{L} \odot g'\left(\mathbf{z}^{(L)}\right)$$
   *(Measures how fast the loss changes with respect to output layer logits).*

2. **Error Backpropagation Equation ($\boldsymbol{\delta}^{(l-1)}$ in terms of $\boldsymbol{\delta}^{(l)}$):**
   $$\boldsymbol{\delta}^{(l-1)} = \left( (W^{(l)})^T \boldsymbol{\delta}^{(l)} \right) \odot g'\left(\mathbf{z}^{(l-1)}\right)$$
   *(Takes the error $\boldsymbol{\delta}^{(l)}$ from layer $l$, projects it back through the weight matrix $(W^{(l)})^T$, and scales it by the local activation derivative $g'(\mathbf{z}^{(l-1)})$).*

3. **Gradient with respect to Biases ($\nabla_{\mathbf{b}^{(l)}} \mathcal{L}$):**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}$$

4. **Gradient with respect to Weights ($\nabla_{W^{(l)}} \mathcal{L}$):**
   $$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \boldsymbol{\delta}^{(l)} \left(\mathbf{a}^{(l-1)}\right)^T$$
   *(The weight gradient matrix is formed by the outer product of the layer's error vector $\boldsymbol{\delta}^{(l)}$ and the incoming activation vector from the previous layer).*

---

### 3.3 Step-by-Step Derivation for a 2-Layer MLP

Consider a single training instance $(\mathbf{x}, y)$ with a 2-layer network using Mean Squared Error loss $\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$:
* Layer 1: $\mathbf{z}^{(1)} = W^{(1)}\mathbf{x} + \mathbf{b}^{(1)}, \quad \mathbf{h} = \text{ReLU}\left(\mathbf{z}^{(1)}\right)$
* Layer 2: $z^{(2)} = (\mathbf{w}^{(2)})^T \mathbf{h} + b^{(2)}, \quad \hat{y} = \sigma\left(z^{(2)}\right)$

#### Step 1: Derivative of Loss w.r.t. Output Logit $\delta^{(2)}$
$$\delta^{(2)} = \frac{\partial \mathcal{L}}{\partial z^{(2)}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z^{(2)}} = (\hat{y} - y) \cdot \sigma'\left(z^{(2)}\right)$$

#### Step 2: Gradients for Output Parameters ($\mathbf{w}^{(2)}, b^{(2)}$)
$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}^{(2)}} = \delta^{(2)} \mathbf{h}, \qquad \frac{\partial \mathcal{L}}{\partial b^{(2)}} = \delta^{(2)}$$

#### Step 3: Backpropagate Error to Hidden Layer $\boldsymbol{\delta}^{(1)}$
$$\boldsymbol{\delta}^{(1)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(1)}} = \left( \frac{\partial z^{(2)}}{\partial \mathbf{h}} \cdot \frac{\partial \mathcal{L}}{\partial z^{(2)}} \right) \odot \text{ReLU}'\left(\mathbf{z}^{(1)}\right) = \left( \mathbf{w}^{(2)} \cdot \delta^{(2)} \right) \odot \mathbb{I}\left\{\mathbf{z}^{(1)} > \mathbf{0}\right\}$$

#### Step 4: Gradients for Hidden Layer Parameters ($W^{(1)}, \mathbf{b}^{(1)}$)
$$\frac{\partial \mathcal{L}}{\partial W^{(1)}} = \boldsymbol{\delta}^{(1)} \mathbf{x}^T, \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(1)}} = \boldsymbol{\delta}^{(1)}$$

---

### 3.4 Single-Path Derivation Trace (2022 Moed B)

Let us trace the scalar gradient calculation along a single 1D computational path where $\hat{y} = \sigma(z)$, $z = w h$, and $h = \max(0, u)$. We compute $\frac{\partial \mathcal{L}}{\partial u}$:

$$\frac{\partial \mathcal{L}}{\partial u} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial h} \cdot \frac{\partial h}{\partial u}$$

1. $\frac{\partial \hat{y}}{\partial z} = \sigma(z)(1 - \sigma(z)) = \hat{y}(1 - \hat{y})$
2. $\frac{\partial z}{\partial h} = w$
3. $\frac{\partial h}{\partial u} = \mathbb{I}\{u > 0\}$ (Derivative of ReLU)
4. Combine the terms together:
   $$\frac{\partial \mathcal{L}}{\partial u} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \Big[\hat{y}(1 - \hat{y})\Big] \cdot w \cdot \mathbb{I}\{u > 0\}$$
    

## 4. Convolutional Neural Networks (ConvNets / CNNs)

Fully connected networks (MLPs) scale terribly when processing spatial image data. A modest $1000 \times 1000$ pixel RGB image flattened into a vector yields $3,000,000$ input features, requiring millions of weights for a single neuron in the next layer, leading to catastrophic overfitting.

**ConvNets** resolve this by introducing a structurally restricted network architecture based on two core principles:
1. **Local Receptive Fields:** Neurons connect only to a small local neighborhood of features in the previous layer, capturing local patterns like edges and textures.
2. **Shared Weights (Parameter Sharing):** The exact same small matrix of weights—called a **kernel** or **filter**—is slid (convolved) across the entire input grid. This dramatically lowers the number of parameters and makes the network inherently **invariant to spatial translations (shifts)**.

---

### 4.1 Mathematical Formalization of 2D Convolution

For an input feature map $I$ of shape $(H_{\text{in}} \times W_{\text{in}} \times C_{\text{in}})$ and a 2D filter kernel $K$ of spatial size $(K_h \times K_w)$, the activation at output spatial location $(i, j)$ for the $k$-th output channel is:

$$S^{(k)}(i, j) = (I * K^{(k)})(i, j) + b^{(k)} = \sum_{c=1}^{C_{\text{in}}} \sum_{m=1}^{K_h} \sum_{n=1}^{K_w} I(i + m - 1, j + n - 1, c) \cdot K^{(k)}(m, n, c) + b^{(k)}$$

---

### 4.2 Spatial Dimensions & Parameter Counting Formulas (Exam Essential)

#### A. Output Spatial Dimension Formula
Given an input size $W_{\text{in}} \times H_{\text{in}}$, filter size $K_w \times K_h$, zero-padding $P$, and stride $S$:

$$W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} - K_w + 2P}{S} \right\rfloor + 1, \qquad H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} - K_h + 2P}{S} \right\rfloor + 1$$

*   **Zero-Padding Types:**
    *   *Valid Padding ($P=0$):* No padding added; spatial dimensions shrink after convolution.
    *   *Same Padding ($P = \frac{K-1}{2}$ for odd $K$, with $S=1$):* Output spatial dimensions match input dimensions exactly ($W_{\text{out}} = W_{\text{in}}$).

#### B. Total Parameter Calculation Rule
For a convolutional layer with $C_{\text{in}}$ input channels, $C_{\text{out}}$ output channels, and spatial filter size $K_h \times K_w$:

$$\text{Total Parameters} = \underbrace{\Big( K_h \times K_w \times C_{\text{in}} + 1 \Big)}_{\text{Weights + 1 Bias per Filter}} \times C_{\text{out}}$$

---

### 4.3 Pooling Layers (Spatial Downsampling)

Pooling layers reduce the spatial size of feature maps to decrease memory usage and parameter count while inducing local translation invariance:

*   **Max Pooling:** Selects the maximum value within a $K_p \times K_p$ sliding window. Preserves the most prominent feature activations (e.g. sharp edges).
    $$P_{\text{max}}(i, j) = \max_{m, n} I(i + m - 1, j + n - 1)$$
*   **Average Pooling:** Computes the arithmetic mean over the window. Smooths spatial features.
    $$P_{\text{avg}}(i, j) = \frac{1}{K_p \cdot K_p} \sum_{m=1}^{K_p} \sum_{n=1}^{K_p} I(i + m - 1, j + n - 1)$$

*Note:* Pooling layers contain **zero trainable parameters**; they only downsample spatial dimensions.
    

## 5. Python Implementation Snippets

### 5.1 Pure NumPy Implementation (Explicit Analytical Backprop)

This block demonstrates how to implement a 2-layer MLP with manual analytical backpropagation (vectorized Chain Rule gradient updates) using **only NumPy**:

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1.0 - s)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

class NeuralNetworkNumPy:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.1):
        self.lr = lr
        # Kaiming / He Initialization for ReLU hidden layer
        self.W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((hidden_dim, 1))
        # Xavier Initialization for Sigmoid output layer
        self.W2 = np.random.randn(output_dim, hidden_dim) * np.sqrt(1.0 / hidden_dim)
        self.b2 = np.zeros((output_dim, 1))

    def forward(self, x):
        """
        x: Input column vector of shape (input_dim, 1)
        """
        self.x = x
        # Layer 1 (Hidden ReLU)
        self.z1 = np.dot(self.W1, x) + self.b1   # (hidden_dim, 1)
        self.h1 = relu(self.z1)                  # (hidden_dim, 1)
        
        # Layer 2 (Output Sigmoid)
        self.z2 = np.dot(self.W2, self.h1) + self.b2 # (output_dim, 1)
        self.y_hat = sigmoid(self.z2)              # (output_dim, 1)
        return self.y_hat

    def backward(self, y_true):
        """
        y_true: Target column vector of shape (output_dim, 1)
        Computes analytical gradients using the Chain Rule & updates weights via SGD.
        """
        # Step 1: Output layer error delta2 = dL/dz2  (for L = 0.5 * (y_hat - y_true)^2)
        dL_dyhat = self.y_hat - y_true
        dyhat_dz2 = sigmoid_derivative(self.z2)
        delta2 = dL_dyhat * dyhat_dz2             # (output_dim, 1)
        
        # Step 2: Output layer parameter gradients
        dW2 = np.dot(delta2, self.h1.T)           # (output_dim, hidden_dim)
        db2 = delta2                              # (output_dim, 1)
        
        # Step 3: Backpropagate error to hidden layer delta1 = dL/dz1
        delta1 = np.dot(self.W2.T, delta2) * relu_derivative(self.z1) # (hidden_dim, 1)
        
        # Step 4: Hidden layer parameter gradients
        dW1 = np.dot(delta1, self.x.T)            # (hidden_dim, input_dim)
        db1 = delta1                              # (hidden_dim, 1)
        
        # SGD Parameter Update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

# Quick training test
nn = NeuralNetworkNumPy(input_dim=4, hidden_dim=8, output_dim=1, lr=0.1)
x_sample = np.array([[1.0], [-0.5], [2.0], [0.1]])
y_target = np.array([[1.0]])

print("Initial Prediction:", nn.forward(x_sample)[0, 0])
for step in range(100):
    nn.forward(x_sample)
    nn.backward(y_target)
print("Prediction after 100 Backprop Steps:", nn.forward(x_sample)[0, 0])
```

---

### 5.2 PyTorch Implementation Snippet (Autograd Framework)

This block shows how to define the same MLP architecture using PyTorch's automatic differentiation (`autograd`):

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultiLayerPerceptron, self).__init__()
        # Layer 1 parameters
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linear activation activation
        self.relu = nn.ReLU()
        # Layer 2 parameters
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        z1 = self.fc1(x)
        h1 = self.relu(z1)
        out = self.fc2(h1)
        return out

# Training simulation setup
model = MultiLayerPerceptron(input_dim=10, hidden_dim=32, output_dim=1)
criterion = nn.BCEWithLogitsLoss() # Includes internal stable sigmoid tracking
optimizer = optim.SGD(model.parameters(), lr=0.01)
```



---
## 🔗 Navigation
**Previous:** [[ ]] | **Next:** [[ ]]