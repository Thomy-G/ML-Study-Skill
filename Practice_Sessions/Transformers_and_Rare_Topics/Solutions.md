# 💡 Transformers & Rare Topics — Official Solutions | פתרונות רשמיים

---

## 🔹 Part A: True/False Questions / חלק א'

### Question 1: Transformers & Sequence Architectures
* **a. Correct (True)**. Self-attention compares every token to every other token ($T \times T$ matrix multiplication), taking $O(T^2 \cdot d)$ operations per layer. RNNs iterate sequentially step-by-step ($T$ steps), doing $d \times d$ matrix multiplications per step ($O(T \cdot d^2)$).
* **b. Correct (True)**. Scaled Dot-Product Attention is permutation-equivariant/invariant because matrix multiplication $Q K^T$ does not encode positional indices. Adding Positional Encodings introduces spatial order information.
* **c. Incorrect (False)**. Multi-Head Attention splits the embedding dimension $d$ into $h$ heads of size $d_k = d/h$. The projection matrices $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$ have the exact same total number of parameters ($3d^2$) as Single-Head Attention.
* **d. Correct (True)**. Setting future token logit positions to $-\infty$ forces $\text{Softmax}(-\infty) = 0$, preventing the decoder from attending to future tokens during auto-regressive generation.
* **e. Correct (True)**. Residual connections $x + F(x)$ provide direct gradient highways during backpropagation, and LayerNorm stabilizes activation variances, enabling deep 100+ layer Transformer models to converge smoothly.

### Question 2: Generative Models, Embeddings & Fairness
* **a. Correct (True)**. The Skip-gram model maximizes $\sum \ln P(w_{t+j} \mid w_t)$ for context window $j \in [-c, c] \setminus \{0\}$, forcing words appearing in similar contexts to have similar vector embeddings.
* **b. Correct (True)**. VAEs optimize $\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q(z|x)}[\ln p(x|z)] - D_{KL}(q(z|x) \| p(z))$, balancing reconstruction accuracy against prior latent distribution Gaussianity.
* **c. Incorrect (False)**. DDPMs require iterating through $T \approx 1000$ sequential denoising steps during inference to generate a single image, making sampling much slower than single-pass GAN generators.
* **d. Correct (True)**. Demographic Parity specifies $P(\hat{Y}=1 \mid A=0) = P(\hat{Y}=1 \mid A=1)$, demanding equal selection rates across demographic groups $A$ regardless of ground truth $Y$.
* **e. Incorrect (False)**. Hard Margin SVM in $d$-dimensional space has a VC dimension of $\text{VCdim} = d + 1$. Its capacity is constrained to linear hyperplanes, so it cannot shatter arbitrary sets of $d+2$ points.

---

## 🔹 Part B: Open Questions / חלק ב'

### Question 3: Transformers & Scaled Dot-Product Attention Mechanics
* **a. Projection Equations & Matrix Dimensions:**
  Given sequence matrix $X \in \mathbb{R}^{T \times d_{\text{in}}}$:
  $$Q = X W_Q \in \mathbb{R}^{T \times d_k}, \quad K = X W_K \in \mathbb{R}^{T \times d_k}, \quad V = X W_V \in \mathbb{R}^{T \times d_v}$$
  where $W_Q, W_K \in \mathbb{R}^{d_{\text{in}} \times d_k}$ and $W_V \in \mathbb{R}^{d_{\text{in}} \times d_v}$.

* **b. Attention Scaling Division by $\sqrt{d_k}$ Proof:**
  * **Proof of Dot Product Variance**:
    Let query vector $\mathbf{q} = [q_1, \dots, q_{d_k}]^T$ and key vector $\mathbf{k} = [k_1, \dots, k_{d_k}]^T$ have independent components with mean 0 and variance 1 ($\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0, \text{Var}(q_i) = \text{Var}(k_i) = 1$).  
    The dot product is $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$.  
    Expected value: $\mathbb{E}[q \cdot k] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = \sum 0 = 0$.  
    Variance:
    $$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k} \left[ \mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i k_i])^2 \right] = \sum_{i=1}^{d_k} \mathbb{E}[q_i^2] \mathbb{E}[k_i^2] = \sum_{i=1}^{d_k} (1 \cdot 1) = d_k$$
  * **Why We Divide by $\sqrt{d_k}$**:  
    Without scaling, for large key dimension $d_k$ (e.g. $d_k = 64$), the variance of dot products grows to $d_k = 64$ (standard deviation $\sigma = 8$). Large dot product inputs push the Softmax function into extreme saturation regions where outputs are close to 1 or 0, causing the Softmax gradients $\sigma'(z) = \sigma(z)(1-\sigma(z))$ to vanish ($\to 0$) and halting gradient descent training.  
    Dividing by $\sqrt{d_k}$ rescales the variance back to $\text{Var}\left( \frac{q \cdot k}{\sqrt{d_k}} \right) = \frac{\text{Var}(q \cdot k)}{d_k} = \frac{d_k}{d_k} = 1$, preserving stable gradient flow!

* **c. Multi-Head Attention Architecture:**
  Instead of computing a single attention function with $d$-dimensional keys, Multi-Head Attention projects Queries, Keys, and Values $h$ times with different learned linear projections to $d_k = d/h$ dimensions:
  $$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V) \in \mathbb{R}^{T \times d_v}$$
  The outputs of all $h$ heads are concatenated along the feature dimension and projected again:
  $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O \in \mathbb{R}^{T \times d_{\text{model}}}$$
  * **Advantage**: Allows the model to jointly attend to information from different representation subspaces at different positions simultaneously (e.g., one head attends to syntactic subject-verb agreements, while another attends to coreference nouns).

---

### Question 4: Word Embeddings & Word2Vec Skip-Gram Derivation
* **a. Softmax Bottleneck:**
  $$P(w_O \mid w_C) = \frac{\exp\left( (\mathbf{v}'_{w_O})^T \mathbf{v}_{w_C} \right)}{\sum_{w \in V} \exp\left( (\mathbf{v}'_w)^T \mathbf{v}_{w_C} \right)}$$
  * **Computational Cost**: The denominator requires summing dot products $\sum_{w \in V} \exp((\mathbf{v}'_w)^T \mathbf{v}_{w_C})$ over all words $w$ in vocabulary $V$. For $|V| = 10^6$ and vector dimension $d = 300$, computing this denominator for every target word requires $3 \times 10^8$ operations per training step, making exact Softmax intractable.

* **b. Negative Sampling Gradient Derivation:**
  $$\mathcal{L}_{\text{NEG}} = -\ln \sigma\left( (\mathbf{v}'_{w_O})^T \mathbf{v}_{w_C} \right) - \sum_{k=1}^K \ln \sigma\left( -(\mathbf{v}'_{w_k})^T \mathbf{v}_{w_C} \right)$$
  Using the derivative rule $\frac{d}{dz} \ln \sigma(z) = 1 - \sigma(z)$ and $\frac{d}{dz} \ln \sigma(-z) = -(1 - \sigma(-z)) = -(1 - (1 - \sigma(z))) = -\sigma(z)$:

  Differentiating $\mathcal{L}_{\text{NEG}}$ w.r.t. center word vector $\mathbf{v}_{w_C}$:
  $$\nabla_{\mathbf{v}_{w_C}} \mathcal{L}_{\text{NEG}} = -\left( 1 - \sigma\left((\mathbf{v}'_{w_O})^T \mathbf{v}_{w_C}\right) \right) \mathbf{v}'_{w_O} + \sum_{k=1}^K \sigma\left( (\mathbf{v}'_{w_k})^T \mathbf{v}_{w_C} \right) \mathbf{w}'_{w_k}$$
  * **Interpretation**: The gradient pulls the center vector $\mathbf{v}_{w_C}$ toward true context word $\mathbf{v}'_{w_O}$ (scaled by prediction error $1 - \sigma$) and pushes it away from noise word vectors $\mathbf{v}'_{w_k}$ (scaled by noise prediction score $\sigma$).

* **c. Linear Semantic Vector Analogies:**
  Skip-gram objective enforces dot products $(\mathbf{v}'_w)^T \mathbf{v}_c$ to reflect co-occurrence log-odds. Because relations like (King $\to$ Queen) and (Man $\to$ Woman) represent identical relational translations in log-odds space, vector differences $\mathbf{v}_{\text{King}} - \mathbf{v}_{\text{Man}}$ encode an abstract gender direction vector $\mathbf{v}_{\text{gender}}$. Adding this direction vector to $\mathbf{v}_{\text{Woman}}$ yields a point whose nearest neighbor in embedding space is $\mathbf{v}_{\text{Queen}}$.

---

### Question 5: Self-Supervised Learning, VAEs & Diffusion Models
* **a. (i) VAE Evidence Lower Bound (ELBO):**
  $$\mathcal{L}_{\text{ELBO}}(\phi, \theta; x) = \mathbb{E}_{q_\phi(z \mid x)} \left[ \ln p_\theta(x \mid z) \right] - D_{KL}\left( q_\phi(z \mid x) \,\|\, p(z) \right)$$
  * **Term 1 (Reconstruction Loss)**: Encourages decoder $p_\theta(x|z)$ to accurately reconstruct input $x$ from latent code $z$.
  * **Term 2 (KL Regularization)**: Forces encoder output distribution $q_\phi(z|x)$ to remain close to standard Gaussian prior $p(z) = \mathcal{N}(0, I)$, preventing latent space gaps.

* **a. (ii) Reparameterization Trick:**
  Instead of sampling $z \sim q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \Sigma_\phi(x))$ directly (which is a non-differentiable stochastic sampling node that blocks backpropagation gradient flow), we isolate the randomness into an independent noise vector $\epsilon \sim \mathcal{N}(0, I)$ and compute $z$ deterministically:
  $$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$$
  This allows gradients $\nabla_\phi \mathcal{L}$ to flow directly through mean $\mu_\phi(x)$ and standard deviation $\sigma_\phi(x)$ via standard backpropagation!

* **b. (i)-(ii) Diffusion Forward & Reverse Processes:**
  * **Forward Noising Process $q(x_t \mid x_{t-1})$**: A fixed Markov chain that gradually adds small Gaussian noise to data image $x_0$ across timesteps $t=1,\dots,T$:
    $$x_t = \sqrt{1 - \beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon \implies x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon \quad \text{where } \epsilon \sim \mathcal{N}(0, I)$$
    As $t \to T$, $x_T$ becomes pure isotropic Gaussian noise $\mathcal{N}(0, I)$.
  * **Reverse Denoising Process $p_\theta(x_{t-1} \mid x_t)$**: A learned neural network $\epsilon_\theta(x_t, t)$ (typically a U-Net) trained to predict the noise $\epsilon$ added at step $t$, optimizing simple MSE loss:
    $$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]$$
