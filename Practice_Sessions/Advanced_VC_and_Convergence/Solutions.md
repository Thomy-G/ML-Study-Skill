# 💡 Advanced VC & Convergence — Official Solutions | פתרונות רשמיים

---

## 🔹 Part A: True/False Questions / חלק א'

### Question 1: Learning Theory & ERM
* **a. Correct (True)**. In the realizable case, there exists a target hypothesis $h^* \in H$ such that $h^*(x_i) = y_i$ for all samples in the dataset. Therefore, the empirical training error of the optimal hypothesis is strictly $\mathcal{L}_S(h^*) = 0$.
* **b. Incorrect (False)**. The ERM hypothesis is not necessarily unique. Multiple distinct hypotheses in $H$ can achieve the exact same minimum empirical loss on a given training set $S$.
* **c. Correct (True)**. Under 0-1 loss $L(h(x), y) = \mathbb{I}\{h(x) \neq y\}$, the expected risk is $R(h) = \mathbb{E}_{(x,y) \sim D}[\mathbb{I}\{h(x) \neq y\}] = P_{(x,y) \sim D}(h(x) \neq y)$, which is the exact probability of misclassifying a randomly drawn sample.

---

### Question 2: Deep Learning & Cross-Validation
* **a. Incorrect (False)**. Deep neural networks have massive expressivity (universal approximation theorem). Most practical failures in deep learning arise from generalization gaps (overfitting) or optimization difficulties (local minima, saddle points, vanishing gradients).
* **b. Correct (True)**. $K$-fold cross-validation averages performance over $K$ distinct validation splits, reducing estimation variance and providing a significantly more reliable estimate of generalization performance than a single split.
* **c. Correct (True)**. A ResNet residual block computes $\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$, adding the output of the convolutional layers $\mathcal{F}(\mathbf{x})$ directly to the identity mapping of the input $\mathbf{x}$.
* **d. Correct (True)**. Early ConvNet layers detect low-level visual primitives (edges, blobs, corners, textures), whereas deeper layers combine these primitives into high-level semantic object detectors (faces, wheels, animal parts).

---

## 🔹 Part B: Open Questions / חלק ב'

### Question 3: Convergence of Linear Regression

* **a. Optimization Task Formulation:**
  Find parameter vector $w^* \in \mathbb{R}^d$ minimizing Mean Squared Error (MSE) over dataset $\{(x_i, y_i)\}_{i=1}^n$:
  $$\min_{w \in \mathbb{R}^d} L(w) = \frac{1}{n} \|y - X w\|_2^2 = \frac{1}{n} \sum_{i=1}^n (y_i - w^T x_i)^2$$
  where $X \in \mathbb{R}^{n \times d}$ is the sample design matrix and $y \in \mathbb{R}^n$ is the target vector.

* **b. Proof of Unique Solution $w^*$ & Orthonormal Simplification:**
  Compute the gradient $\nabla L(w)$:
  $$\nabla L(w) = \frac{2}{n} X^T (X w - y)$$
  Equating the gradient to zero for first-order optimality condition $\nabla L(w^*) = 0$:
  $$\frac{2}{n} X^T (X w^* - y) = 0 \iff X^T X w^* = X^T y$$
  Since $X^T X \in \mathbb{R}^{d \times d}$ is invertible (full column rank $\text{rank}(X) = d$), multiply both sides by $(X^T X)^{-1}$:
  $$w^* = (X^T X)^{-1} X^T y \quad \blacksquare$$
  Since $L(w)$ is strictly convex when $X^T X$ is positive definite, $w^*$ is the **unique global minimizer**.  
  *Orthonormal Simplification*: If $X$ is orthonormal ($X^T X = I$), then:
  $$w^* = I^{-1} X^T y = X^T y$$

* **c. Recursive Gradient Descent Update Proof:**
  The Gradient Descent update rule with step size $\eta > 0$ is:
  $$w_t = w_{t-1} - \eta \nabla L(w_{t-1}) = w_{t-1} - \frac{2\eta}{n} X^T (X w_{t-1} - y) = w_{t-1} - \frac{2\eta}{n} X^T X w_{t-1} + \frac{2\eta}{n} X^T y$$
  From Part (b), recall the Normal Equations identity $X^T y = X^T X w^*$. Substitute this into the update rule:
  $$w_t = w_{t-1} - \frac{2\eta}{n} X^T X w_{t-1} + \frac{2\eta}{n} X^T X w^*$$
  Subtract $w^*$ from both sides:
  $$w_t - w^* = w_{t-1} - w^* - \frac{2\eta}{n} X^T X (w_{t-1} - w^*) = \left( I - \frac{2\eta}{n} X^T X \right) (w_{t-1} - w^*)$$
  Applying this linear contraction mapping recursively for $t$ steps yields:
  $$w_t - w^* = \left( I - \frac{2\eta}{n} X^T X \right)^t (w_0 - w^*) \quad \blacksquare$$

* **d. 1-Dimensional Convergence Condition Proof ($d=1$):**
  For scalar inputs $d=1$, $X \in \mathbb{R}^{n \times 1}$ is a column vector and $X^T X = \|X\|_2^2 = \sum_{i=1}^n x_i^2$.  
  The vector contraction equation simplifies to a scalar geometric sequence:
  $$w_t - w^* = \left( 1 - \frac{2\eta}{n} \|X\|_2^2 \right)^t (w_0 - w^*)$$
  For any initial parameter $w_0 \neq w^*$, the sequence $w_t \to w^*$ as $t \to \infty$ if and only if the geometric ratio has absolute value strictly less than 1:
  $$\left| 1 - \frac{2\eta}{n} \|X\|_2^2 \right| < 1$$
  Expanding the absolute value inequality:
  $$-1 < 1 - \frac{2\eta}{n} \|X\|_2^2 < 1$$
  * Left inequality: $1 - \frac{2\eta}{n} \|X\|_2^2 < 1 \implies \frac{2\eta}{n} \|X\|_2^2 > 0 \implies \eta > 0$.
  * Right inequality: $-1 < 1 - \frac{2\eta}{n} \|X\|_2^2 \implies \frac{2\eta}{n} \|X\|_2^2 < 2 \implies \eta < \frac{n}{\|X\|_2^2}$.

  Combining both inequalities yields the necessary and sufficient condition:
  $$0 < \eta < \frac{n}{\|X\|_2^2} \quad \blacksquare$$

---

### Question 4: VC-Dimension of Axis-Aligned Hyper-Rectangles

* **a. $H_{\text{int}}$ Shattering Proof for $|C|=2$:**
  Let $C = \{-1, +1\} \subset \mathbb{R}$. The hypothesis class $H_{\text{int}} = \{ h_{a,b}(x) = \mathbb{I}\{x \in [a, b]\} \}$.  
  We must realize all $2^2 = 4$ binary labelings on $C$:
  1. Labeling $(0, 0)$: Choose interval $[2, 3]$. Neither point is in interval $\implies (0, 0)$.
  2. Labeling $(1, 0)$: Choose interval $[-2, 0]$. Contains $-1$, excludes $+1 \implies (1, 0)$.
  3. Labeling $(0, 1)$: Choose interval $[0, 2]$. Excludes $-1$, contains $+1 \implies (0, 1)$.
  4. Labeling $(1, 1)$: Choose interval $[-2, 2]$. Contains both points $\implies (1, 1)$.  
  Thus, $C = \{-1, +1\}$ is shattered by $H_{\text{int}}$, proving $\text{VCdim}(H_{\text{int}}) \ge 2$.

* **b. $H_{\text{int}}$ Non-Shattering Proof for $|C|=3 \implies \text{VCdim}(H_{\text{int}}) = 2$:**
  Consider any 3 ordered real points $c_1 < c_2 < c_3$.  
  Consider the labeling $(1, 0, 1)$ where $c_1$ and $c_3$ are labeled $+1$ (inside interval) and $c_2$ is labeled $0$ (outside interval).  
  If $h_{a,b}$ labels $c_1$ and $c_3$ as $+1$, then by definition of interval $a \le c_1 \le b$ and $a \le c_3 \le b$.  
  Since $c_1 < c_2 < c_3$, it follows by transitivity that $a \le c_1 < c_2 < c_3 \le b \implies a \le c_2 \le b$.  
  This forces $c_2$ to be inside $[a, b]$, so $h_{a,b}(c_2) = 1$, making labeling $(1, 0, 1)$ impossible to achieve!  
  Since no 3-point set can be shattered, $\text{VCdim}(H_{\text{int}}) = 2$. $\quad \blacksquare$

* **c. Hyper-Rectangle $H_{\text{rec}}^d$ Shattering Proof for $|C|=2d$:**
  In $\mathbb{R}^d$, let $e_i$ be the standard basis vector along dimension $i$. Define $C = \{ e_i \}_{i=1}^d \cup \{ -e_i \}_{i=1}^d$, containing $2d$ points.  
  For any given target labeling $y = (y_1^+, \dots, y_d^+, y_1^-, \dots, y_d^-)$, construct boundary intervals $[a_i, b_i]$ for each coordinate $i=1,\dots,d$:
  * If $e_i$ is labeled $+1$, set upper bound $b_i = 2$; if $0$, set $b_i = 0.5$.
  * If $-e_i$ is labeled $+1$, set lower bound $a_i = -2$; if $0$, set $a_i = -0.5$.
  The resulting hyper-rectangle $\prod_{i=1}^d [a_i, b_i]$ contains point $e_i$ if and only if $y_i^+ = 1$, and contains $-e_i$ if and only if $y_i^- = 1$.  
  Thus, $C$ of size $2d$ is shattered, proving $\text{VCdim}(H_{\text{rec}}^d) \ge 2d$.

* **d. Hyper-Rectangle $H_{\text{rec}}^d$ Non-Shattering Proof for $|C|=2d+1 \implies \text{VCdim}(H_{\text{rec}}^d) = 2d$:**
  Consider any set $C$ of $2d+1$ points in $\mathbb{R}^d$.  
  Construct the minimal bounding box around $C$ by taking the extreme coordinate bounds along each dimension $i=1,\dots,d$:
  $$x_i^{\min} = \min_{x \in C} x_i, \quad x_i^{\max} = \max_{x \in C} x_i$$
  These extreme bounds require at most $2$ points per dimension, so at most $2d$ points define the boundary of the minimal bounding box.  
  Since $|C| = 2d + 1$, by the Pigeonhole Principle there exists at least one internal point $x_{\text{int}} \in C$ that is not an extreme boundary point along any dimension:
  $$x_{i,\text{int}} \in [x_i^{\min}, x_i^{\max}] \quad \forall i=1,\dots,d$$
  Assign label $+1$ to all bounding points and label $0$ to internal point $x_{\text{int}}$.  
  Any hyper-rectangle $H$ that encloses all $2d$ bounding points must contain $[x_i^{\min}, x_i^{\max}]$ for every dimension $i$, which forces it to also enclose the internal point $x_{\text{int}}$.  
  Thus, labeling all boundary points $+1$ and internal point $0$ cannot be realized by any axis-aligned hyper-rectangle.  
  Therefore, no set of $2d+1$ points can be shattered $\implies \text{VCdim}(H_{\text{rec}}^d) = 2d$. $\quad \blacksquare$

---

### Question 5: Backpropagation & Imbalanced Loss

* **a. Dimensions of Bias Vectors $b_1$ and $b_2$:**
  * Since $z_1^{(i)} = W_1 x^{(i)} + b_1$ produces hidden activation vector $a_1^{(i)} \in \mathbb{R}^{D_{a_1}}$, bias vector $b_1$ must match hidden dimension: $b_1 \in \mathbb{R}^{D_{a_1}}$.
  * Since $z_2^{(i)} = W_2 a_1^{(i)} + b_2$ produces scalar logit $z_2^{(i)} \in \mathbb{R}$, bias scalar $b_2 \in \mathbb{R}^1$.

* **b. Class Weight Calculations ($\alpha, \beta$):**
  Dataset has $n_1 = 200$ positive images ($y=1$) and $n_0 = 2000$ negative images ($y=0$).  
  In the dataset, class frequencies are $P_{\text{data}}(y=1) = \frac{200}{2200} = \frac{1}{11}$ and $P_{\text{data}}(y=0) = \frac{2000}{2200} = \frac{10}{11}$.  
  To re-weight the empirical loss to reflect a 50-50 real-world prior ($P_{\text{target}}(y=1) = 0.5, P_{\text{target}}(y=0) = 0.5$), set weights inversely proportional to empirical frequency:
  $$\alpha \propto \frac{P_{\text{target}}(y=1)}{P_{\text{data}}(y=1)} = \frac{0.5}{200/2200} = 5.5$$
  $$\beta \propto \frac{P_{\text{target}}(y=0)}{P_{\text{data}}(y=0)} = \frac{0.5}{2000/2200} = 0.55$$
  Setting $\alpha = 5.5$ and $\beta = 0.55$ (or ratio $\alpha / \beta = 10$) balances the expected loss contribution equally between giraffe and non-giraffe classes.

* **c. Backpropagation Gradient Update Rules:**
  Let loss for sample $i$ be $L^{(i)} = \alpha y^{(i)} \ln(\hat{y}^{(i)}) + \beta (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)})$.  
  Using chain rule for scalar prediction output $\hat{y}^{(i)} = \sigma(z_2^{(i)})$:
  $$\frac{\partial L^{(i)}}{\partial z_2^{(i)}} = \frac{\partial L^{(i)}}{\partial \hat{y}^{(i)}} \cdot \sigma'(z_2^{(i)}) = \left( \frac{\alpha y^{(i)}}{\hat{y}^{(i)}} - \frac{\beta (1 - y^{(i)})}{1 - \hat{y}^{(i)}} \right) \cdot \hat{y}^{(i)} (1 - \hat{y}^{(i)}) = \alpha y^{(i)} (1 - \hat{y}^{(i)}) - \beta (1 - y^{(i)}) \hat{y}^{(i)} \equiv \delta_2^{(i)}$$

  Backpropagating error to hidden layer 1:
  $$\delta_1^{(i)} = \frac{\partial L^{(i)}}{\partial z_1^{(i)}} = \delta_2^{(i)} \cdot W_2^T \odot \text{ReLU}'(z_1^{(i)})$$

  **Gradient Descent Parameter Update Rules** (with learning rate $\eta > 0$ and batch size $m$):
  $$W_2 \leftarrow W_2 + \frac{\eta}{m} \sum_{i=1}^m \delta_2^{(i)} (a_1^{(i)})^T$$
  $$b_2 \leftarrow b_2 + \frac{\eta}{m} \sum_{i=1}^m \delta_2^{(i)}$$
  $$W_1 \leftarrow W_1 + \frac{\eta}{m} \sum_{i=1}^m \delta_1^{(i)} (x^{(i)})^T$$
  $$b_1 \leftarrow b_1 + \frac{\eta}{m} \sum_{i=1}^m \delta_1^{(i)}$$

* **d. $L_2$ Regularized Update Rule & Impact on Part (b):**
  With $L_2$ regularization $\frac{\lambda}{2} \|W_1\|_F^2$, the total objective adds penalty $\lambda \|W_1\|_F^2$.  
  Differentiating yields additional gradient $2 \lambda W_1$.  
  The new update rule for $W_1$ is:
  $$W_1 \leftarrow W_1 (1 - 2\eta\lambda) + \frac{\eta}{m} \sum_{i=1}^m \delta_1^{(i)} (x^{(i)})^T$$
  * **Impact on Part (b)**: **No, the answer to Part (b) should NOT change**.  
    Class weights $\alpha$ and $\beta$ compensate for empirical training data class imbalance relative to test population distribution ($200$ vs $2000$). Adding $L_2$ weight regularization penalizes parameter magnitude to reduce overfitting, but does not alter the underlying dataset class distribution. Thus, optimal class balancing weights $\alpha = 5.5, \beta = 0.55$ remain identical.
