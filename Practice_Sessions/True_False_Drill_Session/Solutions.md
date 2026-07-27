# 💡 Comprehensive True/False Drill Session — Official Solutions | פתרונות רשמיים

---

## 🔹 Section 1: Optimization, Gradient Descent & Regularization

* **Q1. Incorrect (False)**. Stochastic gradient descent (SGD) uses single-sample stochastic gradients $\nabla \mathcal{L}_i(w)$, which fluctuate noisy estimations of the true total gradient. Training loss may temporarily increase at individual steps, but converges on average.
* **Q2. Correct (True)**. Under standard SGD, weight decay $w_{t+1} = (1 - \eta \lambda)w_t - \eta \nabla \mathcal{L}_0(w_t)$ is algebraically identical to gradient descent on $\mathcal{L}_{\text{reg}}(w) = \mathcal{L}_0(w) + \frac{\lambda}{2}\|w\|^2$.
* **Q3. Correct (True)**. $L_1$ penalty $|\lambda w|$ has a constant gradient magnitude $\lambda \text{sign}(w)$ near zero, driving weights precisely to 0. $L_2$ penalty $\frac{\lambda}{2}w^2$ has gradient $\lambda w$ which vanishes near zero, shrinking weights without forcing exact zeros.
* **Q4. Correct (True)**. Gradient descent with learning rate $\eta \le 1/L$ on a strongly convex $L$-smooth function satisfies $\|w_t - w^*\|^2 \le (1 - \frac{\mu}{L})^t \|w_0 - w^*\|^2$, proving linear convergence to the unique global minimizer.
* **Q5. Incorrect (False)**. ADAM is an adaptive learning rate optimizer designed to accelerate optimization convergence. It does not act as a regularizer to prevent overfitting; unregularized Adam can overfit faster than SGD.

---

## 🔹 Section 2: Generalization, PAC Learning & VC-Dimension

* **Q6. Incorrect (False)**. $\text{VCdim}(H) = d < n$ means $H$ cannot shatter *every* arbitrary dataset of size $n$. However, zero-error target hypotheses $h \in H$ can easily exist for specific separable datasets $S_n$.
* **Q7. Correct (True)**. By the Finite Hypothesis Class PAC theorem, if $|H| < \infty$ and $m \ge \frac{1}{\epsilon} \ln \left( \frac{|H|}{\delta} \right)$, then with probability $\ge 1-\delta$, any empirical risk minimizer (ERM) $h_S$ achieves generalization error $R(h_S) \le \epsilon$.
* **Q8. Incorrect (False)**. Increasing neural network depth or capacity increases hypothesis class expressivity, which typically *expands* the generalization gap between training error and test error unless strong regularization (Dropout/L2) or massive data is applied.
* **Q9. Correct (True)**. A linear hyperplane classifier in $\mathbb{R}^d$ can shatter any set of $d+1$ points in general position, but cannot shatter $d+2$ points (Radon's Theorem), giving $\text{VCdim}(H_{\text{linear}}) = d+1$.
* **Q10. Incorrect (False)**. Increasing model capacity *reduces bias* (allows fitting complex target concepts) but *increases variance* (makes model sensitive to sample noise).

---

## 🔹 Section 3: Linear Models, Logistic Regression & SVMs

* **Q11. Correct (True)**. For linearly separable data, the NLL loss $\sum \ln(1 + e^{-y_i w^T x_i})$ approaches 0 only as $y_i w^T x_i \to \infty$. Standard gradient descent pushes $\|w\| \to \infty$, so no finite minimum exists without regularization.
* **Q12. Correct (True)**. For $K=2$, the cluster boundary is the perpendicular bisector plane $\|\mathbf{x} - \mu_1\|^2 = \|\mathbf{x} - \mu_2\|^2 \implies 2(\mu_2 - \mu_1)^T \mathbf{x} + \|\mu_1\|^2 - \|\mu_2\|^2 = 0$, defining a linear hyperplane.
* **Q13. Correct (True)**. By Kernel Trick (Representer Theorem), $w = \sum \alpha_i y_i \Phi(x_i)$. Predictions evaluate $\hat{y} = \text{sign}\left(\sum \alpha_i y_i K(x_i, x) + b\right)$ without computing high/infinite-dimensional feature maps $\Phi(x)$ directly.
* **Q14. Incorrect (False)**. Novikoff's theorem guarantees Perceptron convergence in at most $(R/\gamma)^2$ steps *only if the dataset is strictly linearly separable* with positive margin $\gamma > 0$. If data is non-separable, Perceptron loops indefinitely without converging.
* **Q15. Correct (True)**. Soft-SVM optimizes $\frac{1}{2}\|w\|^2 + C \sum \xi_i$ subject to $y_i(w^T x_i + b) \ge 1 - \xi_i$ and $\xi_i \ge 0$, where slack variables $\xi_i$ measure margin violations.

---

## 🔹 Section 4: Deep Learning, CNNs, RNNs & Transformers

* **Q16. Correct (True)**. BatchNorm normalizes internal activations, reducing internal covariate shift, smoothing loss landscape gradients, and permitting higher learning rates for faster convergence.
* **Q17. Incorrect (False)**. Dropout is active **only during training**. During inference/testing, all neurons remain active and weights are scaled by $(1-p)$ (or inverted dropout scales during training) to ensure deterministic predictions.
* **Q18. Correct (True)**. Convolutional layers apply localized $K_h \times K_w$ filter kernels across spatial locations (weight sharing), reducing parameters from $O(H \cdot W \cdot C_{\text{in}} \cdot C_{\text{out}})$ in FC layers to $O(K_h \cdot K_w \cdot C_{\text{in}} \cdot C_{\text{out}})$.
* **Q19. Correct (True)**. Unscaled dot products $QK^T$ have variance proportional to key dimension $d_k$. Large variances push Softmax into extreme saturation regions where gradients vanish ($\sigma' \to 0$). Dividing by $\sqrt{d_k}$ rescales variance to 1.
* **Q20. Correct (True)**. Backpropagation through time (BPTT) computes gradients $\prod_{k=t}^T W_{hh}$. Eigenvalues $\lambda > 1$ cause exponential gradient explosion, while $\lambda < 1$ causes exponential gradient vanishing.

---

## 🔹 Section 5: Unsupervised Learning, Generative Models & Decision Trees

* **Q21. Correct (True)**. For spherical equal variance GMM $\Sigma_k = \sigma^2 I$ and uniform priors $\pi_k = 1/K$, as $\sigma \to 0$, soft responsibilities $q(z_{i,k}=1)$ become hard binary indicators $\mathbb{I}\{k = \arg\min_l \|x_i - \mu_l\|^2\}$ (Step 1 of $K$-Means), and the M-step mean update becomes the arithmetic mean of assigned points (Step 2 of $K$-Means).
* **Q22. Correct (True)**. PCA finds an orthonormal basis $V \in \mathbb{R}^{d \times r}$ that maximizes projected sample variance, which is mathematically equivalent to minimizing the total mean squared reconstruction error $\|x - V V^T x\|^2$.
* **Q23. Correct (True)**. Setting shared variance $\Sigma_0 = \Sigma_1 = \text{diag}(\sigma_j^2)$ in Gaussian Naïve Bayes cancels quadratic feature terms $x_j^2$ in log-posterior ratio $\ln \frac{P(Y=1|x)}{P(Y=0|x)}$, yielding a linear decision boundary $\mathbf{w}^T \mathbf{x} + b = 0$.
* **Q24. Correct (True)**. Information Gain $\text{IG}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Entropy}(S_v)$ measures the expected reduction in entropy achieved by partitioning dataset $S$ on attribute $A$.
* **Q25. Correct (True)**. Sampling $z \sim \mathcal{N}(\mu, \Sigma)$ creates a non-differentiable stochastic sampling node. The Reparameterization Trick computes $z = \mu + \sigma \odot \epsilon$ with independent noise $\epsilon \sim \mathcal{N}(0, I)$, allowing backpropagation gradients $\nabla_\phi$ to flow directly through $\mu$ and $\sigma$.
