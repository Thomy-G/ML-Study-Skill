# 📝 Rare & Advanced Topics Practice Session | אימון בנושאים متקדמים ונדירים במבחנים

**Course / קורס:** Bar-Ilan University Machine Learning 89-511 / למידת מכונה, אוניברסיטת בר-אילן  
**Topics Covered / נושאים:** Transformers & Multi-Head Attention, Self-Supervised Learning & VAEs, Diffusion Models, Algorithmic Fairness & Bias, Word Embeddings (Word2Vec/Skip-gram), PAC Learning for Infinite Spaces & VC-Dimension.

---

## 🔹 Part A: True/False Questions (30 Points Total)
## חלק א': שאלות נכון / לא נכון (30 נקודות)

### 📌 Question 1 (15 Points) — Transformers & Sequence Architectures
* **a.** Self-attention has a computational complexity of $O(T^2 \cdot d)$ per layer for sequence length $T$ and embedding dimension $d$, whereas standard RNNs require $O(T \cdot d^2)$.  
  *מנגנון Self-Attention דורש סיבוכיות חישוב של $O(T^2 \cdot d)$ לכל שכבה עבור אורך סדרה $T$, בעוד RNN רגיל דורש $O(T \cdot d^2)$.*
* **b.** Positional Encodings are necessary in Transformers because the Scaled Dot-Product Attention operation is permutation invariant with respect to sequence tokens.  
  *קידוד מיקומים (Positional Encoding) נחוץ ב-Transformers מכיוון שפעולת Self-Attention היא אינווריאנטית לתמורות של הטוקנים בסדרה.*
* **c.** Multi-Head Attention strictly increases the number of parameters compared to Single-Head Attention with the same total projection dimension.  
  *מנגנון Multi-Head Attention מגדל בהכרח את מספר הפרמטרים ברשת בהשוואה ל-Single-Head Attention עם ממד היטל כולל זהה.*
* **d.** Causal (Masked) Self-Attention in GPT-style decoders sets future attention logits to $-\infty$ before applying Softmax to prevent data leakage from future tokens.  
  *ב-Masked Self-Attention, לוגיטים של טוקנים עתידיים מוגדרים ל-$-\infty$ לפני מנגנון ה-Softmax כדי למנוע דליפת מידע מתקדמת.*
* **e.** Transformer residual connections and Layer Normalization (LayerNorm) help prevent vanishing/exploding gradients in 100+ layer architectures.  
  *חיבורי דילוג (Residual Connections) ו-LayerNorm ב-Transformers מונעים היעלמות והתפוצצות גרדיאנטים ברשתות עמוקות מאוד.*

---

### 📌 Question 2 (15 Points) — Generative Models, Embeddings & Fairness
* **a.** In Word2Vec (Skip-gram model with Negative Sampling), word vectors are learned by maximizing the probability of predicting context words given a center word.  
  *ב-Word2Vec (Skip-gram), וקטורי מילים נלמדים ע״י מקסום ההסתברות לחזות מילות הקשר מתוך מילת מרכז.*
* **b.** Variational Autoencoders (VAEs) maximize the Evidence Lower Bound (ELBO), which consists of a reconstruction loss term and a KL-divergence regularization term $D_{KL}(q(z|x) \| p(z))$.  
  *רשתות VAE ממקסמות את גבול ה-ELBO המורכב מהפסד שחזור ורגולריזציית KL-Divergence.*
* **c.** Denoising Diffusion Probabilistic Models (DDPMs) generate images in a single forward pass through a neural network, making inference faster than GANs.  
  *מודלי דיפוזיה (DDPM) מייצרים תמונות במעבר יחיד קדימה (Single Forward Pass) ולכן הם מהירים יותר מ-GANs.*
* **d.** Demographic Parity fairness metric requires that the positive prediction rate $P(\hat{Y}=1 \mid A=a)$ is equal across all protected sensitive attribute groups $A$.  
  *מדד ההוגנות Demographic Parity דורש משיעור החיזוי החיובי להיות שווה בין קבוצות שונות של התכונה המוגנת $A$.*
* **e.** Hard Margin SVM has an infinite VC dimension because it can separate any dataset with zero training error.  
  *ל-Hard Margin SVM יש VC dimension אינסופי מכיוון שהוא יכול להפריד כל אוסף נתונים מופרד לינארית ללא שגיאת אימון.*

---

## 🔹 Part B: Open Questions (Choose 2 out of 3, 35 Points Each)
## חלק ב': שאלות פתוחות (בחר/י 2 מתוך 3, 35 נקודות לשאלה)

---

### 📌 Question 3: Transformers & Scaled Dot-Product Attention Mechanics
### שאלה 3: מנגנון Attention ומבנה ה-Transformer

Consider an input sequence matrix $X \in \mathbb{R}^{T \times d_{\text{in}}}$ containing $T$ tokens.
* **a.** Define the Query, Key, and Value projection matrices $W_Q, W_K \in \mathbb{R}^{d_{\text{in}} \times d_k}$ and $W_V \in \mathbb{R}^{d_{\text{in}} \times d_v}$. Write the mathematical formula for Query ($Q$), Key ($K$), and Value ($V$) matrices.  
  *הגדר/י את מטריצות ההיטל $W_Q, W_K, W_V$ וכתוב/כתבי את הנוסחאות עבור $Q, K, V$.*
* **b.** Write the exact equation for Scaled Dot-Product Attention:
  $$\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$
  Explain mathematically **WHY** we scale the dot products $Q K^T$ by dividing by $\sqrt{d_k}$. What happens to the gradients of Softmax if $d_k$ is very large and no scaling is applied? Prove mathematically using variance of independent random variables $q_i, k_i \sim \mathcal{N}(0, 1)$.  
  *הסבר/י מתמטית מדוע מנגנון ה-Attention מחלק ב-$\sqrt{d_k}$. מה קורה לנגזרות ה-Softmax כאשר $d_k$ גדל ללא חלוקה זו? הוכח/י בעזרת שונות של משתנים אקראיים.*
* **c.** Explain the Multi-Head Attention architecture. Show how outputs from $h$ independent attention heads $H_1, \dots, H_h$ are combined into a final output matrix. What is the main advantage of Multi-Head Attention over Single-Head Attention?  
  *תאר/י את ארכיטקטורת Multi-Head Attention. כיצד משלבים פלטים מ-$h$ ראשים שונים ומהו היתרון המרכזי של multi-head?*

---

### 📌 Question 4: Word Embeddings & Word2Vec Skip-Gram Derivation
### שאלה 4: ייצוגי מילים וגזירת אלגוריתם Word2Vec Skip-Gram

Given a vocabulary $V$ of size $|V|$, each word $w \in V$ is represented by a center word vector $\mathbf{v}_w \in \mathbb{R}^d$ and a context word vector $\mathbf{v}'_w \in \mathbb{R}^d$.
* **a.** Write the Softmax probability equation for observing context word $w_O$ given center word $w_C$:
  $$P(w_O \mid w_C) = \frac{\exp\left( (\mathbf{v}'_{w_O})^T \mathbf{v}_{w_C} \right)}{\sum_{w \in V} \exp\left( (\mathbf{v}'_w)^T \mathbf{v}_{w_C} \right)}$$
  Explain why computing this exact Softmax probability is computationally expensive for large vocabularies ($|V| = 10^6$).  
  *רשום/רשמי את נוסחת ה-Softmax לניבוי מילת הקשר והסבר/י מדוע חיושבה יקר עבור אוצר מילים גדול.*
* **b.** Define the **Negative Sampling** objective function for a single center word $w_C$ and positive context word $w_O$, using $K$ noise words $w_1, \dots, w_K \sim P_N(w)$:
  $$\mathcal{L}_{\text{NEG}} = -\ln \sigma\left( (\mathbf{v}'_{w_O})^T \mathbf{v}_{w_C} \right) - \sum_{k=1}^K \ln \sigma\left( -(\mathbf{v}'_{w_k})^T \mathbf{v}_{w_C} \right)$$
  Derive the gradient of $\mathcal{L}_{\text{NEG}}$ with respect to the center word vector $\mathbf{v}_{w_C}$.  
  *הגדר/י את פונקציית ההפסד של Negative Sampling וגזור/גזרי את הנגזרת לפי וקטור מילת המרכז $\mathbf{v}_{w_C}$.*
* **c.** Discuss how Word2Vec embeddings capture semantic linear relationships (e.g. $\mathbf{v}_{\text{King}} - \mathbf{v}_{\text{Man}} + \mathbf{v}_{\text{Woman}} \approx \mathbf{v}_{\text{Queen}}$).  
  *הסבר/י כיצד וקטורי Word2Vec משמרים יחסים סמנטיים לינאריים.*

---

### 📌 Question 5: Self-Supervised Learning, VAEs & Diffusion Models
### שאלה 5: למידה בעיבוד עצמי, VAEs ומודלי דיפוזיה

* **a.** Consider a Variational Autoencoder (VAE) with encoder $q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \Sigma_\phi(x))$ and decoder $p_\theta(x \mid z)$.  
  * **(i)** Write the Evidence Lower Bound (ELBO) objective function $\mathcal{L}_{\text{ELBO}}(\phi, \theta; x)$.  
  * **(ii)** Explain the **Reparameterization Trick**: $z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$. Why is this trick necessary for backpropagation?  
  *פתח/י את ה-ELBO והסבר/י את ה-Reparameterization Trick ומדוע הוא הכרחי ל-Backprop.*
* **b.** In Denoising Diffusion Probabilistic Models (DDPM):  
  * **(i)** Describe the **Forward (Noising) Process** $q(x_t \mid x_0)$ that adds Gaussian noise over timesteps $t=1,\dots,T$.  
  * **(ii)** Describe the **Reverse (Denoising) Process** $p_\theta(x_{t-1} \mid x_t)$ trained using a neural network $\epsilon_\theta(x_t, t)$ to predict added noise.  
  *תאר/י את תהליך הדיפוזיה הקדמי (הוספת רעש) והאחורי (ניקוי רעש) ב-DDPM.*
