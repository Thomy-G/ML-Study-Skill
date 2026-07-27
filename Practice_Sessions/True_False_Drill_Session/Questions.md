# 📝 Comprehensive True/False Exam Drill Session | אימון נכון/לא נכון מקיף למבחן

**Course / קורס:** Bar-Ilan University Machine Learning 89-511 / למידת מכונה, אוניברסיטת בר-אילן  
**Description / תיאור:** A targeted 25-question True/False drill spanning the entire BIU ML syllabus to prepare for Part A exam questions.

---

## 🔹 25 High-Yield True / False Questions
## 25 שאלות נכון / לא נכון למבחן

### Section 1: Optimization, Gradient Descent & Regularization
* **Q1.** Stochastic Gradient Descent (SGD) with mini-batch size $m=1$ is guaranteed to decrease the training loss monotonically at every single iteration step.  
  *אלגוריתם SGD עם $m=1$ מובטח להפחית את פונקציית ההפסד באופן מונוטוני בכל צעד וצעד.*
* **Q2.** Under standard SGD, Weight Decay $w \leftarrow (1 - \eta \lambda)w - \eta \nabla \mathcal{L}_0$ is mathematically identical to optimizing an $L_2$ regularized loss function $\mathcal{L}_0(w) + \frac{\lambda}{2}\|w\|^2$.  
  *ב-SGD רגיל, Weight Decay שקול אלגברית לחלוטין לאופטימיזציה של פונקציית הפסד עם רגולריזציית $L_2$.*
* **Q3.** Adding $L_1$ regularization ($\lambda \|w\|_1$) promotes sparse weight vectors (driving many parameters exactly to 0), whereas $L_2$ regularization shrinks weights toward 0 without making them exactly 0.  
  *רגולריזציית $L_1$ מעודדת וקטור משקולות דליל (Sparse), בעוד $L_2$ משקעת משקולות לאפס מבלי לאפס אותן מוחלטת.*
* **Q4.** Gradient descent is guaranteed to converge linearly to the unique global minimum when optimizing any strictly convex and $L$-smooth function with step size $\eta \le 1/L$.  
  *Gradient descent מובטח להתכנס בקצב לינארי למינימום הגלובלי היחיד עבור פונקציה קמורה חזקה וסמוטלית.*
* **Q5.** The ADAM optimizer acts as a strong regularizer that prevents overfitting in deep neural networks.  
  *אופטימייזר ADAM פועל כרגולריזטור חזק המונע יתר-התאמה (Overfitting) ברשתות עמוקות.*

---

### Section 2: Generalization, PAC Learning & VC-Dimension
* **Q6.** If a hypothesis class $H$ has $\text{VCdim}(H) = d < n$, then there cannot exist any dataset of size $n$ that can be classified with zero error by a hypothesis $h \in H$.  
  *אם ל-H יש $\text{VCdim}(H) = d < n$, לא ייתכן קיומו של אוסף נתונים בגודל $n$ שמסווג ללא שגיאות ע״י $h \in H$.*
* ** concept Q7.** A finite hypothesis class $|H| < \infty$ is PAC learnable in the realizable setting with sample complexity $m \ge \frac{1}{\epsilon} \ln \left( \frac{|H|}{\delta} \right)$.  
  *מחלקת היפותזות סופית $|H| < \infty$ היא PAC learnable בסביבה הניתנת למימוש עם סיבוכיות דגימה $m \ge \frac{1}{\epsilon} \ln \left( \frac{|H|}{\delta} \right)$.*
* **Q8.** Increasing the depth or width of a neural network always decreases the generalization gap between training error and test error.  
  *הגדלת עומק או רוחב של רשת ניורונים תמיד מקטינה את פער ההכללה (Generalization Gap).*
* **Q9.** The VC dimension of a linear hyperplane classifier in $d$-dimensional space $\mathbb{R}^d$ is $d+1$.  
  *ה-VC dimension של מפריד היפר-מישור לינארי במרחב $d$-ממדי הוא $d+1$.*
* **Q10.** According to the Bias-Variance tradeoff, increasing model capacity reduces variance but increases bias.  
  *לפי טרייד-אוף Bias-Variance, הגדלת קיבולת המודל מפחיתה שונות (Variance) אך מעלה הטיה (Bias).*

---

### Section 3: Linear Models, Logistic Regression & SVMs
* **Q11.** For linearly separable binary data, unregularized Logistic Regression has no finite minimum, driving the weight norm $\|w\| \to \infty$ during training.  
  *עבור נתונים המופרדים לינארית, לרגרסיה לוגיסטית ללא רגולריזציה אין מינימום סופי והמשקולות שואפות לאינסוף.*
* **Q12.** A $K$-Means clustering algorithm with $K=2$ produces a linear decision boundary between the two clusters.  
  *אלגוריתם $K$-Means עבור $K=2$ מייצר גבול החלטה לינארי בין שני האשכולות.*
* **Q13.** Dual Kernel SVM allows training non-linear classifiers in high or infinite-dimensional feature spaces $\Phi(x)$ using only pairwise kernel evaluations $K(x_i, x_j)$.  
  *בעיית הדואל ב-Kernel SVM מאפשרת לאמן מסווגים לא-לינאריים ע״י חישוב ערכי גרעין $K(x_i, x_j)$ בלבד.*
* **Q14.** The Perceptron algorithm is guaranteed to converge in a finite number of updates for any dataset, even if the data is not linearly separable.  
  *אלגוריתם Perceptron מובטח להתכנס במספר צעדים סופי עבור כל אוסף נתונים, גם אם אינו מופרד לינארית.*
* **Q15.** Soft-SVM uses slack variables $\xi_i \ge 0$ to penalize data points that violate the classification margin.  
  *Soft-SVM משתמש במשתני הרפיה $\xi_i \ge 0$ כדי להעניש נקודות המפרות את שולי הסיווג.*

---

### Section 4: Deep Learning, CNNs, RNNs & Transformers
* **Q16.** Batch Normalization (BatchNorm) enables using higher learning rates by stabilizing internal activation distributions and smoothing the loss surface.  
  *שכבת BatchNorm מאפשרת שימוש בקצבי למידה גבוהים יותר ע״י ייצוב התפלגות האקטיבציות והחלקת משטח ההפסד.*
* **Q17.** Dropout randomly sets a fraction $p$ of hidden layer activations to zero during inference (testing).  
  *טכניקת Dropout מאפסת באקראי חלק $p$ מהאקטיבציות בזמן הסקה (Inference/Testing).*
* **Q18.** Convolutional layers share kernel weights across spatial locations, significantly reducing parameter counts compared to fully connected layers.  
  *שכבות קונבולוציה חולקות משקולות במרחב, מה שמפחית דרמטית את מספר הפרמטרים בהשוואה לשכבות FC.*
* **Q19.** Scaled Dot-Product Attention divides dot products $QK^T$ by $\sqrt{d_k}$ to prevent Softmax gradients from vanishing when key dimension $d_k$ is large.  
  *מנגנון Scaled Dot-Product Attention מחלק ב-$\sqrt{d_k}$ כדי למנוע היעלמות נגזרות Softmax כאשר $d_k$ גדול.*
* **Q20.** Recurrent Neural Networks (RNNs) suffer from vanishing and exploding gradients when processing long sequences due to repeated matrix multiplication over time steps.  
  *רשתות RNN סובלות מהיעלמות והתפוצצות גרדיאנטים בסדרות ארוכות עקב מכפלות מטריצות חוזרות בזמן.*

---

### Section 5: Unsupervised Learning, Generative Models & Decision Trees
* **Q21.** In the limit as cluster variance $\sigma \to 0$, the Expectation-Maximization (EM) algorithm for a Gaussian Mixture Model (GMM) becomes mathematically identical to $K$-Means clustering.  
  *בגבול שבו שונות האשכולות $\sigma \to 0$, אלגוריתם EM עבור GMM הופך לשקול מתמטית ל-$K$-Means.*
* **Q22.** Principal Component Analysis (PCA) selects orthogonal projection axes that minimize total reconstruction error $\|x - \hat{x}\|^2$.  
  *אלגוריתם PCA בוחר צירים אורתוגונליים הממזערים את שגיאת השחזור הכוללת.*
* **Q23.** Gaussian Naïve Bayes with shared feature variances across classes always produces a linear decision boundary.  
  *מודל בייס נאיבי גאוסי עם שונות משותפת בין המחלקות מייצר תמיד גבול החלטה לינארי.*
* **Q24.** Information Gain in Decision Trees measures the reduction in Entropy after splitting a dataset on an attribute.  
  *מדד Information Gain בעצי החלטה מודד את ההפחתה באנטרופיה בעקבות פיצול הנתונים לפי תכונה.*
* **Q25.** Variational Autoencoders (VAEs) use the Reparameterization Trick ($z = \mu + \sigma \odot \epsilon$) to allow backpropagation gradients to flow through stochastic sampling nodes.  
  *מודלי VAE משתמשים ב-Reparameterization Trick כדי לאפשר זרימת גרדיאנטים ב-Backpropagation דרך דגימה אקראית.*
