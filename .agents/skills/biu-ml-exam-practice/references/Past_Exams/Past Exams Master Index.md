---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Resource
date_added: 2026-07-26
---
# Machine Learning — Past Exams & Official Solutions Index (2022–2025)

**MOC:** [[Machine learning MOC]]  
**Course:** [[English]] (Bar-Ilan University, Prof. Gal Chechik & Prof. Joseph Keshet)

---

## 📚 Overview & Archive Inventory

This resource hub compiles official past exam forms, student solution rubrics, and detailed step-by-step mathematical answers extracted directly from the course repository.

| Year | Term / Moed | File Link | Topics & Question Types Covered |
| :--- | :--- | :--- | :--- |
| **2022** | Moed A & B | [[2022 Machine Learning Past Exams (Moed A & B)]] | $k$-NN Metrics, CNN Architecture & Output Shapes, Margin Perceptron vs. Soft-SVM, Computational Graphs & Softmax NLL Backpropagation. |
| **2023** | Moed A & B | [[2023 Machine Learning Past Exams (Moed A & B)]] | Parameter Estimation (MLE/MAP), Linear Regression & Ridge Duality, Softmax Gradient Derivations, Decision Trees & Information Gain. |
| **2024** | Moed A, B & C | [[2024 Machine Learning Past Exams (Moed A, B & C)]] | Support Vector Machines (Hard/Soft Margin, Dual Formulation), Kernel Proofs, Autoencoders & PCA Equivalence, ResNet Skip Connections. |
| **2025** | Moed A & B | [[2025 Machine Learning Past Exams (Moed A & B)]] | Contrastive Self-Supervised Learning (SimCLR), Masked Autoencoders (MAE), Diffusion Models & Guidance, Algorithmic Fairness & $do$-calculus. |

---

## 🎯 Exam Strategy & High-Yield Exam Patterns

1. **Calculus & Matrix Backpropagation**: Always write down local derivatives using the Chain Rule ($\boldsymbol{\delta}^{(l)} = (W^{(l+1)})^T \boldsymbol{\delta}^{(l+1)} \odot g'(\mathbf{z}^{(l)})$).
2. **CNN Parameter Calculation**:
   $$\text{Params} = (K_h \times K_w \times C_{\text{in}} + 1) \times C_{\text{out}}$$
   $$\text{Output Spatial Dim} = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$
3. **Kernel Validity Proofs**: Expand $K(\mathbf{x}, \mathbf{z})$ algebraically into an explicit dot product $\phi(\mathbf{x})^T \phi(\mathbf{z})$ or prove the Gram matrix is Positive Semi-Definite ($G \succeq 0$).
4. **Softmax Invariance & Shift**: $\text{Softmax}(\mathbf{z} + C) = \text{Softmax}(\mathbf{z})$.
