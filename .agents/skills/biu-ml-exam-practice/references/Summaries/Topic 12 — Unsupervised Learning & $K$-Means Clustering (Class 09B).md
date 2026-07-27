---
type: uni_general
course: "[[English]]"
status: 🟢 Active
order: Summary
date_added: 2026-06-21
---
 # Topic 12 — Unsupervised Learning & $K$-Means Clustering (Class 09B)

**MOC:** [[Machine learning MOC]]
**Course:** [[English]]

# Study Guide: Topic 12 — Unsupervised Learning & $K$-Means Clustering (Class 09B)

## 1. Context: Transitioning to Unlabeled Data Spaces

Until this point, every model family explored operated in a **Supervised** capacity, relying on a matching ground-truth target vector $\mathbf{y}$ to calculate empirical risk. We now transition to **Unsupervised Learning**, where our training observations contain absolutely no external human annotations or explicit scalar reward metrics:

$$\mathcal{D} = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n\}, \quad \mathbf{x}_i \in \mathbb{R}^d$$

Instead of finding predictive mappings to targets, our mathematical goal shifts toward uncovering the latent structural layout, density properties, or geometric groupings built directly into the feature space distribution itself.

## 2. Hard Partitioning & The Clustering Problem

**Clustering** is the task of grouping data points such that items assigned to the same cluster share high structural geometric proximity, while items split across separate clusters are highly distinct.

Under a **Hard Partitioning** strategy, we segment our feature space into exactly $K$ discrete, non-overlapping groups. Let $z_{i,k} \in \{0, 1\}$ be a binary indicator flag that tracks assignments:

$$z_{i,k} = \begin{cases} 1 & \text{if sample } \mathbf{x}_i \text{ is assigned to cluster } k \\ 0 & \text{otherwise} \end{cases}$$

Every single sample is strictly required to belong to exactly one cluster: $\sum_{k=1}^K z_{i,k} = 1$ for all $i$.

## 3. The $K$-Means Objective Function (Distortion Risk)

To formulate clustering as an optimization task, we define a prototype centroid vector $\boldsymbol{\mu}_k \in \mathbb{R}^d$ for each cluster $k$. We measure group cohesion by calculating the sum of squared Euclidean distances from every individual sample to its assigned cluster centroid.

This formulation defines the **Distortion Risk / Inertia Cost** function $J$:

$$J(Z, M) = \sum_{i=1}^n \sum_{k=1}^K z_{i,k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

where $Z \in \{0,1\}^{n \times K}$ represents the complete assignment matrix, and $M = \{\boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_K\}$ is the set of prototype centroids.

Finding the global minimum for this objective function is an NP-hard problem because of the combinatorially vast number of ways to partition $n$ continuous points into $K$ discrete groups.

## 4. Lloyd's Algorithmic Protocol (Coordinate Descent)

The standard **$K$-Means Algorithm** (Lloyd's Algorithm) find a local minimum for the distortion risk by executing a two-step iterative **Coordinate Descent** optimization loop:

### Step 1: Initialization

Select $K$ initial coordinate locations to act as the starting prototypes $\{\boldsymbol{\mu}_1^{(0)}, \dots, \boldsymbol{\mu}_K^{(0)}\}$ (e.g., choosing $K$ random samples from the dataset).

### Step 2: Alternate Until Convergence

#### A. The Assignment Step (Optimize $Z$ given fixed $M$)

For fixed, unchanging centroid locations, minimize $J$ by assigning every data point to its closest prototype centroid vector using the Euclidean distance metric:

$$z_{i,k}^{(t)} = \begin{cases} 1 & \text{if } k = \arg\min_{j \in \{1,\dots,K\}} \|\mathbf{x}_i - \boldsymbol{\mu}_j^{(t-1)}\|^2 \\ 0 & \text{otherwise} \end{cases}$$

#### B. The Refitting / Update Step (Optimize $M$ given fixed $Z$)

For fixed assignments, recalculate each centroid vector $\boldsymbol{\mu}_k$ to minimize the local squared variance. Taking the vector derivative of $J$ with respect to $\boldsymbol{\mu}_k$ and setting it to zero proves that the optimal centroid location is the exact **arithmetic mean** of all data points currently assigned to that group:

$$\boldsymbol{\mu}_k^{(t)} = \frac{\sum_{i=1}^n z_{i,k}^{(t)} \mathbf{x}_i}{\sum_{i=1}^n z_{i,k}^{(t)}} = \frac{1}{|S_k|} \sum_{\mathbf{x}_i \in S_k} \mathbf{x}_i$$

where $S_k$ is the subset of samples assigned to cluster $k$.

### Computational Complexity

Per single iteration loop: $\mathcal{O}(n \cdot K \cdot d)$, where $n$ is the number of data points, $K$ is the number of clusters, and $d$ is the feature dimensionality. This linear scaling profile makes $K$-means highly efficient for massive datasets.

## 5. Mathematical Proof of Monotonic Convergence (From 2023 Moed A)

A common exam proof requires showing that **Lloyd's $K$-means algorithm is guaranteed to converge to a stable local minimum because the distortion risk cost function decreases monotonically at every single step.**

### Proof Structure

Let $J(Z^{(t)}, M^{(t)})$ be the cost value at iteration $t$.

1. **During the Assignment Step:** We update $Z^{(t)}$ to $Z^{(t+1)}$ while keeping the centroids $M^{(t)}$ fixed. Because every single point $\mathbf{x}_i$ is intentionally assigned to its absolute nearest centroid, the Euclidean distance term cannot increase:
    
    $$\|\mathbf{x}_i - \boldsymbol{\mu}_{z_i^{(t+1)}}^{(t)}\|^2 \leq \|\mathbf{x}_i - \boldsymbol{\mu}_{z_i^{(t)}}^{(t)}\|^2 \implies J(Z^{(t+1)}, M^{(t)}) \leq J(Z^{(t)}, M^{(t)})$$
    
2. **During the Refitting Step:** We update the centroids to $M^{(t+1)}$ while keeping assignments $Z^{(t+1)}$ fixed. Since the sample mean is the absolute mathematical minimizer for a sum of squared errors, any other coordinate choices would yield a larger or equal cost value:
    
    $$J(Z^{(t+1)}, M^{(t+1)}) \leq J(Z^{(t+1)}, M^{(t)})$$
    
3. Combining both inequalities proves the monotonic descent property:
    
    $$J(Z^{(t+1)}, M^{(t+1)}) \leq J(Z^{(t+1)}, M^{(t)}) \leq J(Z^{(t)}, M^{(t)})$$
    

Because the objective function $J$ is bounded below by $0$, and because the number of unique assignments for a finite dataset is finite, this monotonic decay guarantees that the algorithm must converge to a stable local minimum in a finite number of steps.

## 6. Python / NumPy Implementation Snippet

```python
import numpy as np

def kmeans_lloyds(X, K, max_iters=100, seed=42):
    """
    X: Data matrix of shape (n, d)
    K: Number of clusters
    """
    np.random.seed(seed)
    n, d = X.shape
    
    # 1. Initialization: pick K random rows from X as initial centroids
    random_indices = np.random.choice(n, K, replace=False)
    centroids = X[random_indices]
    
    assignments = np.zeros(n, dtype=int)
    
    for iteration in range(max_iters):
        # 2A. Assignment Step: Compute distance from all points to all centroids
        # Broad-cast geometry: (n, 1, d) - (1, K, d) -> (n, K, d)
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        new_assignments = np.argmin(distances, axis=1)
        
        # Check for convergence (no assignment shifts)
        if np.array_equal(assignments, new_assignments) and iteration > 0:
            break
        assignments = new_assignments
        
        # 2B. Refitting Step: Compute the mean of each cluster group
        for k in range(K):
            cluster_points = X[assignments == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)
                
    return assignments, centroids
```

## 7. Solved Exam-Style Examples

### Example 1: Calculating 1D Centroid Updates

**Problem Statement:** You have a 1D dataset containing 6 data points: $\mathcal{D} = \{1, 2, 4, 7, 9, 11\}$. You initialize your $K=2$ cluster centroids at coordinates $\mu_1 = 3$ and $\mu_2 = 8$. Perform one complete iteration of Lloyd's algorithm: find the initial assignments, and compute the updated centroid locations.

#### Step-by-Step Analytical Derivation

1. **Assignment Phase:** Calculate the absolute distance from each point to both centroids ($\mu_1=3, \mu_2=8$) and assign the point to the closer centroid:
    
    - Point $1$: $|1-3|=2$, $|1-8|=7 \rightarrow$ Assigned to Cluster 1.
        
    - Point $2$: $|2-3|=1$, $|2-8|=6 \rightarrow$ Assigned to Cluster 1.
        
    - Point $4$: $|4-3|=1$, $|4-8|=4 \rightarrow$ Assigned to Cluster 1.
        
    - Point $7$: $|7-3|=4$, $|7-8|=1 \rightarrow$ Assigned to Cluster 2.
        
    - Point $9$: $|9-3|=6$, $|9-8|=1 \rightarrow$ Assigned to Cluster 2.
        
    - Point $11$: $|11-3|=8$, $|11-8|=3 \rightarrow$ Assigned to Cluster 2.
        
2. Isolate the resulting data partitions:
    
    - $S_1 = \{1, 2, 4\}$
        
    - $S_2 = \{7, 9, 11\}$
        
3. **Refitting Phase:** Compute the new arithmetic means for each cluster:
    
    $$\mu_1^{(1)} = \frac{1+2+4}{3} = \frac{7}{3} \approx 2.3333$$
    
    $$\mu_2^{(1)} = \frac{7+9+11}{3} = \frac{27}{3} = 9.0000$$
    

**Final Exam Answer:** After one iteration, the data points are partitioned into $S_1 = \{1, 2, 4\}$ and $S_2 = \{7, 9, 11\}$, and the updated cluster centroids are **$\mu_1 = \frac{7}{3}$** and **$\mu_2 = 9$**.

### Example 2: Structural Sensitivity to Outliers

**Problem Statement:** Explain why the $K$-means algorithm is highly sensitive to anomalous noise outliers, and describe how this sensitivity impacts the resulting Voronoi decision boundaries.

#### Analytical Evaluation

- **The Root Mathematical Cause:** The objective function minimizes _squared_ Euclidean distances ($\|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$). Squaring distances disproportionately penalizes large gaps.
    
- **The Centroid Shift Effect:** If a single anomaly point lies extremely far away from the true data density, its huge squared distance term dominates the gradient calculation during the refitting phase. To reduce this specific penalty, the algorithm forces the nearest centroid vector $\boldsymbol{\mu}_k$ to shift significantly away from its dense local core cluster toward the distant outlier.
    
- **Voronoi Boundary Alteration:** Because the geometric boundaries separating clusters are defined by the perpendicular bisectors between centroids, shifting a centroid to accommodate an outlier warps the Voronoi cells, often causing nearby regular data points to be misassigned to incorrect groups.
    

---
## 🔗 Navigation
**Previous:** [[ ]] | **Next:** [[ ]]