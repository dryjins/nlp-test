# Interpretable Word Embedding Projection via FiLM-inspired Axis Alignment

## 1. Project Overview & Motivation
We are exploring a method to project high-dimensional word embeddings (GloVe 300D) into a lower-dimensional space (e.g., 3D) while preserving specific semantic arithmetic properties (e.g., `King - Man + Woman = Queen`).

Unlike standard dimensionality reduction (PCA/t-SNE) which results in entangled latent features, our goal is **Interpretable Axis Alignment**. We want to force specific semantic concepts (like gender or royalty) to align with specific geometric axes (x, y, z) using a supervised approach inspired by **FiLM (Feature-wise Linear Modulation)**.

**Key Concept:** Instead of just compressing data, we use FiLM-like modulation to "select" and "align" semantic axes.

---

## 2. Shift in Strategy: From "Compression" to "Alignment"

### ⛔ Old Approach: Unsupervised Compression
*   Simply training an Autoencoder to compress 300D $\rightarrow$ 3D.
*   **Problem:** The resulting 3D axes are arbitrary. "Gender" might be a mix of x and y, making vector arithmetic messy and uninterpretable.

### ✅ New Approach: Supervised Axis Alignment (XAI Perspective)
*   We explicitly define what each axis represents.
*   **Constraint:** "The vector difference between `Man` and `Woman` must align parallel to the X-axis $(1, 0, 0)$."
*   **Result:** The X-axis becomes the **"Gender Axis"**. The Y-axis becomes the **"Royalty Axis"**, etc.

---

## 3. Experimental Setup

### Phase 1: Data Preparation (The "Anchor" Sets)
Do not use the entire GloVe vocabulary immediately. We need clear contrastive pairs to learn the axes.

**Prepare a dataset of Semantic Pairs:**
*   **Gender Pairs (for X-axis):**
    *   (Man, Woman), (King, Queen), (Prince, Princess), (Actor, Actress), (Father, Mother)...
    *   *Target Direction:* $(1, 0, 0)$
*   **Royalty/Status Pairs (for Y-axis):**
    *   (King, Man), (Queen, Woman), (Prince, Boy), (Princess, Girl)...
    *   *Target Direction:* $(0, 1, 0)$

### Phase 2: Modeling
Create a simple **Learnable Projector**.

*   **Architecture:**
    *   Input: 300D GloVe Vector
    *   Layer: Linear Transformation ($W \in \mathbb{R}^{3 \times 300}$)
    *   Output: 3D Latent Vector ($z$)

### Phase 3: The Loss Function (Crucial)
The loss function is the mechanism that enforces interpretability. It consists of two parts:

$$ L_{total} = L_{align} + \lambda L_{ortho} $$

1.  **Alignment Loss ($L_{align}$):**
    For every pair $(a, b)$ in the Gender set:
    $$ z_{diff} = \text{Encoder}(a) - \text{Encoder}(b) $$
    $$ L_{gender} = 1 - \text{CosineSimilarity}(z_{diff}, [1, 0, 0]) $$
    *(Forces gender difference to lie on the X-axis)*

2.  **Orthogonality/Reconstruction (Optional but recommended):**
    Ensure the learned axes are not correlated (e.g., Gender and Royalty should be independent).

---

## 4. How to Run the Experiment (Step-by-Step)

1.  **Load GloVe:** Load the pre-trained GloVe 300D vectors.
2.  **Extract Pairs:** Filter the "Anchor" pairs listed in Phase 1.
3.  **Train the Projector:**
    *   Initialize a simple `nn.Linear(300, 3)`.
    *   Train using the **Alignment Loss** described above.
4.  **Verification (The "Aha!" Moment):**
    *   Take the trained model.
    *   Compute: $v = \text{Project}(\text{King}) - \text{Project}(\text{Man}) + \text{Project}(\text{Woman})$.
    *   Check the Euclidean distance between $v$ and $\text{Project}(\text{Queen})$ in 3D space.
    *   **Visual Check:** Plot the 3D points. You should see "Men" on one side of the X-axis and "Women" on the other, regardless of their royalty status.

## 5. Connection to FiLM
While this initial experiment uses a linear projector, the underlying principle is **Feature-wise Modulation**. By enforcing these constraints, we are essentially training the network to "attend" to specific dimensions of the 300D input that correspond to our human concepts, filtering out noise—just like FiLM selects features based on a condition.

---

### 🚀 Next Steps
Once this linear alignment works, we can introduce **Non-linear FiLM layers** to handle polysemy (words with multiple meanings) by conditioning the projection on the context of the sentence.
