# Research: Abstracting Concepts from Word Embeddings via FCA Lattice

## Goal

The core objective is to **explicitly recover abstract concepts** that are already implicitly encoded
in word embeddings (word2vec, GloVe, Dolma-based vectors) by using a Formal Concept Analysis (FCA)
lattice structure.

Word co-occurrence training encodes latent semantic abstractions (e.g., `cat`, `dog` → `animal`,
`pet`) but leaves them entangled in a continuous, unstructured vector space. We want to surface
these abstractions as explicit, interpretable nodes in a concept lattice.

---

## Core Hypothesis

> Distributional embeddings implicitly learn hypernym-level structure through co-occurrence
> statistics. A concept lattice, induced from those embeddings via FCA, can make that structure
> explicit and navigable.

---

## Key Observations

- The join of `cat` and `dog` in the embedding space does not point to a single parent; it can
  align with `animal`, `pet`, `mammal`, or `companion_animal` — i.e., multiple upward paths exist.
- In FCA terms: multiple formal concepts can share the same pair `{cat, dog}` in their extent,
  each with a different intent (attribute set), corresponding to different abstraction axes.
- We want a mathematical operator that maps a set of child concepts to one or more stable,
  higher-level concept nodes in the lattice — i.e., an **upward operator** from small extents to
  larger extents.

---

## Proposed Architecture: FiLM-based Upward Operator

The upward operator is defined as a **FiLM (Feature-wise Linear Modulation) network**:

1. **Input**: child word embeddings `w_1, w_2, ..., w_n ∈ R^D`
2. **Condition vector** (order-invariant):
   - `c = Pool([w_1, ..., w_n])` — e.g., mean or `[avg, max]` concatenation
3. **FiLM generator**:
   - `γ, β = G(c)` — small MLP (LeakyReLU hidden, linear output), outputs per-axis scale/shift
4. **Global concept frame** `P ∈ R^(D×r)` (shared axis matrix, learned or pre-initialized):
   - Base concept code: `u_0 = x_0^T P`
   - Modulated code: `u+ = γ ⊙ u_0 + β`
5. **Output** (abstract concept direction):
   - `z_raw = P u+`, then L2-normalize → `z`

This operator is **order-invariant** by construction (pooling-based condition), which matches the
set-theoretic nature of the FCA join operation.

### Design Choices

| Component | Choice | Reason |
|---|---|---|
| Pooling | avg or [avg, max] | Set-invariance (commutative join) |
| Hidden activation | LeakyReLU | Preserves sign information in embedding space |
| Output layer | Linear + L2 normalize | Avoids distorting embedding geometry |
| Architecture | Shallow MLP (not 1D CNN) | No positional/adjacency bias needed for set join |

---

## FCA / Lattice Side

- **Formal context** is derived by thresholding axis activations:
  - `A[i, k] = 1 if u_{i,k} ≥ θ`
- **Formal concepts** `(X, Y)` are computed via standard FCA closure (or fuzzy variant with
  Łukasiewicz residuum).
- Concept stability (Δ-stability) is used to filter for robust, interpretable nodes.
- The FiLM operator output `z` is used as an **extent seed** — we query the lattice for concepts
  whose extent contains the child set and whose intent aligns with `z`.

---

## Efficient Search: HNSW

Because the space of candidate abstract concept axes spans the full vocabulary, brute-force search
is infeasible. We use HNSW (Hierarchical Navigable Small World) for:

- **Phase 1**: ANN index over word embeddings → fast top-k parent candidate retrieval given `z`
- **Phase 2** (later): ANN index over NCA axis codes `u_i` → concept-space regularization,
  constraining FiLM-induced movement to locally plausible neighborhoods

---

## Training Signal

The FiLM operator is trained with:

- **Parent attraction loss**: cosine similarity between `z` and weak parent proxies (e.g.,
  hypernym neighbors, cluster centroids)
- **Child repulsion loss**: push `z` away from the input child embeddings
- **Δ-stability regularizer** (optional): prefer `z` directions that correspond to stable formal
  concepts in the induced lattice

---

## Coordinate System Note

- The 300d embedding space uses a single **global coordinate system**.
- Semantic regions occupy different local submanifolds, so effective axes vary by region.
- NCA's global axis matrix `P` approximates a shared tangent frame across the manifold.
- FiLM modulates coordinates (values) while keeping the frame `P` fixed — enabling
  context-dependent abstraction within a consistent geometric reference.

---

## Summary Pipeline

```
child words {w_1, ..., w_n}
        ↓  (order-invariant pooling)
   condition vector c
        ↓  (FiLM generator G)
   γ, β  (per-axis scale/shift)
        ↓  (modulate base concept code u_0)
   u+ = γ ⊙ u_0 + β
        ↓  (project back via P, normalize)
   z  ← abstract concept direction
        ↓  (HNSW retrieval + FCA closure)
   stable formal concepts covering {w_1,...,w_n}
```
