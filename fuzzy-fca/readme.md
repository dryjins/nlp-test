# Experiments to Run Next: FiLM + Fuzzy FCA over Word Embeddings

This document describes follow-up experiments to evaluate and extend the FiLM-based “parent direction” operator and its interaction with fuzzy Formal Concept Analysis (FCA).

We assume you already have:

- A text-based embedding file (e.g., Dolma 300d / GloVe).
- The FiLM training notebook that:
  - loads embeddings,
  - defines hand-crafted triplets (child1, child2, parent),
  - trains a FiLM module with pure parent-attraction loss,
  - and prints evaluation logs like:

  - `cos(avg, parent)`, `cos(FiLM, parent)`
  - top-k neighbors of the average vector vs FiLM output.

## 0. Embedding Source

You can use one of the following options:

1. **Dolma 300d (large) — recommended for richer structure**

   - Download from Kaggle:

     - https://www.kaggle.com/datasets/nguyendatphik18ct/sdfghjjjjjj

   - After downloading and unpacking, point the notebook to the text file, e.g.:

     ```python
     EMB_PATH = "data/dolma_300_2024_1.2M.100_combined.txt"
     ```

2. **Smaller GloVe variant (lighter, easier to run)**

   - For quick experiments or low-resource environments, you may replace Dolma with a smaller GloVe file (e.g., `glove.6B.300d.txt`).
   - In that case, set:

     ```python
     EMB_PATH = "path/to/glove.6B.300d.txt"
     ```

The rest of the pipeline (FiLM, evaluation, fuzzy FCA) is agnostic to the specific embedding source, as long as it is in the standard `word dim1 ... dimD` text format.

---

## 1. Unseen Child-Pair Generalization

**Goal:** Check whether FiLM generalizes to new child pairs that were not seen during training, but share the *same* parent semantics.

### 1.1. Construct unseen test pairs

For each parent concept used in training (e.g., `animal`, `fruit`, `vehicle`, …):

- Training examples (already used):
  - `(cat, dog) -> animal`
  - `(tiger, lion) -> animal`
- Define **new unseen pairs** (not in training):
  - `(wolf, fox) -> animal`
  - `(cat, tiger) -> animal`
  - `(dog, wolf) -> animal`
- Do the same for other domains:
  - `fruit`: `(mango, grape)`, `(pear, orange)`, …
  - `vehicle`: `(truck, bus)`, `(car, motorcycle)`, …
  - `instrument`: `(violin, cello)`, `(drums, guitar)`, …

Make a separate list like:

```python
UNSEEN_TEST = [
    ("wolf", "fox", "animal"),
    ("cat", "tiger", "animal"),
    ("dog", "wolf", "animal"),
    ("mango", "grape", "fruit"),
    ("pear", "orange", "fruit"),
    ("truck", "bus", "vehicle"),
    ("violin", "cello", "instrument"),
    # ...
]
```

**Important:** Ensure these pairs were **not** part of the training triplet list.

### 1.2. Evaluation metrics for unseen pairs

For each `(w1, w2, wp)` in `UNSEEN_TEST`:

1. Compute:
   - average vector: `v_avg = 0.5 * (w1 + w2)`
   - FiLM output: `z = FiLM(w1, w2)`
2. Report:
   - `cos(v_avg, wp)`
   - `cos(z, wp)`
   - `cos(z, w1)`, `cos(z, w2)`
3. Retrieve top-k neighbors for `v_avg` and `z` (as in the current notebook):
   - Check:
     - Does `wp` appear in top-k for `z`?
     - Does `z` move closer to the semantic parent cluster (e.g., other animals, fruits, vehicles)?

**Success criteria (informal):**

- On average, `cos(z, wp)` > `cos(v_avg, wp)` for unseen pairs.
- For most unseen pairs, `wp` appears higher in the top-k list for `z` than for `v_avg`.
- The neighbor list of `z` is clearly “more abstract” (e.g., `animal`, `animals`, `pet`) than that of `v_avg` (which tends to be siblings like `cat`, `dog`, `wolf`, etc.).

---

## 2. Using FiLM Output as a Fuzzy FCA Extent Seed

**Goal:** Combine the embedding-level parent operator (FiLM) with fuzzy FCA, and see whether the FCA closure recovers expected objects such as `animal`, `pet`, etc., in the concept extent.

### 2.1. Object pool for FCA

Use (at least) these word sets as objects for FCA:

- Core words:
  - `cat`, `dog`, `animal`, `pet`, `tiger`, `lion`, …
- Plus additional words:
  - nearest neighbors of `cat`, `dog`, `animal`, `pet`,
  - and possibly top neighbors of other parents like `fruit`, `vehicle`, …

Let the object list be:

```python
OBJ_WORDS = [
    "cat", "dog", "animal", "pet",
    "tiger", "lion", "wolf", "fox",
    "apple", "banana", "fruit", "grape", "mango",
    "car", "bus", "vehicle", "truck", "automobile",
    # ...
]
```

Build:

- `X ∈ ℝ^{B×D}` by stacking embeddings for `OBJ_WORDS`.

### 2.2. Construct fuzzy semantic context

Reuse or adapt the semantic coordinate + fuzzy context code from `FuzzyFCA.ipynb`:

1. Define semantic axes (anchors), for example:

   - animal-related: `"animal"`, `"pet"`, `"mammal"`
   - fruit-related: `"fruit", "vegetable", "food"`
   - vehicle-related: `"vehicle"`, `"transport"`, `"car"`
   - etc.

2. For each object embedding in `X`:
   - Compute cosine similarity to each anchor.
   - Optionally apply a temperature-softmax to get discriminative coordinates.

3. Apply triangular membership functions along each coordinate dimension to build:
   - `context ∈ [0,1]^{B×M}`.

Ensure you have:

- `obj_names` (list of object words).
- `attr_names` (human-readable attribute names, e.g., `animal_axis_bin2`).

### 2.3. Build FiLM-based fuzzy extent seed

Given a child pair `(cat, dog)`:

1. Compute:

   - `z = FiLM(w_cat, w_dog)` (shape: `(D,)`).

2. Convert `z` into a fuzzy extent over `OBJ_WORDS`:

   - For each object `g`, compute `cos(X_g, z)`.
   - Apply a softmax over objects:

   - `A_seed[g] = softmax_g(cos(X_g, z) / τ)` for some temperature `τ` (e.g., `0.1`–`0.5`).

3. This `A_seed ∈ [0,1]^B` is the extent seed.

### 2.4. Fuzzy closure and interpretation

Use the fuzzy closure function from the fuzzy FCA code:

- `(A_closed, B_vec) = fuzzy_closure(A_seed, context)`

Then analyze:

1. Extent memberships:

   - Check `A_closed` entries for:
     - `cat`, `dog`, `animal`, `pet`, `tiger`, `lion`, `wolf`, `fox`, etc.
   - Use a threshold `θ` (e.g., `0.5` or `0.7`) to decide which objects are “strongly in” the concept.

2. Intent memberships:

   - Examine which attributes in `B_vec` are high (≥ θ).
   - Interpret them as “this FiLM-based join concept lives in the region of [animalness / petness / fruitness / ...]”.

**Questions to answer:**

- For `(cat, dog)`:
  - Does the extent of the FiLM-based concept include `animal` and/or `pet` with high membership?
  - How does this compare to:
    - Closure from a purely average-based seed,
    - Closure from single-object seeds (`cat` only, `dog` only)?

- For other domains:
  - `(apple, banana)`: Does `fruit` appear strongly in the extent?
  - `(car, bus)`: Does `vehicle` appear strongly?

---

## 3. Weak-Parent (Noisy Supervision) Experiments

**Goal:** Remove the assumption that the true parent word is known, and instead use noisy / approximate parents derived from embeddings themselves (self-supervision). Test whether FiLM still produces meaningful “upper-level” vectors.

### 3.1. Weak parent generation via centroid neighbors

For each sibling group `S = {w_1, …, w_n}`:

1. Compute centroid:

   - `c = (1 / n) * Σ v(w_i)`.

2. Find nearest neighbors of `c` in the embedding space.

3. Choose the top candidate(s) as weak parent(s), e.g.:

   - `weak_parent = top-1 neighbor of c`,
   - or even top-3 as multiple candidate parents.

Example sibling groups:

- `{cat, dog, rabbit, hamster}` → weak parent ≈ `pet` or `animal`.
- `{apple, pear, banana, orange}` → weak parent ≈ `fruit`.
- `{car, truck, bus, motorcycle}` → weak parent ≈ `vehicle`.

Store them as weak triplets:

```python
WEAK_TRIPLETS = [
    ("cat", "dog", "weak_parent_for_animal_cluster"),
    ("apple", "banana", "weak_parent_for_fruit_cluster"),
    # ...
]
```

### 3.2. Train FiLM on weak parents

Repeat the FiLM training using `WEAK_TRIPLETS` instead of (or in addition to) the manually curated `TRIPLETS`. Use the same loss:

- `loss = 1 - cos(FiLM(w1, w2), weak_parent)`.

### 3.3. Evaluate versus the *true* parent

For cases where you know the true parent (e.g., `animal`, `fruit`, `vehicle`):

1. Train FiLM **only** with weak parents.

2. At evaluation time, measure:

   - `cos(FiLM(w1, w2), true_parent)` (even though true parent was never used as a training label).
   - Top-k neighbors of FiLM output:
     - Does the true parent appear near the top?
     - Are neighbors still “upper-level” words (e.g., animals, fruits, vehicles)?

### 3.4. Compare three conditions

For a fixed test set of child pairs:

1. FiLM trained on **true parents** (strong supervision).
2. FiLM trained on **weak parents only**.
3. **No FiLM** — just the average vector baseline.

Compare:

- `cos(output, true_parent)`,
- top-k neighbor quality (rank of true parent),
- qualitative neighbor lists (“does it look like an upper-level concept?”).

---

## 4. Reporting and Visualization

For each experiment, collect:

- A small table summarizing:
  - average `cos(avg, parent)`,
  - average `cos(FiLM, parent)`,
  - average `cos(FiLM, child)` (to confirm it moves away from children).
- A few representative neighbor lists (as you already printed) for:
  - successful cases,
  - borderline cases,
  - failure cases.

Try to categorize failure modes, for example:

- Parent word is too polysemous (“royalty”, “city”, etc.).
- Sibling group is heterogeneous or noisy.
- Weak parent is itself too specific or off-domain.

These analyses will help refine:

- FiLM architecture (e.g., add repulsion loss, regularization),
- fuzzy FCA context design (anchors, bins, thresholds),
- and the overall pipeline for “join as upper-level concept direction”.
