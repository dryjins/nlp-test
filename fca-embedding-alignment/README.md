# FCA Embedding Alignment

Comparing neural architectures for projecting GloVe word embeddings onto interpretable semantic axes — **gender** and **socioeconomic status** — using orthogonality constraints.

## Overview

Word embeddings encode rich semantic information, but that information is distributed across hundreds of dimensions in an unstructured way. This project learns a low-dimensional projection (300d → 3d) that maps embeddings onto disentangled, interpretable axes.

Four projection architectures are compared:

| Model | Description |
|---|---|
| `linear` | Simple linear map, no bias |
| `linear_bias` | Linear map with a learnable bias term |
| `film_gamma` | FiLM-conditioned: scales input by `γ(condition)` before projecting |
| `film_gamma_beta` | Full FiLM: `γ(condition) · x + β(condition)` before projecting |

## Results

Models are evaluated on three criteria:

- **Analogy accuracy** — does `king − man + woman ≈ queen` in the projected space?
- **Axis purity** — do word-pair differences align with the intended axis?
- **Axis leakage** — does information from one axis bleed into the other?

| Model | Analogy cos ↑ | Gender purity ↑ | Status purity ↑ | Gender leakage ↓ | Status leakage ↓ |
|---|---|---|---|---|---|
| `linear` | **0.9877** | **0.9937** | 0.9764 | **0.0325** | **0.0742** |
| `linear_bias` | 0.9871 | 0.9876 | **0.9791** | 0.0900 | 0.1416 |
| `film_gamma` | 0.8757 | 0.5115 | 0.1563 | 0.3690 | 0.5518 |
| `film_gamma_beta` | 0.6048 | −0.1459 | 0.1688 | 0.6066 | 0.5824 |

**Finding:** Simple linear projection outperforms the more complex FiLM-conditioned models on every metric. Adding conditioning via FiLM hurts disentanglement significantly.

## Data

- **Embeddings:** [GloVe](https://nlp.stanford.edu/projects/glove/) `glove-wiki-gigaword-300` (400k vocabulary, 300 dimensions), loaded via `gensim`
- **Gender axis:** 100 hand-curated word pairs (e.g. *he/she*, *king/queen*, *actor/actress*) — 83 pass quality threshold
- **Status axis:** 106 word pairs encoding social hierarchy (e.g. *emperor/peasant*, *master/servant*) — 51 pass quality threshold
- Quality threshold: cosine similarity to mean axis direction > 0.3

## Project Structure

```
.
├── configs/                  # Hydra config files per model
│   ├── linear.yaml
│   ├── linear_bias.yaml
│   ├── film_gamma.yaml
│   └── film_gamma_beta.yaml
├── notebooks/                # Experiment notebooks (one per model + comparison)
│   ├── 01_linear.ipynb
│   ├── 02_linear_bias.ipynb
│   ├── 03_film_gamma.ipynb
│   ├── 04_film_gamma_beta.ipynb
│   └── 05_comparison.ipynb
├── src/                      # Core library
│   ├── data.py               # Data loading, pair filtering, quality analysis
│   ├── models.py             # Model architectures + factory function
│   ├── losses.py             # Alignment loss + orthogonality regularization
│   ├── train.py              # Training loop with early stopping
│   ├── evaluate.py           # Evaluation metrics
│   └── visualize.py          # 3D interactive visualizations (Plotly)
└── results/                  # Saved models and HTML visualizations
```

## Loss Function

Training minimizes a combined loss:

```
L = alignment_loss + λ_ortho · orthogonality_loss
```

- **Alignment loss:** cosine similarity between projected word-pair differences and the target axis direction
- **Orthogonality loss:** penalizes non-orthogonal rows in the projection matrix, encouraging disentangled axes
- Default `λ_ortho = 0.1`

## Installation

```bash
# Clone the repo
git clone https://github.com/semthedev/fca-embedding-alignment.git
cd fca-embedding-alignment

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch numpy scikit-learn gensim plotly pandas
```

GloVe embeddings are downloaded automatically by `gensim` on first run.

## Usage

Open and run any notebook under `notebooks/`:

```bash
jupyter notebook notebooks/01_linear.ipynb
```

Or use the `src` modules directly:

```python
from src.data import load_data
from src.models import build_model
from src.train import train
from src.evaluate import run_all_evaluations

# Load data
train_loader, val_loader, embeddings = load_data(cfg)

# Build and train model
model = build_model(cfg)
model, history = train(model, train_loader, val_loader, cfg)

# Evaluate
metrics = run_all_evaluations(model, embeddings)
print(metrics)
```

## Configuration

Each model has a YAML config in `configs/`. Example (`configs/linear.yaml`):

```yaml
model: linear
input_dim: 300
output_dim: 3
lr: 1.0e-3
num_epochs: 1500
lambda_ortho: 0.1
batch_size: 32
```

FiLM models add a `condition_dim: 3` parameter for the conditioning network.