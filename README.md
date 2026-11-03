# segmentation-evaluation

A model-independent Python framework for evaluating semantic segmentation models.

The central design principle is that evaluation code should not depend on the
framework used to train a model. Model-specific inference adapters will expose
a common interface, while metrics, cost-sensitive decision rules, calibration,
and reporting will remain reusable.

## Planned capabilities

- Confusion matrices and segmentation metrics (Dice, IoU).
- Proper scoring rules (cross-entropy, Brier score, spherical score).
- Expected cost (EC) and normalized expected cost (NEC).
- Bayesian cost-sensitive decision rules.
- Decision-rule optimization and validation procedures.
- Probability caching and reproducible evaluation.
- Visualization and result export.
- Model adapters for frameworks such as PyTorch and JAX/Flax.

## Installation

For development:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd segmentation-evaluation

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev,images,plots,data]"
```

## Development status

The repository currently contains the initial package scaffold. Functionality
will be added incrementally, with tests for each component.

## Design contract

The common evaluation representation will be:

- Probability maps: `(H, W, K)`, where `K` is the number of classes.
- Ground-truth masks: `(H, W)`, containing integer class labels.

Model-specific preprocessing, checkpoint loading, and inference will be kept
outside the model-independent metric and evaluation modules.
