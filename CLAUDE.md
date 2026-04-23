# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STRAP is a research framework for spatio-temporal graph neural networks (STGNNs) with continual/incremental learning. It addresses out-of-distribution generalization in streaming graph data (traffic, air quality, energy) where both node sets and temporal distributions shift over time (year by year).

## Environment Setup

```bash
conda env create -f environment.yaml
conda activate stg
```

Key dependencies: Python 3.11, PyTorch 2.2.1 (CUDA 12.1), PyTorch Geometric 2.5.3.

## Running Experiments

```bash
# Run a single experiment
python main.py --conf conf/AIR/trafficstream.json --gpuid 0 --seed 42 --logname "exp_name" --backbone "stgnn"

# Use --gpuid -1 for CPU
# --backbone options: stgnn, dcrnn, astgnn, tgcn

# Run predefined experiment suite
bash run.sh
```

## Architecture

### Data Flow (Year-by-Year Continual Learning)

1. Raw `.npz` files → Z-score normalization → train/val/test split (60/20/20)
2. `SpatioTemporalDataset` creates sliding windows (`x_len=12` input, `y_len=12` target)
3. Graph adjacency matrices loaded per year from `data/<DATASET>/graph/`
4. Model trained on subgraph of changed/new nodes (not the full graph)
5. Best model saved to `log/<DATASET>/`, loaded as init for the next year

### Key Modules

- **`main.py`** — Entry point; parses args, loads config, calls trainer year-by-year
- **`src/model/model.py`** — All model implementations (~2100 lines). Every model wraps a backbone (`STGNN_Backbone`, `DCRNN_Backbone`, `ASTGNN_Backbone`, `TGCN_Backbone`) + FC head + residual. The `STRAP` class implements retrieval-augmented continual learning; `TrafficStream_Model` is the main baseline.
- **`src/trainer/default_trainer.py`** — Training loop, validation, test evaluation, EWC/replay integration
- **`src/dataer/SpatioTemporalDataset.py`** — PyTorch Geometric Dataset; supports subgraph mode for incremental training
- **`src/model/detect_default.py`** — Gradient-based influence node detection (identifies nodes affected by distribution shift)
- **`src/model/ewc.py`** — Elastic Weight Consolidation (prevents catastrophic forgetting)
- **`src/model/replay.py`** — Experience replay node selection

### Continual Learning Pipeline

Each year the trainer:
1. Loads the previous year's best checkpoint
2. Detects changed/new nodes via gradient influence (`detect_strategy: "feature"` or `"influence"`)
3. Extracts a k-hop subgraph around those nodes (`num_hops`)
4. Trains only on the subgraph with EWC regularization + experience replay
5. Validates and saves the best model (patience=5)

## Configuration

All configs are JSON files in `conf/<DATASET>/<method>.json`. Key fields:

| Field | Purpose |
|---|---|
| `begin_year` / `end_year` | Year range for continual learning |
| `method` | Which model class to instantiate (e.g., `"TrafficStream"`, `"STRAP"`) |
| `backbone` | STGNN backbone type (overridable via CLI) |
| `strategy` | `"incremental"` or `"retrain"` |
| `detect`, `detect_strategy` | Node detection for subgraph selection |
| `ewc`, `ewc_lambda` | EWC regularization |
| `replay`, `replay_strategy` | Replay strategy (e.g., `"inforeplay"`) |
| `data_process` | `0` = load preprocessed, `1` = regenerate from raw |
| `init` | Load previous year's model as initialization |
| `gcn.hidden_channel` | Hidden dimension for GCN layers |

## Model Pattern

All models follow this structure:
```python
class MyModel(nn.Module):
    def __init__(self, args):
        self.backbone = SomeBackbone(args)
    
    def forward(self, data, adj):   # Returns predictions
    def feature(self, data, adj):   # Returns intermediate features (used for node detection/retrieval)
```

The `feature()` method is required by the detection and retrieval components — any new model must implement it.

## Evaluation Metrics

Reported metrics: MAE, RMSE, MAPE — computed in `utils/` and logged to `log/<DATASET>/`.
