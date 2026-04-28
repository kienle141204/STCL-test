# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STRAP is a research framework for spatio-temporal graph neural networks (STGNNs) with continual/incremental learning. It addresses out-of-distribution generalization in streaming graph data (traffic, air quality, energy) where both node sets and temporal distributions shift over time (year by year).

## Environment Setup

```bash
conda env create -f environment.yaml
conda activate stg
```

Key dependencies: Python 3.11, PyTorch 2.2.1 (CUDA 12.1), PyTorch Geometric 2.5.3, Biopython (for `kcluster` in `utils/common_tools.py`).

## Running Experiments

```bash
# Most methods — run with main.py
python main.py --conf conf/AIR/trafficstream.json --gpuid 0 --seed 42 --logname "exp_name" --backbone_type "stgnn"

# STKEC method requires its own entry point
python stkec_main.py --conf conf/AIR/stkec.json --gpuid 0 --seed 42 --backbone_type "stgnn" --logname "stkec_st"

# Use --gpuid -1 for CPU
# --backbone_type options: stgnn, dcrnn, astgnn, tgcn
# Note: run.sh uses abbreviated --backbone which argparse resolves to --backbone_type

# Run predefined experiment suite
bash run.sh

# Generate Excel report from results CSV
python generate_results_excel.py --input results.csv --output results.xlsx
```

Results are appended to `results.csv` (or `results_kprompt.csv` for `KPrompt` method) after each run via `utils/common_tools.py:save_results_csv`.

## Architecture

### Data Flow (Year-by-Year Continual Learning)

1. Raw `.npz` files → Z-score normalization → train/val/test split (60/20/20)
2. `SpatioTemporalDataset` creates sliding windows (`x_len=12` input, `y_len=12` target)
3. Graph adjacency matrices loaded per year from `data/<DATASET>/graph/`
4. Model trained on subgraph of changed/new nodes (not the full graph)
5. Best model saved to `log/<DATASET>/`, loaded as init for the next year

### Key Modules

- **`main.py`** — Entry point for all methods except STKEC; parses args, loads config, calls trainer year-by-year
- **`stkec_main.py`** — Separate entry point for the STKEC method (uses `src/trainer/stkec_trainer.py`)
- **`src/model/model.py`** — All model implementations (~2100+ lines). Every model wraps a backbone (`STGNN_Backbone`, `DCRNN_Backbone`, `ASTGNN_Backbone`, `TGCN_Backbone`) + FC head + residual. The `STRAP`/`RAP_Model` class implements retrieval-augmented continual learning; `TrafficStream_Model` is the main baseline.
- **`src/trainer/default_trainer.py`** — Training loop, validation, test evaluation, EWC/replay integration
- **`src/trainer/stkec_trainer.py`** — Trainer variant for STKEC with cluster loss
- **`src/dataer/SpatioTemporalDataset.py`** — PyTorch Geometric Dataset; supports subgraph mode for incremental training
- **`src/model/detect_default.py`** — Gradient-based influence node detection (identifies nodes affected by distribution shift)
- **`src/model/detect_stkec.py`** — STKEC-specific detection variant
- **`src/model/ewc.py`** — Elastic Weight Consolidation (prevents catastrophic forgetting)
- **`src/model/replay.py`** — Experience replay node selection
- **`generate_results_excel.py`** — Reads `results.csv`, aggregates by method × backbone × dataset, outputs styled Excel with pivot tables

### Continual Learning Pipeline

Each year the trainer:
1. Loads the previous year's best checkpoint
2. Detects changed/new nodes via gradient influence (`detect_strategy: "feature"` or `"influence"`)
3. Selects top-5% nodes by KL divergence score + newly added nodes + replay nodes
4. Extracts a k-hop subgraph around those nodes (`num_hops`)
5. Trains only on the subgraph with EWC regularization + experience replay
6. Validates and saves the best model (patience=5); checkpoint filename is the MAE value (e.g. `16.69.pkl`)

### Model Checkpoint Convention

Checkpoints are saved as `log/<DATASET>/<logname>-<seed>/<year>/<mae>.pkl`. `load_best_model` selects the checkpoint with the **lowest filename** (sorted ascending = lowest MAE). When skipping a year with no changed nodes, the previous year's checkpoint is copied forward.

## Configuration

All configs are JSON files in `conf/<DATASET>/<method>.json`. Key fields:

| Field | Purpose |
|---|---|
| `begin_year` / `end_year` | Year range for continual learning |
| `method` | Which model class to instantiate (e.g., `"TrafficStream"`, `"RAP"`, `"STKEC"`) |
| `backbone` | STGNN backbone type (overridable via `--backbone_type` CLI) |
| `strategy` | `"incremental"` or `"retrain"` |
| `detect`, `detect_strategy` | Node detection for subgraph selection (`"feature"` or `"influence"`) |
| `ewc`, `ewc_lambda` | EWC regularization |
| `replay`, `replay_strategy` | Replay strategy (e.g., `"inforeplay"`) |
| `data_process` | `0` = load preprocessed, `1` = regenerate from raw |
| `init` | Load previous year's model as initialization |
| `gcn.hidden_channel` | Hidden dimension for GCN layers |
| `incremental_train_ratio` | Fraction of training data to use for subsequent years (default 1.0) |
| `load_first_year` | `1` = skip retraining first year, load from `first_year_model_path` |

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

To register a new model, add it to the `methods` dict in `main.py`:
```python
vars(args)["methods"] = {..., 'MyModel': MyModel_Class}
```

## Evaluation Metrics

Reported at horizons 3, 6, 12, and Avg: MAE, RMSE, MAPE — logged per year to `log/<DATASET>/<logname>-<seed>/<logname>.log` and aggregated into `results.csv`.

## Datasets

- **AIR**: Air quality, 2016–2019, `data/AIR/`
- **PEMS**: Traffic (PeMS), 2011–2017, `data/PEMS/`
- **ENERGY-Wind**: Wind energy, year-indexed starting from 0, `data/ENERGY-Wind/`
