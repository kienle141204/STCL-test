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


# NEW IDEAL
# GDAP: Graph-Diffused Adaptive Plasticity

Tài liệu đặc tả để implement phương pháp GDAP vào codebase continual learning hiện tại.

---

## 1. Bối cảnh bài toán

Codebase train một STGNN qua nhiều phase (năm) tuần tự trên các dataset như PEMS (traffic), AIR (air quality), ENERGY-Wind. Mỗi phase có:

- `data/X/FastData/<year>.npz` → `train_x, train_y, val_x, val_y, test_x, test_y`
- `data/X/graph/<year>_adj.npz` → adjacency matrix `A_t`, shape `(N_t, N_t)`, row-normalized
- `N_t` có thể tăng qua các năm (node mới thêm vào cuối)

Backbone mặc định là STGNN: `GCN1 → TCN → GCN2 → Residual → FC`, forward signature `model(data, adj) → [bs*N, y_len]`.

Constraint cứng: **không truy cập raw data của các phase cũ** khi train phase mới. Chỉ được phép giữ một small buffer (statistics, không phải raw samples).

---

## 2. Ý tưởng GDAP

### 2.1 Vấn đề hiện tại

Các method hiện tại (EWC, TrafficStream fine-tune, EAC frozen backbone) đều áp dụng regularization/plasticity đồng đều cho toàn bộ graph. Điều này bỏ qua hai thực tế quan trọng:

1. **Asymmetric shift**: Một số node shift mạnh (xây đường mới, chính sách mới), một số hoàn toàn ổn định. Ép tất cả node có cùng learning dynamics là suboptimal.
2. **Drift propagation**: Trong GNN, message passing làm cho drift tại node A lan sang hàng xóm B. Nếu A shift mạnh, B cũng cần được cho phép update, ngay cả khi signal của B ổn định.

### 2.2 Giải pháp

**Trước khi train** phase $t$, tính per-node plasticity weight $w_i \in (0,1)$ từ:
- Signal statistics của data năm mới (có sẵn, không vi phạm constraint)
- Thay đổi topology $\Delta A = A_t - A_{t-1}$

Sau đó **khuếch tán** drift score qua đồ thị và dùng nó để modulate gradient + EWC penalty theo từng node trong suốt quá trình training.

---

## 3. Công thức

### Bước 1: Tính raw drift score

Với mỗi node $i$, trước training phase $t$:

```
signal_shift_i = |mean(X_t[:, i]) - mean(X_{t-1}[:, i])| / (std(X_{t-1}[:, i]) + eps)
topo_shift_i   = sum(|A_t[i, :] - A_{t-1}[i, :]|)   # L1 norm của hàng i trong ΔA
s_i            = signal_shift_i + gamma * topo_shift_i
```

Nếu `N_t > N_{t-1}` (có node mới): gán `s_i = s_max` (node mới cần plasticity tối đa).

### Bước 2: Diffuse qua đồ thị

```
tilde_s = (I + beta * A_t) @ s      # shape (N_t,)
w       = sigmoid(tilde_s)          # shape (N_t,), mỗi phần tử trong (0, 1)
```

`beta` kiểm soát mức độ lan truyền. `beta=0` → không diffuse, chỉ dùng raw signal. `beta=1` → mỗi node lấy thêm trung bình weighted của hàng xóm.

### Bước 3: Modulate training

Trong mỗi training step, với batch có node indices `node_idx`:

```
w_batch = w[node_idx]               # shape (bs*N_sub,) nếu subgraph, hoặc (bs*N,)

loss_pred = CrossEntropyLoss hoặc MAELoss(pred, target)
loss_ewc  = EWC_regularization(model, fisher_dict, prev_params)

# Scale EWC penalty ngược với plasticity
loss_total = loss_pred + lambda_ewc * mean((1 - w_batch) * loss_ewc_per_node)
```

Ngoài ra, scale gradient của từng node bằng `w_i` qua gradient hook (optional, xem mục 5.3).

---

## 4. Thông tin cần lưu giữa các phase

GDAP cần lưu lại sau mỗi phase để dùng cho phase tiếp theo:

```python
gdap_buffer = {
    "node_mean": np.ndarray,   # shape (N_t,) — mean per node trên train set
    "node_std":  np.ndarray,   # shape (N_t,) — std per node trên train set
    "adj":       np.ndarray,   # shape (N_t, N_t) — A_{t} của phase vừa train
    "fisher":    dict,         # {param_name: tensor} — Fisher Information Matrix
    "prev_params": dict,       # {param_name: tensor} — snapshot params sau phase t
}
```

Đây là statistics, không phải raw data → không vi phạm constraint.

---

## 5. Hướng dẫn implement

### 5.1 Cấu trúc file cần tạo/sửa

```
src/model/model.py          ← thêm class GDAPModel (wrapper)
src/trainer/default_trainer.py ← thêm gdap_step() và gdap_prepare()
conf/<DATASET>/gdap.json    ← config file
main.py                     ← đăng ký method "GDAP"
```

### 5.2 Class GDAPModel

Không cần kiến trúc mới. GDAP là một **wrapper** quanh backbone hiện có:

```python
class GDAPModel(nn.Module):
    def __init__(self, backbone: nn.Module, args):
        super().__init__()
        self.backbone = backbone
        self.args = args
        # Buffers: không lưu vào state_dict nếu chưa compute
        self.register_buffer('plasticity_weights', None)  # shape (N,)
        self.register_buffer('fisher_diag', None)         # flattened FIM
        self.prev_params = {}   # dict, không register_buffer
        self.aux_loss = None

    def forward(self, data, adj):
        out = self.backbone(data, adj)
        return out

    def feature(self, data, adj):
        return self.backbone.feature(data, adj)
```

### 5.3 Hàm gdap_prepare() — chạy TRƯỚC khi train phase t

Gọi một lần trước epoch đầu tiên của phase mới:

```python
def gdap_prepare(model, train_loader, prev_buffer, adj_prev, adj_curr,
                 beta=1.0, gamma=0.5, eps=1e-6):
    """
    Tính plasticity weights w cho phase hiện tại.
    
    Args:
        model: GDAPModel
        train_loader: DataLoader của phase hiện tại (dùng để tính node stats)
        prev_buffer: dict với keys "node_mean", "node_std" từ phase trước
        adj_prev: np.ndarray (N_{t-1}, N_{t-1}) — adj phase trước
        adj_curr: np.ndarray (N_t, N_t) — adj phase hiện tại
        beta: diffusion strength
        gamma: weight của topology shift vs signal shift
    """
    N_curr = adj_curr.shape[0]
    N_prev = prev_buffer["node_mean"].shape[0]
    
    # --- Tính signal statistics của phase hiện tại ---
    # Accumulate per-node mean/std từ train_loader
    node_sum   = np.zeros(N_curr)
    node_sq    = np.zeros(N_curr)
    node_count = np.zeros(N_curr)
    
    for batch in train_loader:
        x = batch.x.cpu().numpy()          # shape (bs*N, T)
        # x được reshape để lấy per-node values
        # Cộng dồn statistics
        node_sum   += x.reshape(-1, N_curr, x.shape[-1]).mean(axis=(0,2)) * len(batch)
        node_count += len(batch)
    
    curr_mean = node_sum / node_count
    # Tính std tương tự (pass 2 hoặc Welford online)
    
    # --- Signal shift ---
    prev_mean = prev_buffer["node_mean"]
    prev_std  = prev_buffer["node_std"]
    
    # Pad nếu N_curr > N_prev (node mới)
    if N_curr > N_prev:
        pad = N_curr - N_prev
        prev_mean = np.concatenate([prev_mean, np.zeros(pad)])
        prev_std  = np.concatenate([prev_std,  np.ones(pad)])
    
    signal_shift = np.abs(curr_mean - prev_mean) / (prev_std + eps)
    
    # Node mới: set signal_shift = max (cần plasticity tối đa)
    if N_curr > N_prev:
        signal_shift[N_prev:] = signal_shift[:N_prev].max()
    
    # --- Topology shift ---
    # Pad adj_prev nếu cần
    if N_prev < N_curr:
        pad = N_curr - N_prev
        adj_prev_padded = np.zeros((N_curr, N_curr))
        adj_prev_padded[:N_prev, :N_prev] = adj_prev
    else:
        adj_prev_padded = adj_prev
    
    delta_A     = np.abs(adj_curr - adj_prev_padded)
    topo_shift  = delta_A.sum(axis=1)                    # L1 norm per row
    
    # --- Raw drift score ---
    s = signal_shift + gamma * topo_shift
    
    # Normalize s vào [0, max] để tránh sigmoid bão hòa hết về 1
    if s.max() > 0:
        s = s / s.max() * 3.0   # scale để sigmoid có range tốt
    
    # --- Diffuse ---
    I = np.eye(N_curr)
    tilde_s = (I + beta * adj_curr) @ s     # (N_curr,)
    
    # --- Plasticity weights ---
    w = 1 / (1 + np.exp(-tilde_s))          # sigmoid
    
    # Lưu vào model
    model.plasticity_weights = torch.tensor(w, dtype=torch.float32).to(model.args.device)
    
    return curr_mean, curr_mean  # trả về stats để lưu vào buffer
```

> **Lưu ý về data layout**: `data.x` trong DataLoader có shape `(bs*N, T)`. Để lấy per-node stats, cần reshape thành `(bs, N, T)` rồi average qua batch dimension và time dimension.

### 5.4 Hàm gdap_step() — thay thế training step thông thường

```python
def gdap_step(model, batch, adj, optimizer, lambda_ewc=0.1):
    """
    Một training step với GDAP loss.
    """
    optimizer.zero_grad()
    
    pred = model(batch, adj)                 # (bs*N, y_len)
    loss_pred = mae_loss(pred, batch.y)      # scalar
    
    # EWC loss (nếu có fisher và prev_params)
    if model.fisher_diag is not None:
        ewc_per_param = compute_ewc_per_param(model)   # scalar (average)
        
        # Nếu có plasticity weights: modulate EWC bằng (1 - w)
        if model.plasticity_weights is not None:
            # w trung bình của batch — nếu full graph thì mean(w)
            w_mean = model.plasticity_weights.mean()
            ewc_loss = (1.0 - w_mean) * ewc_per_param
        else:
            ewc_loss = ewc_per_param
        
        total_loss = loss_pred + lambda_ewc * ewc_loss
    else:
        total_loss = loss_pred
    
    # Gán aux_loss để trainer log được
    model.aux_loss = total_loss - loss_pred
    
    total_loss.backward()
    
    # Optional: scale gradients theo per-node plasticity
    # (Nếu không muốn implement gradient hook, bỏ qua phần này)
    # scale_gradients_by_plasticity(model, batch, adj)
    
    optimizer.step()
    return total_loss.item(), loss_pred.item()
```

### 5.5 Fisher Information Matrix

Compute FIM sau khi train xong phase $t$, lưu vào `gdap_buffer["fisher"]`:

```python
def compute_fisher(model, data_loader, adj, n_samples=200):
    """
    Tính diagonal Fisher Information Matrix.
    Chỉ cần sample một phần nhỏ của train set (n_samples batches).
    """
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    
    model.eval()
    count = 0
    for batch in data_loader:
        if count >= n_samples:
            break
        model.zero_grad()
        pred = model(batch, adj)
        loss = mae_loss(pred, batch.y)
        loss.backward()
        
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.data ** 2
        count += 1
    
    for n in fisher:
        fisher[n] /= count
    
    return fisher


def compute_ewc_per_param(model):
    """
    EWC penalty: sum_i F_i * (theta_i - theta*_i)^2
    """
    loss = 0.0
    for n, p in model.named_parameters():
        if n in model.prev_params and n in model.fisher_diag_dict:
            diff = p - model.prev_params[n]
            loss += (model.fisher_diag_dict[n] * diff ** 2).sum()
    return loss
```

### 5.6 Tích hợp vào training loop (default_trainer.py)

Trong `train_model()` hoặc tương đương, thêm logic sau:

```python
# ── Trước epoch đầu tiên của phase t ──
if year > begin_year and method == "GDAP":
    curr_mean, curr_std = gdap_prepare(
        model, train_loader,
        prev_buffer  = gdap_buffer,
        adj_prev     = adj_prev,        # adj của phase t-1, lưu từ vòng lặp trước
        adj_curr     = args.adj.cpu().numpy(),
        beta         = args.gdap_beta,
        gamma        = args.gdap_gamma,
    )

# ── Training loop ──
for epoch in range(args.epochs):
    for batch in train_loader:
        if method == "GDAP":
            loss, pred_loss = gdap_step(model, batch, args.adj, optimizer,
                                        lambda_ewc=args.gdap_lambda_ewc)
        else:
            # ... training thông thường

# ── Sau khi train xong phase t ──
if method == "GDAP":
    # Tính Fisher cho phase tiếp theo
    fisher = compute_fisher(model, train_loader, args.adj)
    model.fisher_diag_dict = fisher
    model.prev_params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    
    # Lưu statistics vào buffer (không phải raw data)
    gdap_buffer = {
        "node_mean": curr_mean,
        "node_std":  curr_std,
        "adj":       args.adj.cpu().numpy(),
    }
    adj_prev = args.adj.cpu().numpy()
```

---

## 6. Config file (conf/PEMS/gdap.json)

```json
{
    "dataset": "PEMS",
    "method": "GDAP",
    "backbone_type": "stgnn",
    "begin_year": 2011,
    "end_year": 2017,
    "epochs": 50,
    "batch_size": 32,
    "lr": 0.001,
    "weight_decay": 1e-4,
    
    "gdap_beta": 1.0,
    "gdap_gamma": 0.5,
    "gdap_lambda_ewc": 0.1,
    "gdap_n_fisher_samples": 200
}
```

Tạo config tương tự cho `AIR` và `ENERGY-Wind`.

---

## 7. Ablation cần chạy

Để validate contribution, cần 4 ablation variant:

| Variant | Mô tả | Mục đích |
|---|---|---|
| `GDAP_full` | Full method: diffusion + signal + topology | Main result |
| `GDAP_no_diffuse` | Dùng raw $s_i$, không diffuse | Chứng minh diffusion quan trọng |
| `GDAP_no_topo` | Chỉ dùng signal shift ($\gamma=0$) | Chứng minh $\Delta A$ quan trọng |
| `GDAP_no_signal` | Chỉ dùng topology shift | Baseline topology-only |
| `EWC` | EWC thông thường (uniform weight) | So sánh với baseline |

---

## 8. Metrics và evaluation

Giữ nguyên convention hiện tại:

- Report MAE, RMSE, MAPE tại horizon 3, 6, 12, Avg
- So sánh với TrafficStream, EAC, EWC, (optional: SCAA, STLoRA)
- Dataset ưu tiên: PEMS (nhiều phase nhất, node tăng qua năm — showcase tốt nhất cho GDAP)

---

## 9. Những lưu ý implementation

**Node indexing**: Index cũ không thay đổi. Node mới có index từ `N_{t-1}` đến `N_t - 1`. Khi extend `plasticity_weights`, pad thêm với giá trị cao (sigmoid(3) ≈ 0.95) cho node mới.

**Data layout trong DataLoader**: `data.x` có shape `(bs*N, T)`, không phải `(bs, N, T)`. Dùng `data.batch` để reshape nếu cần. Khi tính per-node statistics qua loader, cẩn thận với bước reshape này.

**Normalization của drift score**: Signal shift và topology shift có đơn vị khác nhau. Normalize từng thành phần về [0, 1] trước khi cộng sẽ ổn định hơn là dùng raw values.

**EWC cho node mới**: Node mới không có entry trong `fisher_diag` từ phase trước → exclude khỏi EWC penalty. Chỉ tính EWC cho `N_prev` node đầu tiên.

**Memory**: Fisher matrix lưu một float per parameter → với ResNet-18 (~11M params) tốn ~44MB. Chấp nhận được. Nếu muốn tiết kiệm, chỉ tính FIM cho backbone layers (GCN1, GCN2), bỏ qua FC layer cuối.

**Gradient scaling (optional)**: Nếu muốn implement per-node gradient scaling thay vì chỉ EWC modulation, dùng `register_hook` trên node embeddings sau GCN layer đầu tiên. Tuy nhiên, chỉ EWC modulation đã đủ để test idea chính.

---

## 10. Expected behavior

- Với PEMS 2011→2017: năm 2013-2014 thường có shift lớn (sensor expansion). GDAP nên cho phép update mạnh hơn ở giai đoạn này so với EWC uniform.
- Node ở vùng ổn định (rural sensors) nên có `w ≈ 0.3-0.4`, node ở vùng urban đang phát triển nên có `w ≈ 0.7-0.9`.
- Nếu diffusion quan trọng, `GDAP_full` sẽ outperform `GDAP_no_diffuse` rõ rệt hơn trên các node không tự shift nhưng kề node shift.