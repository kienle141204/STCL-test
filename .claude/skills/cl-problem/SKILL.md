---
name: cl-problem
description: >
  Load complete context for the STRAP continual learning on spatio-temporal graphs problem.
  Use when brainstorming new methods, analyzing existing approaches, reviewing model code,
  or discussing research directions for this project.
---

# Continual Learning on Spatio-Temporal Graphs — Problem Context

## 1. Bài toán tổng quát

Bài toán dự báo chuỗi thời gian trên đồ thị (spatio-temporal graph forecasting) trong bối cảnh **continual learning**: dữ liệu đến theo từng phase (năm), và cả topology của đồ thị lẫn phân phối tín hiệu đều thay đổi qua từng phase.

**Formulation chính thức:**

Cho tập các phase tuần tự `t = 0, 1, 2, ..., T`. Tại mỗi phase t:
- Đồ thị `G_t = (V_t, E_t)` với tập nút `V_t` (có thể tăng: `|V_t| ≥ |V_{t-1}|`) và tập cạnh `E_t`
- Ma trận tín hiệu `X_t ∈ R^{N_t × L × F}` với `N_t = |V_t|`, `L` bước thời gian input, `F` features
- Nhãn `Y_t ∈ R^{N_t × H}` là H bước thời gian cần dự báo

**Mục tiêu:** Học một hàm `f_t: (X_t, A_t) → Ŷ_t` sao cho:
1. Dự báo chính xác tại phase hiện tại (plasticity)
2. Không quên kiến thức từ các phase trước (stability — tránh catastrophic forgetting)
3. **KHÔNG được truy cập dữ liệu raw của các phase cũ** khi huấn luyện phase mới

---

## 2. Cấu trúc dữ liệu trong codebase

### 2.1 Format file
```
data/<DATASET>/
├── RawData/
│   ├── <year>.npz          # signal: key "x", shape (T_samples, N_t)
│   └── ...
├── FastData/
│   ├── <year>.npz          # đã xử lý: train_x, train_y, val_x, val_y, test_x, test_y
│   └── ...
└── graph/
    ├── <year>_adj.npz      # adjacency matrix: key "x", shape (N_t, N_t)
    └── ...
```

### 2.2 Sliding window
- Input window: `x_len = 12` bước (5-phút intervals → 1 giờ cho traffic/air)
- Prediction horizon: `y_len = 12` bước tiếp theo
- Split: 60% train / 20% val / 20% test

### 2.3 Normalization
**Global Z-score normalization** được thực hiện 1 lần duy nhất khi load data (`utils/data_convert.py`):
```python
mean_train = np.mean(train_x)   # 1 scalar toàn bộ tập train
std_train  = np.std(train_x)    # 1 scalar
train_x    = (train_x - mean_train) / std_train
```
- Scalar duy nhất cho toàn bộ dataset (không phải per-node, không phải per-timestep)
- Sau normalization: per-node statistics vẫn khác nhau (node nào busy thì mean > 0)
- `data.x` và `data.y` trong DataLoader đều đã ở Z-scored space

### 2.4 Adjacency matrix
- Load từ file `.npz` mỗi năm (thay đổi theo năm)
- Normalized: `A = A / (sum(A, axis=1, keepdims=True) + 1e-6)` — row-normalized
- Chứa trong `args.adj` (torch.float tensor trên device)
- `N_t` có thể tăng qua các năm (nút mới thêm vào cuối, nút cũ giữ nguyên index)

---

## 3. Ba bộ dataset

### 3.1 PEMS (Traffic — California)
| Thuộc tính | Giá trị |
|---|---|
| Phase range | 2011 → 2017 (7 năm) |
| Tín hiệu | Lưu lượng giao thông (vehicle count/speed) tại cảm biến |
| Số nút ban đầu | ~500, tăng lên ~900+ vào 2017 |
| Đặc trưng shift | Volume tăng theo năm, cảm biến mới thêm, đường mới |
| Config | `conf/PEMS/` |

### 3.2 AIR (Air Quality — Beijing)
| Thuộc tính | Giá trị |
|---|---|
| Phase range | 2016 → 2019 (4 năm) |
| Tín hiệu | Chỉ số ô nhiễm không khí (PM2.5 và các chất khác) |
| Số nút | ~30, tương đối ổn định |
| Đặc trưng shift | Mức ô nhiễm thay đổi theo chính sách, mùa vụ, thời tiết |
| Config | `conf/AIR/` |

### 3.3 ENERGY-Wind (Wind Power)
| Thuộc tính | Giá trị |
|---|---|
| Phase range | 0 → 3 (4 phase) |
| Tín hiệu | Công suất phát điện gió tại từng turbine |
| Số nút | Tương đối ổn định |
| Đặc trưng shift | Công suất thay đổi khi turbine nâng cấp, thời tiết khác nhau theo năm |
| Config | `conf/ENERGY-Wind/` |

---

## 4. Kiến trúc mô hình baseline (STGNN)

Tất cả models trong codebase đều dùng chung backbone pattern:

```
Input x [bs, N, T=12]
    ↓
GCN1: [T → hidden=64] với adj A_t
    ↓ ReLU
TCN: Conv1d(1, 1, kernel=3) trên hidden dimension
    ↓
GCN2: [hidden=64 → out=12] với adj A_t
    ↓
Residual: + data.x [bs*N, 12]
    ↓
FC: Linear(12 → y_len=12) + GELU + Dropout
    ↓
Output [bs*N, 12]
```

**Các backbone khác:**
- `DCRNN_Backbone`: Diffusion Convolutional RNN
- `ASTGNN_Backbone`: Attention-based STGNN
- `TGCN_Backbone`: Temporal GCN

**Pattern bắt buộc cho mọi model mới:**
```python
class MyModel(nn.Module):
    def forward(self, data, adj) -> Tensor  # shape [bs*N, y_len]
    def feature(self, data, adj) -> Tensor  # intermediate features, dùng cho detection
```

---

## 5. Hai loại distribution shift

### 5.1 Spatial distribution shift
- **Biểu hiện:** Mean activity level của từng node thay đổi theo năm
  - Ví dụ: một đoạn đường năm 2015 có traffic thấp, năm 2017 tăng gấp đôi sau khi xây thêm khu dân cư
- **Nguyên nhân:** Phát triển đô thị, thay đổi hành vi, chính sách (air quality), nâng cấp thiết bị (wind)
- **Đặc điểm:** Mỗi node shift KHÁC NHAU và ở các mức độ KHÁC NHAU
- **Không phải** global shift (không thể sửa bằng 1 scalar)
- **Có thể** xuất hiện ở cả mean lẫn variance của node

### 5.2 Temporal distribution shift
- **Biểu hiện:** Pattern theo thời gian trong ngày/tuần thay đổi
  - Ví dụ: rush hour dịch từ 8am sang 9am, cuối tuần traffic pattern thay đổi
- **Nguyên nhân:** Thay đổi thói quen, COVID, remote work, thời tiết năm đó
- **Đặc điểm:** Ảnh hưởng nhiều node cùng lúc (spatial correlated), nhưng mức độ có thể khác nhau
- **Khó hơn** spatial shift vì pattern là relative (shape của time series), không chỉ mean/std

### 5.3 Structural shift (graph topology)
- **Biểu hiện:** A_t ≠ A_{t-1} — cạnh mới, cạnh mất, node mới
- **Đặc điểm:** Adjacency matrix được cung cấp sẵn mỗi năm — backbone nhận A_t trực tiếp
- **Node mới:** Index từ `N_{t-1}` đến `N_t - 1`, chưa từng được train
- **Backbone đã xử lý một phần** structural shift qua A_t; challenge là parameter space của backbone được train với A_1

---

## 6. Pipeline huấn luyện continual learning

```
for year in begin_year..end_year:
    1. Load adj A_t, data (train/val/test)
    2. Nếu year > begin_year và strategy == "incremental":
       - Phát hiện changed nodes (detection)
       - Tạo subgraph quanh changed nodes
       - Train trên subgraph
    3. Nếu strategy == "retrain":
       - Train trên full graph
    4. Validate, early stop (patience=5)
    5. Save best model → dùng làm init cho year tiếp theo
    6. Test, log metrics
```

**Hai strategy chính:**
- `incremental`: Chỉ train trên subgraph của changed/new nodes
- `retrain`: Train trên toàn bộ graph mỗi năm (backbone frozen → nhanh)

**Checkpoint convention:** `log/<DATASET>/<logname>-<seed>/<year>/<mae>.pkl`
- `load_best_model` pick file có tên thấp nhất (= lowest MAE)

---

## 7. Metrics đánh giá

Báo cáo tại **4 horizon**: 3, 6, 12, và Avg (trung bình 1→12).

Với mỗi horizon:
| Metric | Ý nghĩa |
|---|---|
| MAE | Mean Absolute Error (đơn vị gốc, sau denormalize) |
| RMSE | Root Mean Square Error |
| MAPE | Mean Absolute Percentage Error (%) |

**Cách đọc kết quả:** Số nhỏ hơn = tốt hơn cho cả 3 metrics.

**Context về performance** (PEMS dataset, STGNN backbone — approximate):
- TrafficStream (fine-tune, baseline): MAE avg ~17-18
- EAC (frozen backbone + input prompt): MAE avg ~15-16
- Frozen (không train gì): MAE avg ~20+
- Retrain từ đầu mỗi năm: tốt trên năm hiện tại nhưng kém năm cũ (forgetting)

---

## 8. Key constraints và gotchas

### 8.1 N thay đổi qua các năm
- `gamma`, `beta` hay bất kỳ per-node parameter nào phải được `expand` khi N tăng
- Node mới chưa có history → không thể dùng running stats từ năm trước
- Index cũ KHÔNG THAY ĐỔI (node 5 năm 2012 vẫn là node 5 năm 2017)

### 8.2 data.x layout trong DataLoader
- `data.x`: shape `[bs*N, T]` — batch flattened, N là slow dimension
- `data.batch`: mapping `[bs*N]` → batch index
- `to_dense_batch(pred, data.batch)` → `[bs, N, d]`

### 8.3 aux_loss pattern
Models muốn thêm regularization loss ngoài prediction loss:
```python
# Trong forward():
self.aux_loss = some_regularization_loss  # set trong forward()

# Trong trainer (default_trainer.py):
if getattr(model, 'aux_loss', None) is not None:
    loss = loss + model.aux_loss
```

### 8.4 Checkpoint và register_buffer(None)
- `register_buffer('name', None)` → key KHÔNG có trong state_dict
- `register_buffer('name', tensor)` → key CÓ trong state_dict
- Nếu model có buffer=None khi save, nhưng buffer=tensor khi load → mismatch
- Cần override `load_state_dict` để xử lý buffer tùy chọn

### 8.5 Evaluation trên full graph
- Trong training (incremental): có thể dùng subgraph
- Trong testing (`test_model`): LUÔN dùng `args.adj` (full graph), không phải `args.sub_adj`
- Model phải handle được cả hai size N

### 8.6 Per-node parameters và freeze
Khi `freeze_backbone()`:
- `nn.Parameter` có `requires_grad = False` → không update
- Phải explicitly set `requires_grad = True` cho adaptive params sau khi load checkpoint
- `filter(lambda p: p.requires_grad, model.parameters())` trong optimizer

---

## 9. Codebase: các file quan trọng

| File | Vai trò |
|---|---|
| `main.py` | Entry point, year loop, load adj, register methods dict |
| `src/model/model.py` | Toàn bộ model implementations (~2700 lines) |
| `src/trainer/default_trainer.py` | Training loop, validation, test, checkpoint |
| `utils/data_convert.py` | Z-score normalization, sliding window |
| `utils/metric.py` | MAE, RMSE, MAPE calculation |
| `conf/<DATASET>/<method>.json` | Hyperparameters per method per dataset |
| `results.csv` | Aggregated results sau mỗi run |

**Để thêm model mới cần:**
1. Thêm class vào `src/model/model.py` (implement `forward` và `feature`)
2. Import và đăng ký vào `vars(args)["methods"]` dict trong `main.py`
3. Thêm method-specific logic (freeze, expand) vào `default_trainer.py`
4. Tạo config JSON trong `conf/<DATASET>/`

---

## 10. Challenges chưa được giải quyết tốt

1. **Temporal pattern shape shift**: Thay đổi HÌNH DẠNG của time series (không phải chỉ mean/std). Khó hơn nhiều so với level shift.

2. **New node generalization**: Node mới hoàn toàn chưa thấy trong train → zero-shot prediction. Hiện tại các method hầu hết cần "warm up" qua training.

3. **Long-range dependency across years**: Backbone chỉ "nhớ" năm gần nhất. Kiến thức từ nhiều năm trước không được tận dụng tích lũy.

4. **Asymmetric shift**: Một số node shift nhiều (construction zone), một số không shift. Phương pháp lý tưởng phải phân biệt được.

5. **Temporal-spatial interaction shift**: Sự tương quan GIỮA các node thay đổi (không chỉ từng node độc lập). Ma trận A_t capture được một phần, nhưng learned correlation trong backbone thì không.

---

## 11. Lệnh chạy

```bash
# Train một method
python main.py --conf conf/<DATASET>/<method>.json --gpuid 0 --seed 42 \
               --logname "<method>_st" --backbone_type "stgnn"

# backbone_type options: stgnn | dcrnn | astgnn | tgcn
# gpuid -1 cho CPU

# Xem kết quả
python generate_results_excel.py --input results.csv --output results.xlsx
```

---

## 12. Existing methods summary (không phải hướng đi)

| Method | Key | Strategy |
|---|---|---|
| TrafficStream | Baseline fine-tune | incremental |
| EAC | Per-node input prompt (frozen backbone) | retrain |
| SCAA | SIS-gated LoRA adapter | retrain |
| AdaRev | Instance norm + affine (failed) | retrain |
| KPrompt | Attention routing prompts | incremental |
| LSPCL | Spectral contrastive learning | incremental |
| STLoRA | LoRA on backbone | retrain |
| RAP | Retrieval-augmented prediction | incremental |
| EWC | Elastic Weight Consolidation | any |
