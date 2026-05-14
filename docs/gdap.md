# GDAP: Graph-Diffused Adaptive Plasticity

## 1. Bối cảnh bài toán

Trong bài toán dự báo chuỗi thời gian trên đồ thị (spatio-temporal graph forecasting) theo kiểu **continual learning**, dữ liệu đến theo từng phase (năm). Mỗi phase có:

- Đồ thị $G_t = (V_t, E_t)$ với tập nút $V_t$ có thể **tăng** qua các năm
- Tín hiệu $X_t \in \mathbb{R}^{N_t \times L}$ tại các nút
- Adjacency matrix $A_t$ thay đổi theo năm

**Ràng buộc cứng**: Không được truy cập raw data của các phase cũ khi huấn luyện phase mới.

---

## 2. Vấn đề với các phương pháp hiện tại

### 2.1 Catastrophic Forgetting

Khi fine-tune mô hình trên dữ liệu năm mới, trọng số mạng bị kéo xa khỏi các giá trị tốt cho năm cũ → mô hình "quên" kiến thức đã học. Đây là vấn đề cổ điển trong continual learning.

### 2.2 Uniform Plasticity — Thiếu sót cốt lõi

Các phương pháp hiện tại như EWC, TrafficStream fine-tune, EAC frozen backbone đều áp dụng **cùng mức độ học** (plasticity) cho toàn bộ đồ thị. Điều này bỏ qua hai thực tế quan trọng:

**Thực tế 1 — Asymmetric shift**:
Không phải mọi nút đều thay đổi như nhau. Trong PEMS:
- Cảm biến tại khu vực xây dựng mới → traffic tăng đột ngột → cần học nhiều (plasticity cao)
- Cảm biến tại đường cao tốc ổn định → traffic ít thay đổi → không cần học nhiều (plasticity thấp)

Ép tất cả nút có cùng learning dynamics là **suboptimal**: nút ổn định bị quên kiến thức cũ không đáng, nút mới bị underfit vì bị regularize quá mức.

**Thực tế 2 — Drift propagation**:
Trong GNN, message passing làm cho drift tại nút $A$ **lan sang nút hàng xóm** $B$. Nếu $A$ shift mạnh, $B$ cũng cần được cho phép update (ngay cả khi signal của $B$ tự nó ổn định), vì $B$ đang nhận thông tin từ một hàng xóm đã thay đổi.

---

## 3. Ý tưởng GDAP

GDAP giải quyết cả hai vấn đề trên bằng cách tính **per-node plasticity weight** $w_i \in (0, 1)$ và dùng nó để điều chỉnh gradient + EWC penalty **khác nhau cho từng nút** trong suốt quá trình training.

Trực giác:
- $w_i \approx 1$ → nút này đang shift mạnh → cho phép học tự do, giảm EWC penalty
- $w_i \approx 0$ → nút này ổn định → giữ gìn kiến thức cũ, tăng EWC penalty

---

## 4. Cơ chế tính Plasticity Weight

### Bước 1: Raw drift score

Trước khi train phase $t$, với mỗi nút $i$:

$$s_i = \underbrace{\frac{|\bar{X}_t^{(i)} - \bar{X}_{t-1}^{(i)}|}{\sigma_{t-1}^{(i)} + \varepsilon}}_{\text{signal shift}} + \gamma \cdot \underbrace{\sum_j |A_t[i,j] - A_{t-1}[i,j]|}_{\text{topology shift}}$$

- **Signal shift**: mức độ thay đổi activity trung bình của nút, chuẩn hóa theo std năm trước
- **Topology shift**: chuẩn L1 của hàng $i$ trong $\Delta A = A_t - A_{t-1}$ — nút có nhiều cạnh thay đổi thì cần học nhiều
- $\gamma$: trọng số để cân bằng hai thành phần (default 0.5)

**Nút mới** ($i \geq N_{t-1}$): gán $s_i = \max_j s_j$ — nút hoàn toàn mới cần plasticity tối đa.

### Bước 2: Graph diffusion

$$\tilde{s} = (I + \beta A_t) \cdot s$$

- Mỗi nút nhận thêm thông tin drift từ hàng xóm qua $A_t$
- $\beta$ kiểm soát mức độ lan truyền: $\beta = 0$ → không diffuse, $\beta = 1$ → 1-hop neighbourhood averaging
- **Mục đích**: nút $B$ ổn định nhưng kề nút $A$ shift mạnh → $\tilde{s}_B > s_B$ → $B$ được phép update để thích nghi với thay đổi trong neighbourhood

### Bước 3: Sigmoid → plasticity weight

$$w_i = \sigma(\tilde{s}_i) = \frac{1}{1 + e^{-\tilde{s}_i}}$$

$w_i \in (0, 1)$ với nút shift mạnh có $w_i \to 1$, nút ổn định có $w_i \to 0$.

---

## 5. Cơ chế chống quên thảm khốc

GDAP kết hợp **EWC có trọng số thích nghi** (plasticity-modulated EWC) để chống catastrophic forgetting.

### 5.1 Fisher Information Matrix (FIM)

Sau khi train xong phase $t$, tính diagonal FIM:

$$F_i = \mathbb{E}\left[\left(\frac{\partial \mathcal{L}}{\partial \theta_i}\right)^2\right]$$

$F_i$ đo mức độ "quan trọng" của tham số $\theta_i$ đối với performance hiện tại. Tham số quan trọng → gradient lớn → $F_i$ cao.

FIM được lưu vào buffer (không phải raw data → không vi phạm constraint), dùng cho phase tiếp theo.

### 5.2 EWC Loss truyền thống (so sánh)

EWC chuẩn thêm penalty đồng đều:

$$\mathcal{L}_{\text{EWC}} = \sum_i F_i \cdot (\theta_i - \theta_i^*)^2$$

Penalty này **kéo mọi tham số** về giá trị cũ $\theta_i^*$, không quan tâm nút nào đang shift.

### 5.3 GDAP: Plasticity-Modulated EWC

GDAP **scale EWC penalty ngược với plasticity**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pred}} + \lambda_{\text{ewc}} \cdot \underbrace{(1 - \bar{w})}_{\text{stability weight}} \cdot \mathcal{L}_{\text{EWC}}$$

Trong đó $\bar{w} = \frac{1}{N}\sum_i w_i$ là plasticity trung bình của phase hiện tại.

**Cơ chế hoạt động**:

| Trường hợp | $\bar{w}$ | $(1-\bar{w})$ | EWC penalty | Hiệu ứng |
|---|---|---|---|---|
| Năm có nhiều node shift mạnh | $\approx 0.8$ | $\approx 0.2$ | Yếu | Model được học tự do hơn |
| Năm ổn định, ít thay đổi | $\approx 0.3$ | $\approx 0.7$ | Mạnh | Kiến thức cũ được bảo vệ |

**Tại sao hiệu quả hơn EWC đồng đều**:
- Khi graph shift nhiều: EWC chuẩn cản trở việc học → underfitting năm mới; GDAP giảm penalty → học được patterns mới
- Khi graph ổn định: EWC chuẩn lãng phí regularization trên cả graph; GDAP tăng penalty → bảo vệ tốt hơn

### 5.4 Buffer lưu giữa các phase

Để hoạt động mà không truy cập raw data cũ, GDAP lưu:

```
gdap_buffer = {
    "node_mean":   [N] — mean per node của train set
    "node_std":    [N] — std per node của train set
    "adj":         [N×N] — adjacency matrix A_t
    "fisher":      {param_name: tensor} — diagonal FIM
    "prev_params": {param_name: tensor} — snapshot params sau phase t
}
```

Đây là **statistics**, không phải raw samples → không vi phạm ràng buộc no-replay.

---

## 6. Luồng hoạt động theo từng phase

```
Phase t-1 kết thúc:
    └─ compute_gdap_fisher(best_model, train_data_t-1, adj_t-1)
    └─ lưu gdap_buffer = {node_mean, node_std, adj, fisher, prev_params}

Phase t bắt đầu:
    ├─ [Pre-training]
    │   ├─ compute_node_stats(train_x_t) → curr_mean, curr_std
    │   ├─ signal_shift = |curr_mean - prev_mean| / prev_std
    │   ├─ topo_shift = ||A_t[i,:] - A_{t-1}[i,:]||_1
    │   ├─ s = signal_shift + γ·topo_shift  (normalize each to [0,1])
    │   ├─ s̃ = (I + β·A_t) @ s  [graph diffusion]
    │   └─ w = sigmoid(s̃)  → model.plasticity_weights
    │
    ├─ [Training loop — mỗi batch]
    │   ├─ loss_pred = MSELoss(pred, target)
    │   ├─ ewc_penalty = Σ_i F_i·(θ_i - θ*_i)²
    │   └─ loss_total = loss_pred + λ·(1 - mean(w))·ewc_penalty
    │
    └─ [Post-training]
        └─ cập nhật gdap_buffer cho phase t+1
```

---

## 7. Ablation variants

| Variant | Config | Mô tả |
|---|---|---|
| `GDAP_full` | `gdap_beta=1.0, gdap_gamma=0.5` | Full method |
| `GDAP_no_diffuse` | `gdap_beta=0.0` | Không graph diffusion |
| `GDAP_no_topo` | `gdap_gamma=0.0` | Chỉ dùng signal shift |
| `GDAP_no_signal` | chỉ dùng topo_shift | Chỉ dùng topology shift |

Nếu **diffusion quan trọng**: `GDAP_full` sẽ outperform `GDAP_no_diffuse` rõ rệt trên các nút không tự shift nhưng kề nút shift.

---

## 8. Cách chạy

```bash
# PEMS (2011–2017, 7 phases)
python main.py --conf conf/PEMS/gdap.json --gpuid 0 --seed 42 \
               --logname "gdap_st" --backbone_type "stgnn"

# AIR (2016–2019, 4 phases)
python main.py --conf conf/AIR/gdap.json --gpuid 0 --seed 42 \
               --logname "gdap_air" --backbone_type "stgnn"

# ENERGY-Wind (phase 0–3)
python main.py --conf conf/ENERGY-Wind/gdap.json --gpuid 0 --seed 42 \
               --logname "gdap_wind" --backbone_type "stgnn"
```

---

## 9. Hyperparameters

| Param | Default | Ý nghĩa |
|---|---|---|
| `gdap_beta` | 1.0 | Mức độ graph diffusion. $\beta=0$ = không diffuse |
| `gdap_gamma` | 0.5 | Trọng số topology shift so với signal shift |
| `gdap_lambda_ewc` | 0.1 | Cường độ EWC regularization |
| `gdap_n_fisher_samples` | 200 | Số batch để ước lượng FIM (trade-off accuracy vs speed) |

---

## 10. So sánh với các phương pháp liên quan

| Phương pháp | Chống quên | Phân biệt nút | Drift propagation |
|---|---|---|---|
| Fine-tune (TrafficStream) | Không | Không | Không |
| EWC | Có (uniform) | Không | Không |
| **GDAP** | **Có (adaptive)** | **Có** | **Có** |
| EAC | Frozen backbone | Có (per-node prompt) | Không |

---

## 11. Điểm khác biệt cốt lõi

GDAP không thay đổi kiến trúc mô hình. Nó là một **training-time wrapper**: cùng backbone (STGNN/DCRNN/ASTGNN/TGCN), nhưng quá trình tối ưu được điều chỉnh thích nghi theo **tình trạng drift của từng nút** trong đồ thị. Đây là điểm khác biệt so với tất cả các baseline hiện có trong codebase.
