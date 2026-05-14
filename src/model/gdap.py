import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from dataer.SpatioTemporalDataset import SpatioTemporalDataset


def compute_node_stats(train_x_np):
    """
    Computes per-node mean and std from Z-score-normalized training data.

    Args:
        train_x_np: np.ndarray [n_samples, T, N]
    Returns:
        node_mean: np.ndarray [N]
        node_std:  np.ndarray [N]
    """
    node_mean = train_x_np.mean(axis=(0, 1))
    node_std  = train_x_np.std(axis=(0, 1))
    node_std  = np.where(node_std < 1e-6, 1e-6, node_std)
    return node_mean, node_std


def compute_plasticity_weights(adj_curr_np, adj_prev_np, curr_mean, prev_mean, prev_std,
                                N_prev, beta=1.0, gamma=0.5, eps=1e-6, device=None):
    """
    Computes per-node plasticity weights w ∈ (0, 1) via drift score + graph diffusion.

    Signal shift: how much node activity changed relative to its previous std.
    Topology shift: L1 norm of row-wise adjacency change.
    Both components normalized to [0, 1] before combining, then diffused over graph.

    New nodes (index >= N_prev) receive max plasticity automatically.

    Args:
        adj_curr_np: np.ndarray [N_curr, N_curr] row-normalized adj for current year
        adj_prev_np: np.ndarray [N_prev, N_prev] row-normalized adj for previous year
        curr_mean:   np.ndarray [N_curr] per-node mean of current year train data
        prev_mean:   np.ndarray [N_prev]
        prev_std:    np.ndarray [N_prev]
        N_prev:      int
        beta:        diffusion strength (0 = no diffuse, 1 = 1-hop neighbour averaging)
        gamma:       weight of topology shift vs signal shift
        eps:         numerical stability
        device:      torch.device or None
    Returns:
        w: torch.Tensor [N_curr] in (0, 1)
    """
    N_curr = adj_curr_np.shape[0]

    # --- Pad prev stats for new nodes ---
    if N_curr > N_prev:
        pad = N_curr - N_prev
        prev_mean = np.concatenate([prev_mean, np.zeros(pad)])
        prev_std  = np.concatenate([prev_std,  np.ones(pad)])

    # --- Signal shift (normalised per-node) ---
    signal_shift = np.abs(curr_mean - prev_mean) / (prev_std + eps)

    # New nodes get max signal shift → highest plasticity
    if N_curr > N_prev:
        max_existing = float(signal_shift[:N_prev].max()) if N_prev > 0 else 1.0
        signal_shift[N_prev:] = max_existing

    # Normalize each component to [0, 1]
    if signal_shift.max() > 0:
        signal_shift = signal_shift / signal_shift.max()

    # --- Topology shift ---
    if N_prev < N_curr:
        adj_prev_pad = np.zeros((N_curr, N_curr))
        adj_prev_pad[:N_prev, :N_prev] = adj_prev_np
    else:
        adj_prev_pad = adj_prev_np

    delta_A    = np.abs(adj_curr_np - adj_prev_pad)
    topo_shift = delta_A.sum(axis=1)
    if topo_shift.max() > 0:
        topo_shift = topo_shift / topo_shift.max()

    # --- Raw drift score ---
    s = signal_shift + gamma * topo_shift

    # Scale to [0, 3] so sigmoid covers a useful dynamic range
    if s.max() > 0:
        s = s / s.max() * 3.0

    # --- Graph diffusion: s̃ = (I + β·A) @ s ---
    I       = np.eye(N_curr)
    tilde_s = (I + beta * adj_curr_np) @ s

    # --- Plasticity weights ---
    w = 1.0 / (1.0 + np.exp(-tilde_s))
    w_tensor = torch.tensor(w, dtype=torch.float32)
    if device is not None:
        w_tensor = w_tensor.to(device)
    return w_tensor


def compute_gdap_fisher(model, train_x, train_y, adj, n_samples=200, batch_size=32, device=None):
    """
    Computes diagonal Fisher Information Matrix via squared gradients.

    Uses a subset of training data (n_samples batches) for efficiency.
    Only accumulates Fisher for parameters with requires_grad=True.

    Args:
        model:     GDAPModel (or any nn.Module)
        train_x:   np.ndarray [n_total, T, N]
        train_y:   np.ndarray [n_total, T_out, N]
        adj:       torch.Tensor [N, N] — full-graph adj
        n_samples: max batches to use
        batch_size: loader batch size
        device:    torch.device
    Returns:
        fisher: dict {param_name: Tensor}
    """
    fisher = {
        n: torch.zeros_like(p)
        for n, p in model.named_parameters()
        if p.requires_grad
    }

    loader = DataLoader(
        SpatioTemporalDataset("", "", x=train_x, y=train_y, edge_index="", mode="subgraph"),
        batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0,
    )

    model.eval()
    count = 0
    for data in loader:
        if count >= n_samples:
            break
        if device is not None:
            data = data.to(device, non_blocking=False)
        model.zero_grad()
        pred = model(data, adj)
        loss = F.mse_loss(data.y, pred, reduction="mean")
        loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.data ** 2
        count += 1

    if count > 0:
        for n in fisher:
            fisher[n] /= count

    return fisher


def gdap_ewc_loss(model, fisher_dict, prev_params):
    """
    EWC consolidation loss: Σ_i F_i · (θ_i − θ*_i)²

    Args:
        model:       nn.Module
        fisher_dict: dict {name: Tensor} — diagonal Fisher from previous phase
        prev_params: dict {name: Tensor} — parameter snapshot from previous phase
    Returns:
        scalar Tensor
    """
    dev  = next(model.parameters()).device
    loss = torch.zeros(1, device=dev)
    for n, p in model.named_parameters():
        if n in prev_params and n in fisher_dict:
            diff  = p - prev_params[n].to(dev)
            loss  = loss + (fisher_dict[n].to(dev) * diff ** 2).sum()
    return loss.squeeze()
