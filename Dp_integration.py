#!/usr/bin/env python3
"""
personal_cap.py
Personalized Federated Learning (Ditto) with CapsNet on MedMNIST OrganMNIST.
- Trains a global model via FedAvg (with optional DP-SGD).
- Trains a personalized model for each client using Ditto regularization.
- Simulates multiple clients with non-IID (Dirichlet) data splits.
- Logs and compares the performance of the global vs. personalized model for each client.

Run:
  python personal_cap.py
"""

import math
import random
import time
from pathlib import Path
from copy import deepcopy
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
import medmnist
from data_downloader import _choose_dataclass

# =============================
# Configuration
# =============================
CONFIG = {
    # Data
    "view": "axial",
    "image_size": 28,
    "data_root": "data/",
    "data_frac": 0.33,

    # Federated setup
    "num_clients": 4,
    "frac_clients": 1.0,
    "rounds": 30,
    "local_epochs": 5,         # Epochs for clients to train the global model
    "batch_size": 32,
    "num_workers": 0,

    # Non-IID
    "dirichlet_alpha": 0.2,

    # Ditto Personalization (kept non-DP to preserve local accuracy)
    "personalization_epochs": 2,
    "personalization_lr": 1e-4,
    "lambda_ditto": 0.9,

    # Optimization
    "lr": 0.001,
    "weight_decay": 0.0,

    # DP settings
    "dp_enabled": True,            # turn DP-SGD on/off for global model training
    "dp_clip_C": 1.0,              # per-step global grad clip norm
    "dp_noise_multiplier": None,   # if set (float), use this σ; else solve from target ε
    "dp_delta": 1e-3,
    "dp_delta_prime": 1e-4,
    "dp_target_epsilon": 2.0,      # set total ε target (None to disable autotune)
    # Note: ε accounting uses advanced composition approximation (per-round ε_i solved, then σ)
    #       Sensitivity Δ_i = η * E * 2C / n_i with batch-level clipping + σ noise.
    #       This is a practical, research-friendly bound (not tight moments accountant).
    # Misc
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

N_CLASSES = 11  # OrganMNIST has 11 classes

# =============================
# Reproducibility
# =============================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CONFIG["seed"])
device = torch.device(CONFIG["device"])
print(f"Using device: {device}")

# =============================
# CapsNet (INTACT)
# =============================
def squash(tensor, dim=-1, eps=1e-9):
    """Squashing activation function."""
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm + eps)

class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.features(x)

class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, stride=2, num_routes=32*6*6):
        super().__init__()
        self.num_routes = num_routes
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0)
            for _ in range(num_capsules)])
    def forward(self, x):
        u = torch.stack([capsule(x) for capsule in self.capsules], dim=1)
        u = u.view(x.size(0), self.num_routes, -1)
        return squash(u, dim=-1)

class DigitCaps(nn.Module):
    def __init__(self, num_capsules=11, num_routes=32*6*6, in_channels=8, out_channels=24, routing_iters=3):
        super().__init__()
        self.in_channels, self.num_routes, self.num_capsules = in_channels, num_routes, num_capsules
        self.routing_iters = routing_iters
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels) * 0.01)
    def forward(self, x):
        B = x.size(0)
        x = x[:, :, None, :, None]
        W = self.W.expand(B, -1, -1, -1, -1)
        u_hat = torch.matmul(W, x)
        b_ij = torch.zeros(B, self.num_routes, self.num_capsules, 1, 1, device=x.device)
        for i in range(self.routing_iters):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = squash(s_j, dim=3)
            if i < self.routing_iters - 1:
                a_ij = (u_hat * v_j).sum(dim=3, keepdim=True)
                b_ij = b_ij + a_ij
        return v_j.squeeze(1).squeeze(-1)

class Decoder(nn.Module):
    def __init__(self, input_size=28, num_capsules=11, dim_capsule=24):
        super().__init__()
        self.input_size = input_size
        in_features = num_capsules * dim_capsule
        self.reconstruction = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, input_size * input_size), nn.Sigmoid(),
        )
    def forward(self, digit_caps_output, labels=None):
        lengths = torch.norm(digit_caps_output, dim=2)
        if labels is None:
            _, max_idx = lengths.max(dim=1)
            labels = torch.eye(lengths.size(1), device=digit_caps_output.device)[max_idx]
        masked = (digit_caps_output * labels.unsqueeze(2)).view(digit_caps_output.size(0), -1)
        recon = self.reconstruction(masked)
        return recon.view(-1, 1, self.input_size, self.input_size)

class CapsNet(nn.Module):
    def __init__(self, img_size=28, num_classes=11):
        super().__init__()
        self.num_classes = num_classes
        self.conv = ConvLayer(in_channels=1, out_channels=256)
        self.primary = PrimaryCaps(num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, stride=2, num_routes=32*6*6)
        self.digits = DigitCaps(num_capsules=num_classes, num_routes=32*6*6, in_channels=8, out_channels=24, routing_iters=3)
        self.decoder = Decoder(input_size=img_size, num_capsules=num_classes, dim_capsule=24)
        self.mse = nn.MSELoss()
    def forward(self, x, labels=None):
        digs = self.digits(self.primary(self.conv(x)))
        recon = self.decoder(digs, labels)
        return digs, recon
    @staticmethod
    def margin_loss(digit_caps_output, one_hot_labels, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        v = torch.norm(digit_caps_output, dim=2)
        loss = one_hot_labels * F.relu(m_plus - v)**2 + lambda_ * (1.0 - one_hot_labels) * F.relu(v - m_minus)**2
        return loss.sum(dim=1).mean()
    def total_loss(self, data, digit_caps_output, one_hot_labels, recon):
        margin = self.margin_loss(digit_caps_output, one_hot_labels)
        recon_loss = self.mse(recon.view(recon.size(0), -1), data.view(recon.size(0), -1))
        return margin + 0.0005 * recon_loss

# =============================
# Data (MedMNIST OrganMNIST)
# =============================
IMG_SIZE = CONFIG["image_size"]
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5]),
])

DATA_ROOT = Path(CONFIG["data_root"])
DATA_ROOT.mkdir(parents=True, exist_ok=True)

DataClass = _choose_dataclass(CONFIG["view"])
train_full = DataClass(split='train', transform=transform, download=True, root=str(DATA_ROOT))
test_ds    = DataClass(split='test',  transform=transform, download=True, root=str(DATA_ROOT))

if CONFIG["data_frac"] < 1.0:
    num_samples = int(len(train_full) * CONFIG["data_frac"])
    subset_indices = np.random.default_rng(CONFIG["seed"]).choice(len(train_full), num_samples, replace=False)
    train_full = Subset(train_full, subset_indices)
    print(f"Using {num_samples} ({CONFIG['data_frac']:.0%}) of the training data.")

test_loader  = DataLoader(test_ds, batch_size=CONFIG["batch_size"]*2, shuffle=False, num_workers=CONFIG["num_workers"])

# =============================
# Federated partitioning (non-IID)
# =============================
def _collect_labels(dataset: Dataset):
    ys = [int(np.array(dataset[i][1]).squeeze()) for i in range(len(dataset))]
    return np.asarray(ys, dtype=int)

def dirichlet_non_iid_partition(dataset: Dataset, num_clients: int, alpha: float, seed: int = 0):
    targets = _collect_labels(dataset)
    num_classes = int(targets.max() + 1)
    idx_by_class = [np.where(targets == c)[0] for c in range(num_classes)]
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(num_clients)]
    for c_indices in idx_by_class:
        rng.shuffle(c_indices)
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        splits = (np.cumsum(proportions) * len(c_indices)).astype(int)
        splits[-1] = len(c_indices)
        for k, (start, end) in enumerate(zip([0, *splits[:-1]], splits)):
            client_indices[k].extend(c_indices[start:end])
    for k in range(num_clients):
        rng.shuffle(client_indices[k])
        client_indices[k] = list(map(int, client_indices[k]))
    return client_indices

print(f"Using non-IID partitioning (alpha={CONFIG['dirichlet_alpha']}).")
client_parts = dirichlet_non_iid_partition(train_full, CONFIG["num_clients"], alpha=CONFIG["dirichlet_alpha"], seed=CONFIG["seed"])

# Create local train/test splits for each client
client_train_datasets, client_test_datasets = [], []
for i in range(CONFIG["num_clients"]):
    client_indices = client_parts[i]
    train_size = int(0.8 * len(client_indices))
    train_indices, test_indices = client_indices[:train_size], client_indices[train_size:]
    client_train_datasets.append(Subset(train_full, train_indices))
    client_test_datasets.append(Subset(train_full, test_indices))

# =============================
# Log Client Data Distribution
# =============================
def log_client_distributions(client_train_datasets, num_classes):
    print("\n" + "="*50 + "\nClient Data Distributions\n" + "="*50)
    for i, ds in enumerate(client_train_datasets):
        labels = _collect_labels(ds)
        counts_str = "0 samples"
        if len(labels) > 0:
            counts = np.bincount(labels, minlength=num_classes)
            counts_str = f"{len(labels):>4d} samples | [{', '.join(f'{c:3d}' for c in counts)}]"
        print(f"  Client {i:02d}: {counts_str}")
    print("="*50 + "\n")

log_client_distributions(client_train_datasets, N_CLASSES)

# =============================
# DP utilities (advanced composition inversion for σ)
# =============================
def _solve_eps_i_from_total_eps(eps_total, N, R, delta_prime, guess=0.5, max_iter=50):
    """
    Invert: eps_total ≈ sqrt(2 N R ln(1/delta')) * eps_i + (N R * eps_i * (exp(eps_i) - 1))/2
    for eps_i using a damped Newton step. Returns a positive eps_i.
    """
    A = math.sqrt(2.0 * N * R * math.log(1.0 / delta_prime))
    B = 0.5 * N * R

    def f(e):  # lhs - target = 0
        return A * e + B * e * (math.exp(e) - 1.0) - eps_total

    def df(e):
        return A + B * ((math.exp(e) - 1.0) + e * math.exp(e))

    e = max(1e-6, guess)
    for _ in range(max_iter):
        val, g = f(e), df(e)
        step = val / max(g, 1e-12)
        e = max(1e-8, e - 0.5 * step)  # damped
        if abs(step) < 1e-6:
            break
    return e

def _sigma_for_target_total_epsilon(
    eps_total, delta, delta_prime, N, R, eta, E, C, n_i
):
    """
    Using Δ_i = eta * E * 2C / n_i and one-release bound:
      eps_i = (Δ_i / σ) * sqrt(2 ln(1.25/δ))
    combined with advanced composition to solve eps_i, then σ.
    """
    eps_i = _solve_eps_i_from_total_eps(eps_total, N, R, delta_prime)
    sens = eta * E * (2.0 * C) / max(1, n_i)
    denom = math.sqrt(2.0 * math.log(1.25 / delta))
    sigma = (sens / max(eps_i, 1e-8)) * denom
    return sigma, eps_i

# =============================
# FL helpers
# =============================
def make_loader(ds: Dataset, batch_size: int, shuffle: bool):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=CONFIG["num_workers"])

def train_local_caps(model, dataset: Dataset, epochs: int, lr: float, sigma_round: float = 0.0):
    """
    Global-model local training step. If sigma_round > 0 and CONFIG['dp_enabled'] is True,
    apply DP-SGD style clipping + Gaussian noise per step.
    """
    model = deepcopy(model).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=CONFIG["weight_decay"])
    loader = make_loader(dataset, CONFIG["batch_size"], shuffle=True)

    C = CONFIG.get("dp_clip_C", 1.0)
    use_dp = CONFIG.get("dp_enabled", False) and (sigma_round is not None) and (sigma_round > 0.0)

    total_loss, total_correct, total_samples = 0.0, 0, 0
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = torch.as_tensor(y).squeeze().long().to(device)

            one_hot = torch.eye(model.num_classes, device=device).index_select(dim=0, index=y)
            opt.zero_grad()
            out, recon = model(x, labels=one_hot)
            loss = model.total_loss(x, out, one_hot, recon)
            loss.backward()

            if use_dp:
                # Global gradient norm -> clip to C, then add Gaussian noise N(0, (σ*C)^2)
                with torch.no_grad():
                    total_norm_sq = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_norm_sq += (p.grad.detach() ** 2).sum()
                    total_norm = torch.sqrt(total_norm_sq + 1e-12)
                    clip_coef = float(min(1.0, C / total_norm.item()))
                    for p in model.parameters():
                        if p.grad is None: continue
                        p.grad.detach().mul_(clip_coef)
                        p.grad.add_(torch.normal(
                            mean=0.0,
                            std=sigma_round * C,
                            size=p.grad.shape,
                            device=p.grad.device
                        ))

            opt.step()

            total_loss += loss.item() * x.size(0)
            preds = torch.norm(out, dim=2).argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    return model.state_dict(), len(dataset), avg_loss, avg_acc

def train_ditto_personalized(model, global_model, dataset: Dataset, epochs: int, lr: float, lambda_ditto: float):
    """
    Ditto personalization step (kept non-DP by design to preserve local accuracy).
    """
    pers_model = deepcopy(model).to(device)
    pers_model.train()
    global_model.eval()

    opt = torch.optim.Adam(pers_model.parameters(), lr=lr, weight_decay=CONFIG["weight_decay"])
    loader = make_loader(dataset, CONFIG["batch_size"], shuffle=True)
    global_params = {name: param.detach() for name, param in global_model.named_parameters()}

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), torch.as_tensor(y).squeeze().long().to(device)
            one_hot = torch.eye(pers_model.num_classes, device=device).index_select(dim=0, index=y)
            opt.zero_grad()

            out, recon = pers_model(x, labels=one_hot)
            task_loss = pers_model.total_loss(x, out, one_hot, recon)

            reg_loss = 0.0
            for name, param in pers_model.named_parameters():
                if param.requires_grad:
                    reg_loss += torch.sum((param - global_params[name])**2)

            loss = task_loss + lambda_ditto * reg_loss
            loss.backward()
            opt.step()
    return pers_model.state_dict()

@torch.no_grad()
def evaluate_caps(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x, labels = x.to(device), torch.as_tensor(y).squeeze().long().to(device)
        if labels.ndim == 0: labels = labels.unsqueeze(0)
        all_labels.extend(labels.cpu().numpy())
        out, _ = model(x)
        all_preds.extend(torch.norm(out, dim=2).argmax(dim=1).cpu().numpy())
    if not all_labels: return {"accuracy": 0}
    return {"accuracy": np.mean(np.array(all_preds) == np.array(all_labels))}

def fed_avg(state_dicts, num_samples_list):
    total = float(sum(num_samples_list))
    agg = {k: torch.zeros_like(v) for k, v in state_dicts[0].items()}
    for sd, n in zip(state_dicts, num_samples_list):
        w = n / total
        for k in agg:
            if torch.is_floating_point(agg[k]):
                agg[k] += sd[k] * w
            else:
                agg[k] = sd[k]
    return agg

# =============================
# File Naming
# =============================
def get_filenames():
    dp_tag = "dpOFF"
    if CONFIG["dp_enabled"]:
        if CONFIG["dp_noise_multiplier"] is not None:
            dp_tag = f"dpSIG{CONFIG['dp_noise_multiplier']}"
        elif CONFIG["dp_target_epsilon"] is not None:
            dp_tag = f"dpEPS{CONFIG['dp_target_epsilon']}"
        else:
            dp_tag = "dpON"
    base = f"ditto_caps_niid_{CONFIG['dirichlet_alpha']}_{CONFIG['num_clients']}c_" \
           f"lambda{CONFIG['lambda_ditto']}_{dp_tag}"
    csv_name = Path("results") / f"{base}.csv"
    csv_name.parent.mkdir(parents=True, exist_ok=True)
    return csv_name

CSV_PATH = get_filenames()
print(f"Results will be saved to: {CSV_PATH}")

# =============================
# Federated Training with Ditto Personalization (+ DP for global)
# =============================
global_model = CapsNet(img_size=IMG_SIZE, num_classes=N_CLASSES).to(device)
personalized_models = [deepcopy(global_model) for _ in range(CONFIG["num_clients"])]
print(f"Model Parameters: {sum(p.numel() for p in global_model.parameters() if p.requires_grad):,}")

client_test_loaders = [make_loader(ds, CONFIG["batch_size"]*2, False) for ds in client_test_datasets]
results_log = []
t0 = time.time()

for r in range(1, CONFIG["rounds"] + 1):
    start = time.time()

    # 1) Client selection (all, since frac_clients=1.0 by default)
    m = max(1, int(CONFIG["frac_clients"] * CONFIG["num_clients"]))
    selected_clients = np.random.default_rng(CONFIG["seed"] + r).choice(CONFIG["num_clients"], size=m, replace=False)

    # --- Decide per-round σ (noise multiplier) ---
    if CONFIG["dp_enabled"]:
        avg_n = int(np.mean([len(ds) for ds in client_train_datasets])) if client_train_datasets else 1
        if CONFIG["dp_noise_multiplier"] is not None:
            sigma_round = float(CONFIG["dp_noise_multiplier"])
            eps_i_used = None
        elif CONFIG.get("dp_target_epsilon") is not None:
            sigma_round, eps_i_used = _sigma_for_target_total_epsilon(
                eps_total=CONFIG["dp_target_epsilon"],
                delta=CONFIG["dp_delta"],
                delta_prime=CONFIG["dp_delta_prime"],
                N=CONFIG["num_clients"],
                R=CONFIG["rounds"],
                eta=CONFIG["lr"],
                E=CONFIG["local_epochs"],
                C=CONFIG["dp_clip_C"],
                n_i=max(1, avg_n)
            )
        else:
            sigma_round, eps_i_used = 0.8, None
    else:
        sigma_round, eps_i_used = 0.0, None

    # 2) Local training for global model update (DP-SGD if enabled)
    updates, weights, round_losses, round_accs = [], [], [], []
    for cid in selected_clients:
        local_model = deepcopy(global_model)
        sd, n, loss, acc = train_local_caps(
            local_model, client_train_datasets[cid],
            epochs=CONFIG["local_epochs"], lr=CONFIG["lr"],
            sigma_round=sigma_round
        )
        updates.append(sd); weights.append(n); round_losses.append(loss); round_accs.append(acc)

    # 3) Aggregate to update global model
    global_model.load_state_dict(fed_avg(updates, weights))

    # 4) Ditto personalization for all clients (non-DP by design)
    for cid in range(CONFIG["num_clients"]):
        pers_sd = train_ditto_personalized(
            personalized_models[cid], global_model, client_train_datasets[cid],
            epochs=CONFIG["personalization_epochs"], lr=CONFIG["personalization_lr"], lambda_ditto=CONFIG["lambda_ditto"]
        )
        personalized_models[cid].load_state_dict(pers_sd)

    # 5) Evaluation and Logging
    round_time = time.time() - start
    global_test_metrics = evaluate_caps(global_model, test_loader)

    # Weighted average for training loss/acc based on dataset size
    weighted_train_loss = np.average(round_losses, weights=weights) if len(weights) else float('nan')
    weighted_train_acc = np.average(round_accs, weights=weights) if len(weights) else float('nan')

    # Simple DP logging (what σ used this round; optional eps_i_used)
    log_entry = {
        "round": r,
        "train_loss": weighted_train_loss, "train_accuracy": weighted_train_acc,
        "global_test_accuracy": global_test_metrics['accuracy'], "time_seconds": round_time,
        "dp_sigma_round": sigma_round, "dp_eps_i_used": (eps_i_used if eps_i_used is not None else "")
    }

    print(f"Round {r:02d} | Train Loss: {weighted_train_loss:.4f}, "
          f"Train Acc: {weighted_train_acc:.4f} | Global Test Acc: {global_test_metrics['accuracy']:.4f} "
          f"| σ: {sigma_round:.4f} | Time: {round_time:.2f}s")

    for cid in range(CONFIG["num_clients"]):
        loader = client_test_loaders[cid]
        pers_acc, global_acc = 0, 0
        if len(loader.dataset) > 0:
            pers_acc = evaluate_caps(personalized_models[cid], loader)['accuracy']
            global_acc = evaluate_caps(global_model, loader)['accuracy']
        log_entry[f"client_{cid}_pers_acc"] = pers_acc
        log_entry[f"client_{cid}_global_acc"] = global_acc
        print(f"  Client {cid:02d} | Personalized Acc: {pers_acc:.4f} | Global Acc (on local): {global_acc:.4f}")

    results_log.append(log_entry)
    pd.DataFrame(results_log).to_csv(CSV_PATH, index=False)

print(f"\nTraining finished in {time.time() - t0:.2f}s")
print(f"Final results saved to {CSV_PATH}")

# =============================
# Final Cross-Client Evaluation
# =============================
print(f"\n" + "="*50)
print("Final Cross-Client Evaluation")
print("="*50)
print("Evaluating each personalized model on its own data, global data, and another client's data.")

final_results = []
for cid in range(CONFIG["num_clients"]):
    pers_model = personalized_models[cid]

    # 1. Performance on its own local test data
    own_test_loader = client_test_loaders[cid]
    own_acc = evaluate_caps(pers_model, own_test_loader)['accuracy'] if len(own_test_loader.dataset) > 0 else 0

    # 2. Performance on the global test data
    global_test_acc = evaluate_caps(pers_model, test_loader)['accuracy']

    # 3. Performance on another client's test data (cross-client)
    cross_cid = (cid + 1) % CONFIG["num_clients"]
    cross_test_loader = client_test_loaders[cross_cid]
    cross_acc = evaluate_caps(pers_model, cross_test_loader)['accuracy'] if len(cross_test_loader.dataset) > 0 else 0

    final_results.append({
        "client_id": cid,
        "own_local_test_acc": own_acc,
        "global_test_acc": global_test_acc,
        f"cross_client_{cross_cid}_test_acc": cross_acc
    })

    print(f"\n--- Client {cid:02d} Personalized Model ---")
    print(f"  - Accuracy on own local test data: {own_acc:.4f}")
    print(f"  - Accuracy on global test data:    {global_test_acc:.4f}")
    print(f"  - Accuracy on client {cross_cid}'s data:     {cross_acc:.4f}")

# Save the final evaluation results to a new CSV
df_final = pd.DataFrame(final_results)
final_csv_path = Path("results") / f"final_evaluation_{Path(CSV_PATH).name}"
df_final.to_csv(final_csv_path, index=False)
print(f"\nFinal cross-client evaluation results saved to {final_csv_path}")
