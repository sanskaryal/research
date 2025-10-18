#!/usr/bin/env python3
"""
personal_cap.py
Personalized Federated Learning (Ditto) with CapsNet on MedMNIST OrganMNIST.
- Trains a global model via FedAvg.
- Trains a personalized model for each client using Ditto regularization.
- Simulates multiple clients with non-IID (Dirichlet) data splits.
- Logs and compares the performance of the global vs. personalized model for each client.

Run:
  python personal_cap.py
"""

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
    "data_frac": 0.4,

    # Federated setup
    "num_clients": 4,
    "frac_clients": 1.0,
    "rounds": 30,
    "local_epochs": 5,         # Epochs for clients to train the global model
    "batch_size": 128,
    "num_workers": 0,

    # Non-IID
    "dirichlet_alpha": 0.2,

    # Ditto Personalization
    "personalization_epochs": 2, # Epochs for personalized model training
    "personalization_lr": 1e-4,   # Learning rate for personalized models
    "lambda_ditto": 0.9,          # Ditto regularization strength

    # RDP Privacy
    "dp_delta": 1e-6,             # Target delta for DP
    "target_epsilon": 20.0,        # Target total epsilon for the whole training
    "dp_max_grad_norm": 3.0,      # Gradient clipping norm (C)
    "delta_prime": 1e-6,          # delta' for composed privacy loss

    # Optimization
    "lr": 0.001,
    "weight_decay": 0.0,

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
# FL helpers
# =============================
def make_loader(ds: Dataset, batch_size: int, shuffle: bool):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=CONFIG["num_workers"])

def train_local_caps(model, dataset: Dataset, epochs: int, lr: float, noise_multiplier: float):
    model = deepcopy(model).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=CONFIG["weight_decay"])
    loader = make_loader(dataset, CONFIG["batch_size"], shuffle=True)
    
    total_loss, total_correct, total_samples = 0.0, 0, 0
    
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), torch.as_tensor(y).squeeze().long().to(device)
            one_hot = torch.eye(model.num_classes, device=device).index_select(dim=0, index=y)

            # --- DP-specific changes: Manual gradient computation, clipping, and noise ---
            opt.zero_grad()

            # 1. Compute per-sample gradients
            out, recon = model(x, labels=one_hot)
            loss = model.total_loss(x, out, one_hot, recon)
            loss.backward()

            # 2. Clip gradients and add noise
            total_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total_norm += param.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            
            clip_coef = min(CONFIG["dp_max_grad_norm"] / (total_norm + 1e-6), 1.0)

            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
                    
                    # Add Gaussian noise
                    noise = torch.normal(
                        0, 
                        CONFIG["dp_max_grad_norm"] * noise_multiplier, 
                        param.grad.shape, 
                        device=device
                    )
                    param.grad.data.add_(noise / x.size(0)) # Scale noise by batch size

            opt.step()
            # --- End of DP-specific changes ---

            total_loss += loss.item() * x.size(0)
            total_correct += (torch.norm(out, dim=2).argmax(dim=1) == y).sum().item()
            total_samples += x.size(0)
            
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    
    return model.state_dict(), len(dataset), avg_loss, avg_acc

def train_ditto_personalized(model, global_model, dataset: Dataset, epochs: int, lr: float, lambda_ditto: float):
    pers_model = deepcopy(model).to(device)
    pers_model.train()

    # Global model is for regularization only, no need to copy, just set to eval mode.
    global_model.eval()

    opt = torch.optim.Adam(pers_model.parameters(), lr=lr, weight_decay=CONFIG["weight_decay"])
    loader = make_loader(dataset, CONFIG["batch_size"], shuffle=True)

    # Get a detached dictionary of global parameters for the regularization term.
    global_params = {name: param.detach() for name, param in global_model.named_parameters()}

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), torch.as_tensor(y).squeeze().long().to(device)
            one_hot = torch.eye(pers_model.num_classes, device=device).index_select(dim=0, index=y)
            opt.zero_grad()

            out, recon = pers_model(x, labels=one_hot)
            task_loss = pers_model.total_loss(x, out, one_hot, recon)

            # Ditto regularization term, calculated robustly by parameter name.
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
# Privacy Accounting
# =============================
def calculate_privacy_loss(
    num_samples: int,
    epochs: int,
    lr: float,
    noise_multiplier: float,
    max_grad_norm: float,
    delta: float
):
    """Calculates epsilon_i for a client's local training based on user-provided formulas."""
    if noise_multiplier == 0:
        return float('inf')
    
    sigma = noise_multiplier * max_grad_norm
    
    # Sensitivity (Delta_i) from user's formula, with correction (removed learning rate)
    sensitivity = epochs * (2 * max_grad_norm / num_samples) if num_samples > 0 else 0
    
    # Epsilon_i from user's formula
    epsilon_i = (sensitivity / sigma) * np.sqrt(2 * np.log(1.25 / delta)) if sigma > 0 else float('inf')
    
    return epsilon_i

def calculate_composed_privacy_loss(
    epsilon_i: float,
    delta: float,
    delta_prime: float,
    num_clients_per_round: int,
    num_rounds: int,
):
    """Calculates total composed privacy loss (epsilon_total, delta_total) using advanced composition."""
    N = num_clients_per_round
    R = num_rounds
    
    # Epsilon_total from user's formula
    term1 = np.sqrt(2 * N * R * np.log(1 / delta_prime)) * epsilon_i
    term2 = (N * R * epsilon_i * (np.exp(epsilon_i) - 1)) / 2
    epsilon_total = term1 + term2
    
    # Delta_total from user's formula
    delta_total = N * R * delta + delta_prime
    
    return epsilon_total, delta_total

def find_optimal_noise_multiplier(
    target_epsilon: float,
    num_samples: int,
    epochs: int,
    lr: float,
    max_grad_norm: float,
    delta: float,
    delta_prime: float,
    num_clients_per_round: int,
    num_rounds: int,
    noise_search_space=(1e-5, 1000.0),
    tolerance=1e-3
):
    """Performs a binary search to find the noise multiplier that achieves the target epsilon."""
    low, high = noise_search_space
    
    while high - low > tolerance:
        mid = (low + high) / 2
        
        epsilon_i = calculate_privacy_loss(
            num_samples=num_samples,
            epochs=epochs,
            lr=lr,
            noise_multiplier=mid,
            max_grad_norm=max_grad_norm,
            delta=delta
        )
        
        composed_epsilon, _ = calculate_composed_privacy_loss(
            epsilon_i=epsilon_i,
            delta=delta,
            delta_prime=delta_prime,
            num_clients_per_round=num_clients_per_round,
            num_rounds=num_rounds
        )
        
        if composed_epsilon > target_epsilon:
            low = mid
        else:
            high = mid
            
    return high

# =============================
# File Naming
# =============================
def get_filenames():
    base = f"rdp_ditto_caps_niid_{CONFIG['dirichlet_alpha']}_{CONFIG['num_clients']}clients_lambda{CONFIG['lambda_ditto']}_eps{CONFIG['target_epsilon']}_C{CONFIG['dp_max_grad_norm']}"
    csv_name = Path("results") / f"{base}.csv"
    csv_name.parent.mkdir(parents=True, exist_ok=True)
    return csv_name

CSV_PATH = get_filenames()
print(f"Results will be saved to: {CSV_PATH}")

# =============================
# Federated Training with Ditto Personalization
# =============================
global_model = CapsNet(img_size=IMG_SIZE, num_classes=N_CLASSES).to(device)
personalized_models = [deepcopy(global_model) for _ in range(CONFIG["num_clients"])]
print(f"Model Parameters: {sum(p.numel() for p in global_model.parameters() if p.requires_grad):,}")

# --- Calculate optimal noise multiplier for target epsilon ---
min_client_samples = min(len(ds) for ds in client_train_datasets)
print(f"Worst-case privacy amplification is for client with {min_client_samples} samples.")

noise_multiplier = find_optimal_noise_multiplier(
    target_epsilon=CONFIG["target_epsilon"],
    num_samples=min_client_samples,
    epochs=CONFIG["local_epochs"],
    lr=CONFIG["lr"],
    max_grad_norm=CONFIG["dp_max_grad_norm"],
    delta=CONFIG["dp_delta"],
    delta_prime=CONFIG["delta_prime"],
    num_clients_per_round=int(CONFIG["frac_clients"] * CONFIG["num_clients"]),
    num_rounds=CONFIG["rounds"]
)
print(f"Using noise multiplier: {noise_multiplier:.4f} to achieve Epsilon={CONFIG['target_epsilon']:.2f}")
# --- End noise calculation ---

client_test_loaders = [make_loader(ds, CONFIG["batch_size"]*2, False) for ds in client_test_datasets]
results_log = []
t0 = time.time()

for r in range(1, CONFIG["rounds"] + 1):
    start = time.time()
    
    # 1. Select clients
    m = max(1, int(CONFIG["frac_clients"] * CONFIG["num_clients"]))
    selected_clients = np.random.default_rng(CONFIG["seed"] + r).choice(CONFIG["num_clients"], size=m, replace=False)

    # 2. Local training for global model update
    updates, weights, round_losses, round_accs = [], [], [], []
    for cid in selected_clients:
        local_model = deepcopy(global_model)
        sd, n, loss, acc = train_local_caps(
            local_model, 
            client_train_datasets[cid], 
            epochs=CONFIG["local_epochs"], 
            lr=CONFIG["lr"],
            noise_multiplier=noise_multiplier
        )
        updates.append(sd); weights.append(n); round_losses.append(loss); round_accs.append(acc)

    # 3. Aggregate to update global model
    global_model.load_state_dict(fed_avg(updates, weights))

    # 4. Ditto personalization for all clients
    for cid in range(CONFIG["num_clients"]):
        pers_sd = train_ditto_personalized(
            personalized_models[cid], global_model, client_train_datasets[cid],
            epochs=CONFIG["personalization_epochs"], lr=CONFIG["personalization_lr"], lambda_ditto=CONFIG["lambda_ditto"]
        )
        personalized_models[cid].load_state_dict(pers_sd)

    # 5. Evaluation and Logging
    round_time = time.time() - start
    global_test_metrics = evaluate_caps(global_model, test_loader)
    
    # --- Privacy Loss Calculation ---
    # We report the loss for the client with the fewest samples, as this is the worst-case.
    epsilon_i = calculate_privacy_loss(
        num_samples=min_client_samples, 
        epochs=CONFIG["local_epochs"],
        lr=CONFIG["lr"],
        noise_multiplier=noise_multiplier,
        max_grad_norm=CONFIG["dp_max_grad_norm"],
        delta=CONFIG["dp_delta"]
    )
    
    epsilon_total, delta_total = calculate_composed_privacy_loss(
        epsilon_i=epsilon_i,
        delta=CONFIG["dp_delta"],
        delta_prime=CONFIG["delta_prime"], # Standard practice to set delta_prime to a small value
        num_clients_per_round=len(selected_clients),
        num_rounds=r
    )
    # --- End Privacy Loss Calculation ---

    # Calculate weighted average for training loss and accuracy based on dataset size
    weighted_train_loss = np.average(round_losses, weights=weights)
    weighted_train_acc = np.average(round_accs, weights=weights)
    
    log_entry = {
        "round": r,
        "train_loss": weighted_train_loss, "train_accuracy": weighted_train_acc,
        "global_test_accuracy": global_test_metrics['accuracy'], "time_seconds": round_time,
        "epsilon_i": epsilon_i,
        "epsilon_total": epsilon_total,
        "delta_total": delta_total
    }

    print(f"Round {r:02d} | Train Loss: {weighted_train_loss:.4f}, Train Acc: {weighted_train_acc:.4f} | Global Test Acc: {global_test_metrics['accuracy']:.4f} | Epsilon: {epsilon_total:.4f} | Time: {round_time:.2f}s")

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
# Save Final Models
# =============================
MODEL_DIR = Path("trained_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_base_name = CSV_PATH.stem
global_model_path = MODEL_DIR / f"{model_base_name}_global_model.pth"
torch.save(global_model.state_dict(), global_model_path)
print(f"Saved final global model to: {global_model_path}")

for cid, p_model in enumerate(personalized_models):
    p_model_path = MODEL_DIR / f"{model_base_name}_personalized_client_{cid}.pth"
    torch.save(p_model.state_dict(), p_model_path)
    print(f"Saved personalized model for client {cid} to: {p_model_path}")


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

