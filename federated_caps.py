#!/usr/bin/env python3
"""
federated_caps.py
Federated Learning (FedAvg) with your original CapsNet on MedMNIST OrganMNIST.
- Keeps CapsNet intact (routing, squash, margin+reconstruction loss).
- Simulates multiple clients locally.
- Supports IID and non-IID (Dirichlet) client splits.

Run:
  python federated_caps.py

Adjust CONFIG at the top as needed.
"""

import random
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
import medmnist

# Your helper to pick OrganMNIST class by view: ('axial'|'coronal'|'sagittal')
from data_downloader import _choose_dataclass

# =============================
# Configuration
# =============================
CONFIG = {
    # Datas
    "view": "axial",
    "image_size": 28,
    "data_root": "data/",

    # Federated setup
    "num_clients": 10,          # fewer clients
    "frac_clients": 0.8,
    "rounds": 12,              # fewer rounds to test quickly
    "local_epochs": 1,         # 1 local epoch per rounsd at first
    "batch_size": 64,          # smaller batches help CPU
    "num_workers": 0,

    # IID vs non-IID
    "iid": False,               # set False + alpha below for non-IID
    "dirichlet_alpha": 0.5,

    # Optimization
    "lr": 2e-4,
    "weight_decay": 0.0,

    # CapsNet performance knobs (NEW)
    "routing_iters": 2,        # keep routing but fewer iters
    "primary_out_channels": 16,# halve maps -> halves routes
    "primary_stride": 3,       # increases downsampling -> fewer routes

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
    """Initial convolutional layer."""
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1)
    def forward(self, x):
        return F.relu(self.conv(x))

class PrimaryCaps(nn.Module):
    """Primary capsules layer."""
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, stride=2, num_routes=32*6*6):
        super().__init__()
        self.num_routes = num_routes
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0)
            for _ in range(num_capsules)
        ])
    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)          # B, Ck, Co, H, W
        B, Ck, Co, H, W = u.shape
        u = u.view(B, self.num_routes, Ck) # (B, routes, dim)
        return squash(u, dim=-1)

class DigitCaps(nn.Module):
    """Digit capsules layer with dynamic routing."""
    def __init__(self, num_capsules=11, num_routes=32*6*6, in_channels=8, out_channels=16, routing_iters=3):
        super().__init__()
        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules
        self.routing_iters = routing_iters
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels) * 0.01)
    def forward(self, x):
        B = x.size(0)
        x = x[:, :, None, :, None]          # (B, routes, 1, in_dim, 1)
        W = self.W.expand(B, -1, -1, -1, -1)
        u_hat = torch.matmul(W, x)          # (B, routes, caps, out_dim, 1)
        b_ij = torch.zeros(B, self.num_routes, self.num_capsules, 1, 1, device=x.device)
        for i in range(self.routing_iters):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)      # (B, 1, caps, out_dim, 1)
            v_j = squash(s_j, dim=3)                           # squash over out_dim
            if i < self.routing_iters - 1:
                a_ij = (u_hat * v_j).sum(dim=3, keepdim=True)  # agreement
                b_ij = b_ij + a_ij
        return v_j.squeeze(1).squeeze(-1)   # (B, caps, out_dim)

class Decoder(nn.Module):
    """Decoder for reconstruction."""
    def __init__(self, input_size=28, num_capsules=11, dim_capsule=16):
        super().__init__()
        self.input_size = input_size
        in_features = num_capsules * dim_capsule
        self.reconstruction = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, input_size * input_size), nn.Sigmoid(),
        )
    def forward(self, digit_caps_output, labels=None):
        lengths = torch.norm(digit_caps_output, dim=2)  # (B, caps)
        if labels is None:
            _, max_idx = lengths.max(dim=1)
            labels = torch.eye(lengths.size(1), device=digit_caps_output.device)[max_idx]
        masked = (digit_caps_output * labels.unsqueeze(2)).reshape(digit_caps_output.size(0), -1)
        recon = self.reconstruction(masked)
        recon = recon.view(-1, 1, self.input_size, self.input_size)
        return recon

class CapsNet(nn.Module):
    """Complete CapsNet model (unchanged)."""
    def __init__(self, img_size=28, num_classes=11):
        super().__init__()
        self.num_classes = num_classes
        self.conv = ConvLayer(in_channels=1, out_channels=256, kernel_size=9)
        self.primary = PrimaryCaps(num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, stride=2, num_routes=32*6*6)
        self.digits = DigitCaps(num_capsules=num_classes, num_routes=32*6*6, in_channels=8, out_channels=16, routing_iters=3)
        self.decoder = Decoder(input_size=img_size, num_capsules=num_classes, dim_capsule=16)
        self.mse = nn.MSELoss()
    def forward(self, x, labels=None):
        feats = self.conv(x)
        pri = self.primary(feats)
        digs = self.digits(pri)
        recon = self.decoder(digs, labels)
        return digs, recon
    @staticmethod
    def margin_loss(digit_caps_output, one_hot_labels, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        v = torch.norm(digit_caps_output, dim=2)
        left = F.relu(m_plus - v) ** 2
        right = F.relu(v - m_minus) ** 2
        loss = one_hot_labels * left + lambda_ * (1.0 - one_hot_labels) * right
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
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5]),
])

DATA_ROOT = Path(CONFIG["data_root"])
DATA_ROOT.mkdir(parents=True, exist_ok=True)

DataClass = _choose_dataclass(CONFIG["view"])
train_full = DataClass(split='train', transform=transform, download=True, root=str(DATA_ROOT))
valid_ds   = DataClass(split='val',   transform=transform, download=True, root=str(DATA_ROOT))
test_ds    = DataClass(split='test',  transform=transform, download=True, root=str(DATA_ROOT))

valid_loader = DataLoader(valid_ds, batch_size=CONFIG["batch_size"]*2, shuffle=False, num_workers=CONFIG["num_workers"])
test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"]*2, shuffle=False, num_workers=CONFIG["num_workers"])

# =============================
# Federated partitioning
# =============================
def iid_partition(dataset: Dataset, num_clients: int, seed: int = 0):
    N = len(dataset)
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    shards = np.array_split(idx, num_clients)
    return [list(map(int, s)) for s in shards]

def _collect_labels(dataset: Dataset):
    """Return a numpy array of class indices for dataset items."""
    ys = []
    for i in range(len(dataset)):
        y = dataset[i][1]
        # medmnist label often arrives as array([k]) -> make int
        if isinstance(y, (list, tuple, np.ndarray)):
            y = int(np.array(y).squeeze())
        else:
            y = int(y)
        ys.append(y)
    return np.asarray(ys, dtype=int)

def dirichlet_non_iid_partition(dataset: Dataset, num_clients: int, alpha: float, seed: int = 0):
    """
    Label-skew via Dirichlet allocation over classes.
    Smaller alpha => higher heterogeneity.
    """
    targets = _collect_labels(dataset)
    num_classes = int(targets.max() + 1)
    idx_by_class = [np.where(targets == c)[0] for c in range(num_classes)]
    rng = np.random.default_rng(seed)

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx_c = idx_by_class[c]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        splits = (np.cumsum(proportions) * len(idx_c)).astype(int)
        prev = 0
        for k, split in enumerate(splits):
            client_indices[k].extend(idx_c[prev:split])
            prev = split
    for k in range(num_clients):
        rng.shuffle(client_indices[k])
        client_indices[k] = list(map(int, client_indices[k]))
    return client_indices

num_clients = CONFIG["num_clients"]
if CONFIG["iid"]:
    client_parts = iid_partition(train_full, num_clients, seed=CONFIG["seed"])
else:
    client_parts = dirichlet_non_iid_partition(
        train_full, num_clients, alpha=CONFIG["dirichlet_alpha"], seed=CONFIG["seed"]
    )

client_datasets = [Subset(train_full, idxs) for idxs in client_parts]

# =============================
# FL helpers (CapsNet-aware)
# =============================
def make_loader(ds: Dataset, batch_size: int, shuffle: bool):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=CONFIG["num_workers"])

def train_local_caps(model, dataset: Dataset, epochs: int, lr: float):
    model = deepcopy(model).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=CONFIG["weight_decay"])

    loader = make_loader(dataset, CONFIG["batch_size"], shuffle=True)
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = torch.as_tensor(y).squeeze().long().to(device)

            one_hot = torch.eye(model.num_classes, device=device).index_select(dim=0, index=y)
            opt.zero_grad()
            out, recon = model(x, labels=one_hot)
            loss = model.total_loss(x, out, one_hot, recon)  # margin + 0.0005 * recon
            loss.backward()
            opt.step()

    return model.state_dict(), len(dataset)

@torch.no_grad()
def evaluate_caps(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device)
        y = torch.as_tensor(y).squeeze().long().to(device)
        out, _ = model(x)                       # no labels during eval
        preds = torch.norm(out, dim=2).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0

def fed_avg(state_dicts, num_samples_list):
    """Weighted average by client sample counts."""
    total = float(sum(num_samples_list))
    agg = {k: torch.zeros_like(v) for k, v in state_dicts[0].items()}
    for sd, n in zip(state_dicts, num_samples_list):
        w = n / total
        for k in agg.keys():
            agg[k] += sd[k] * w
    return agg

# =============================
# Federated Training (FedAvg)
# =============================
global_model = CapsNet(img_size=IMG_SIZE, num_classes=N_CLASSES).to(device)
print(f"Model Parameters: {sum(p.numel() for p in global_model.parameters() if p.requires_grad):,}")

rounds = CONFIG["rounds"]
frac = CONFIG["frac_clients"]
local_epochs = CONFIG["local_epochs"]

best_val = -1.0
t0 = time.time()

for r in range(1, rounds + 1):
    start = time.time()
    base_state = deepcopy(global_model.state_dict())

    # sample subset of clients
    m = max(1, int(frac * num_clients))
    selected = np.random.default_rng(CONFIG["seed"] + r).choice(num_clients, size=m, replace=False)

    # local training
    updates, weights = [], []
    for cid in selected:
        client_model = deepcopy(global_model).to(device)
        client_model.load_state_dict(base_state)
        sd, n = train_local_caps(client_model, client_datasets[cid], epochs=local_epochs, lr=CONFIG["lr"])
        updates.append(sd)
        weights.append(n)

    # aggregate
    new_state = fed_avg(updates, weights)
    global_model.load_state_dict(new_state)

    # evaluate on MedMNIST validation
    val_acc = evaluate_caps(global_model, valid_loader)

    if val_acc > best_val:
        best_val = val_acc
        # torch.save(global_model.state_dict(), "best_fed_caps.pth")

    print(
        f"Round {r:02d} | Clients: {m}/{num_clients} | "
        f"Val Acc: {val_acc:.4f} | Time: {time.time() - start:.2f}s"
    )

print(f"\nTraining finished in {time.time() - t0:.2f}s")
print(f"Best Validation Accuracy: {best_val:.4f}")

# Final test
test_acc = evaluate_caps(global_model, test_loader)
print(f"Final Test Accuracy: {test_acc:.4f}")