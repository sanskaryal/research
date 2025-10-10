#!/usr/bin/env python3
"""
federated_cnn.py
Federated Learning (FedAvg) with CNN on MedMNIST OrganMNIST.
- Uses simple CNN architecture with cross-entropy loss.
- Simulates multiple clients locally.
- Supports IID and non-IID (Dirichlet) client splits.

Run:
  python federated_cnn.py

Adjust CONFIG at the top as needed.
"""

import random
import time
from pathlib import Path
from copy import deepcopy
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

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
    "data_frac": 0.3,          # Fraction of training data to use (1.0 = all)

    # Federated setup
    "num_clients": 10,          # fewer clients
    "frac_clients": 1,
    "rounds": 50,              # fewer rounds to test quickly
    "local_epochs": 5,         # 1 local epoch per rounsd at first
    "batch_size": 64,          # smaller batches help CPU
    "num_workers": 0,

    # IID vs non-IID
    "iid": False,               # set False + alpha below for non-IID
    "dirichlet_alpha": 0.1,

    # Optimization
    "lr": 0.01,               # SGD learning rate as specified
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
# CNN Model
# =============================
class CNN(nn.Module):
    """Simple CNN model for OrganMNIST classification."""
    def __init__(self, num_classes=11):
        super().__init__()
        self.num_classes = num_classes
        
        # Conv1: 32 filters, kernel 5×5, stride 1, padding 2
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28→14
        
        # Conv2: 64 filters, kernel 5×5, stride 1, padding 2  
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14→7
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 500)  # 7x7 after two pooling operations
        self.fc2 = nn.Linear(500, num_classes)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        # Conv1 + ReLU + MaxPool
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Conv2 + ReLU + MaxPool
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC1 + ReLU
        x = F.relu(self.fc1(x))
        
        # FC2 (output logits)
        x = self.fc2(x)
        
        return x

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

# Subset the training data if data_frac < 1.0
if CONFIG["data_frac"] < 1.0:
    num_samples = int(len(train_full) * CONFIG["data_frac"])
    subset_indices = np.random.default_rng(CONFIG["seed"]).choice(
        len(train_full), num_samples, replace=False
    )
    train_full = Subset(train_full, subset_indices)
    print(f"Using {num_samples} ({CONFIG['data_frac']:.0%}) of the training data.")


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
# Log Client Data Distribution
# =============================
def log_client_distributions(client_datasets, num_classes):
    """Prints the class distribution for each client."""
    print("\n" + "="*50)
    print("Client Data Distributions")
    print("="*50)
    for i, ds in enumerate(client_datasets):
        labels = _collect_labels(ds)
        if len(labels) == 0:
            print(f"  Client {i:02d}: 0 samples")
            continue
        
        counts = np.bincount(labels, minlength=num_classes)
        dist_str = ", ".join([f"{c:3d}" for c in counts])
        print(f"  Client {i:02d}: {len(labels):>4d} samples | [{dist_str}]")
    print("="*50 + "\n")

log_client_distributions(client_datasets, N_CLASSES)


# =============================
# FL helpers (CNN-aware)
# =============================
def make_loader(ds: Dataset, batch_size: int, shuffle: bool):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=CONFIG["num_workers"])

def train_local_cnn(model, dataset: Dataset, epochs: int, lr: float):
    model = deepcopy(model).to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=CONFIG["weight_decay"])

    loader = make_loader(dataset, CONFIG["batch_size"], shuffle=True)
    
    total_loss, total_correct, total_samples = 0.0, 0, 0
    
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = torch.as_tensor(y).squeeze().long().to(device)

            opt.zero_grad()
            logits = model(x)
            loss = model.criterion(logits, y)
            loss.backward()
            opt.step()
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    return model.state_dict(), len(dataset), avg_loss, avg_acc

@torch.no_grad()
def evaluate_cnn(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    
    for x, y in loader:
        x = x.to(device)
        labels = torch.as_tensor(y).squeeze().long().to(device)
        
        if labels.ndim == 0:
            all_labels.append(labels.cpu().item())
        else:
            all_labels.extend(labels.cpu().numpy())
        
        logits = model(x)
        loss = model.criterion(logits, labels)
        total_loss += loss.item() * x.size(0)

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())

    if not all_labels:
        return {"accuracy": 0, "loss": 0, "precision": 0, "recall": 0, "f1_score": 0}

    avg_loss = total_loss / len(all_labels)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

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
# File Naming
# =============================
def get_filenames():
    """Create descriptive filenames for outputs."""
    iid_str = "iid" if CONFIG["iid"] else f"niid_{CONFIG['dirichlet_alpha']}"
    base = f"fed_cnn_{iid_str}_{CONFIG['num_clients']}clients"
    
    if CONFIG["data_frac"] < 1.0:
        base += f"_frac{CONFIG['data_frac']}"
    
    csv_name = Path("results") / f"{base}.csv"
    model_name = Path("trained_models") / f"{base}.pth"
    
    csv_name.parent.mkdir(parents=True, exist_ok=True)
    model_name.parent.mkdir(parents=True, exist_ok=True)
    
    return csv_name, model_name

CSV_PATH, MODEL_PATH = get_filenames()
print(f"Results will be saved to: {CSV_PATH}")
print(f"Model will be saved to: {MODEL_PATH}")

# =============================
# Federated Training (FedAvg)
# =============================
global_model = CNN(num_classes=N_CLASSES).to(device)
print(f"Model Parameters: {sum(p.numel() for p in global_model.parameters() if p.requires_grad):,}")

rounds = CONFIG["rounds"]
frac = CONFIG["frac_clients"]
local_epochs = CONFIG["local_epochs"]

best_val = -1.0
t0 = time.time()
results_log = []

for r in range(1, rounds + 1):
    start = time.time()
    base_state = deepcopy(global_model.state_dict())

    # sample subset of clients
    m = max(1, int(frac * num_clients))
    selected = np.random.default_rng(CONFIG["seed"] + r).choice(num_clients, size=m, replace=False)

    # local training
    updates, weights = [], []
    round_train_losses, round_train_accs = [], []
    for cid in selected:
        client_model = deepcopy(global_model).to(device)
        client_model.load_state_dict(base_state)
        sd, n, loss, acc = train_local_cnn(client_model, client_datasets[cid], epochs=local_epochs, lr=CONFIG["lr"])
        updates.append(sd)
        weights.append(n)
        round_train_losses.append(loss)
        round_train_accs.append(acc)

    # aggregate
    new_state = fed_avg(updates, weights)
    global_model.load_state_dict(new_state)

    # evaluate on MedMNIST validation and test sets
    val_metrics = evaluate_cnn(global_model, valid_loader)
    test_metrics = evaluate_cnn(global_model, test_loader)

    if val_metrics["accuracy"] > best_val:
        best_val = val_metrics["accuracy"]
        torch.save(global_model.state_dict(), MODEL_PATH)

    round_time = time.time() - start
    results_log.append({
        "round": r,
        "train_loss": np.mean(round_train_losses),
        "train_accuracy": np.mean(round_train_accs),
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1_score": val_metrics["f1_score"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1_score": test_metrics["f1_score"],
        "time_seconds": round_time,
    })

    print(
        f"Round {r:02d} | "
        f"Train Loss: {np.mean(round_train_losses):.4f}, Train Acc: {np.mean(round_train_accs):.4f} | "
        f"Val Acc: {val_metrics['accuracy']:.4f} | "
        f"Test Acc: {test_metrics['accuracy']:.4f} | "
        f"Time: {round_time:.2f}s"
    )

df = pd.DataFrame(results_log)
df.to_csv(CSV_PATH, index=False)
print(f"\nTraining finished in {time.time() - t0:.2f}s")
print(f"Best Validation Accuracy: {best_val:.4f}")

# Final test on the best model
global_model.load_state_dict(torch.load(MODEL_PATH))
final_test_metrics = evaluate_cnn(global_model, test_loader)
print(f"\nFinal Test Metrics on Best Model: {final_test_metrics}")

# Append test metrics to results
final_test_metrics["round"] = "best_model_test"
df = pd.concat([df, pd.DataFrame([final_test_metrics])], ignore_index=True)
df.to_csv(CSV_PATH, index=False)