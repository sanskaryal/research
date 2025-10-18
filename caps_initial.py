#!/usr/bin/env python3
import random
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import medmnist
from torchvision import transforms
from data_downloader import _choose_dataclass

# -----------------------------
# Configuration
# -----------------------------
# Hardcoded parameters instead of argparse
CONFIG = {
    "view": "axial",          # 'axial', 'coronal', or 'sagittal'
    "epochs": 50,
    "batch_size": 64,
    "lr": 0.001,
    "seed": 42,
    "image_size": 28,
    "num_workers": 0,         # Set to 0 for simplicity
    "data_fraction": 0.1,
}

# -----------------------------
# Utils
# -----------------------------
def seed_everything(seed: int = 42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# Model: CapsNet
# -----------------------------
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
        u = torch.stack(u, dim=1)
        B, Ck, Co, H, W = u.shape
        u = u.view(B, self.num_routes, Ck)
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
        lengths = torch.norm(digit_caps_output, dim=2)
        if labels is None:
            _, max_idx = lengths.max(dim=1)
            labels = torch.eye(lengths.size(1), device=digit_caps_output.device)[max_idx]
        masked = (digit_caps_output * labels.unsqueeze(2)).reshape(digit_caps_output.size(0), -1)
        recon = self.reconstruction(masked)
        recon = recon.view(-1, 1, self.input_size, self.input_size)
        return recon


class CapsNet(nn.Module):
    """Complete CapsNet model."""
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

# -----------------------------
# Training and Evaluation
# -----------------------------
def train_one_epoch(model, loader, opt, device):
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.squeeze().long().to(device)
        one_hot = torch.eye(model.num_classes, device=device).index_select(dim=0, index=y)
        
        opt.zero_grad()
        out, recon = model(x, labels=one_hot)
        loss = model.total_loss(x, out, one_hot, recon)
        loss.backward()
        opt.step()
        
        total_loss += loss.item()
        
        # Calculate training accuracy
        preds = torch.norm(out, dim=2).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        
    return total_loss / len(loader), correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluates the model on a dataset."""
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.squeeze().long().to(device)
        
        out, _ = model(x)
        preds = torch.norm(out, dim=2).argmax(dim=1)
        
        correct += (preds == y).sum().item()
        total += y.size(0)
        
    return correct / total

# -----------------------------
# Main Execution
# -----------------------------
seed_everything(CONFIG["seed"])

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Setup Data
DATA_ROOT = "data/"
DataClass = _choose_dataclass(CONFIG["view"])
IMG_SIZE = CONFIG["image_size"]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    transforms.Normalize(mean=[.5], std=[.5]),
])

train_ds = DataClass(split='train', transform=transform, download=True, root=DATA_ROOT)
# Reduce the dataset size based on the data_fraction
if CONFIG["data_fraction"] < 1.0:
    num_train_samples = int(len(train_ds) * CONFIG["data_fraction"])
    train_ds, _ = torch.utils.data.random_split(train_ds, [num_train_samples, len(train_ds) - num_train_samples])

valid_ds = DataClass(split='val',   transform=transform, download=True, root=DATA_ROOT)
test_ds  = DataClass(split='test',  transform=transform, download=True, root=DATA_ROOT)

# Number of classes is fixed at 11 for OrganMNIST
N_CLASSES = 11

train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
valid_loader = DataLoader(valid_ds, batch_size=CONFIG["batch_size"] * 2, shuffle=False, num_workers=CONFIG["num_workers"])
test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"] * 2, shuffle=False, num_workers=CONFIG["num_workers"])

# Setup Model and Optimizer
model = CapsNet(img_size=IMG_SIZE, num_classes=N_CLASSES).to(device)
opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Setup CSV logging
log_file = f"training_log_frac{CONFIG['data_fraction']}.csv"
with open(log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "loss", "train_acc", "val_acc"])

# Training Loop
t0 = time.time()
best_val_acc = -1.0

for epoch in range(CONFIG["epochs"]):
    e0 = time.time()
    
    loss, train_acc = train_one_epoch(model, train_loader, opt, device)
    val_acc = evaluate(model, valid_loader, device)
    
    # Log to CSV
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, loss, train_acc, val_acc])

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # Optional: Save model here if needed
        # torch.save(model.state_dict(), "best_model.pth")
    
    print(
        f"Epoch {epoch:02d} | "
        f"Loss: {loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Time: {time.time() - e0:.2f}s"
    )

print(f"\nTraining finished in {time.time() - t0:.2f}s")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")

# Final evaluation on the test set
test_acc = evaluate(model, test_loader, device)
print(f"Final Test Accuracy: {test_acc:.4f}")