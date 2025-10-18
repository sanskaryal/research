import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from data_downloader import _choose_dataclass

# =============================
# Configuration
# =============================
CONFIG = {
    "view": "axial",
    "image_size": 28,
    "data_root": "data/",
    "batch_size": 128,
    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
device = torch.device(CONFIG["device"])
print(f"Using device: {device}")

# =============================
# CapsNet Model Definition (Copied from personal_cap_rdp.py)
# =============================
def squash(tensor, dim=-1, eps=1e-9):
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

# =============================
# Data Loading
# =============================
IMG_SIZE = CONFIG["image_size"]
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5]),
])
DATA_ROOT = Path(CONFIG["data_root"])
DATA_ROOT.mkdir(parents=True, exist_ok=True)
DataClass = _choose_dataclass(CONFIG["view"])
test_ds = DataClass(split='test',  transform=transform, download=True, root=str(DATA_ROOT))
test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
print(f"Loaded {len(test_ds)} test samples.")

# =============================
# FGSM Attack
# =============================
def fgsm_attack(model, images, labels, epsilon):
    images.requires_grad = True
    
    # Forward pass to get capsule outputs
    digs, _ = model(images)
    
    # Create one-hot labels for margin loss
    one_hot_labels = F.one_hot(labels, num_classes=model.num_classes).float()
    
    # Use the model's native margin loss for a more effective attack
    loss = model.margin_loss(digs, one_hot_labels)

    model.zero_grad()
    loss.backward()

    # Create perturbed image
    perturbed_image = images + epsilon * images.grad.sign()
    # Clamp to maintain original data range [-1, 1]
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    
    return perturbed_image

# =============================
# Evaluation
# =============================
def evaluate_robustness(model, loader, epsilon_values):
    model.eval()
    results = {}
    loss_fn = nn.CrossEntropyLoss() # Placeholder for attack, not evaluation

    for epsilon in epsilon_values:
        total_correct, total_samples, success_count, original_correct_count, correct_conf_sum = 0, 0, 0, 0, 0.0

        for images, labels in loader:
            images = images.to(device)
            labels = torch.as_tensor(labels).squeeze().long().to(device)

            # 1. Get original predictions on clean images
            with torch.no_grad():
                orig_digs, _ = model(images)
                orig_lengths = torch.norm(orig_digs, p=2, dim=2)
                orig_probs = F.softmax(orig_lengths, dim=1)
                orig_preds = orig_probs.argmax(dim=1)

            # 2. Generate adversarial images
            if epsilon == 0:
                perturbed_images = images
            else:
                # Note: fgsm_attack enables grad on images internally
                perturbed_images = fgsm_attack(model, images, labels, epsilon)

            # 3. Get predictions on adversarial images
            with torch.no_grad():
                adv_digs, _ = model(perturbed_images)
                adv_lengths = torch.norm(adv_digs, p=2, dim=2)
                adv_probs = F.softmax(adv_lengths, dim=1)
                adv_preds = adv_probs.argmax(dim=1)
                adv_conf = adv_probs.gather(1, adv_preds.unsqueeze(1)).squeeze()

            # 4. Calculate metrics for the batch
            total_samples += labels.size(0)
            total_correct += (adv_preds == labels).sum().item()
            
            orig_correct_mask = (orig_preds == labels)
            original_correct_count += orig_correct_mask.sum().item()
            
            attack_success_mask = orig_correct_mask & (adv_preds != labels)
            success_count += attack_success_mask.sum().item()
            
            correct_conf_sum += adv_conf[adv_preds == labels].sum().item()

        # 5. Calculate final metrics for this epsilon value
        accuracy = 100 * total_correct / total_samples if total_samples > 0 else 0
        attack_success_rate = 100 * success_count / original_correct_count if original_correct_count > 0 else 0
        avg_correct_conf = 100 * correct_conf_sum / total_correct if total_correct > 0 else 0
        
        results[epsilon] = {
            "accuracy": accuracy,
            "attack_success_rate": attack_success_rate,
            "correct_confidence": avg_correct_conf
        }
        
    return results

# =============================
# Main Execution
# =============================
def main():
    model_paths = [
        "/home/ap1284@DS.UAH.edu/Atit/CapsNet/research/trained_models/fed_caps_niid_0.1_4clients_frac0.25.pth",
        "/home/ap1284@DS.UAH.edu/Atit/CapsNet/research/trained_models/rdp_ditto_caps_niid_0.2_4clients_lambda0.9_eps20.0_C3.0_global_model.pth"
    ]
    
    epsilon_values = [0, 0.05, 0.1, 0.2]
    all_results = []

    for path_str in model_paths:
        model_path = Path(path_str)
        print(f"\n--- Evaluating Model: {model_path.name} ---")

        # Initialize and load model
        model = CapsNet(img_size=CONFIG["image_size"], num_classes=11).to(device)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"Error loading model: {e}")
            continue

        # Evaluate robustness
        metrics_by_epsilon = evaluate_robustness(model, test_loader, epsilon_values)
        
        for epsilon, metrics in metrics_by_epsilon.items():
            print(f"  Epsilon: {epsilon:<4} | "
                  f"Accuracy: {metrics['accuracy']:.2f}% | "
                  f"Attack Success: {metrics['attack_success_rate']:.2f}% | "
                  f"Correct Confidence: {metrics['correct_confidence']:.2f}%")
            
            all_results.append({
                "model": model_path.name,
                "epsilon": epsilon,
                "accuracy": metrics['accuracy'],
                "attack_success_rate": metrics['attack_success_rate'],
                "correct_confidence": metrics['correct_confidence']
            })
        
    # Save results to CSV
    df = pd.DataFrame(all_results)
    csv_path = Path("results") / "fgsm_attack_detailed_results.csv"
    csv_path.parent.mkdir(exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to {csv_path}")


if __name__ == "__main__":
    main()
