import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time, copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split

from quantum_circuit_simulator import quantum_circuit

###############################
#         DATA LOADING        #
###############################
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

def load_dataset(name):
    print(f"Loading dataset: {name}\n")
    if name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(root="FashionMNIST", train=True, download=True, transform=transform)
        test_dataset  = datasets.FashionMNIST(root="FashionMNIST", train=False, download=True, transform=transform)
    elif name == "MNIST":
        train_dataset = datasets.MNIST(root="MNIST", train=True, download=True, transform=transform)
        test_dataset  = datasets.MNIST(root="MNIST", train=False, download=True, transform=transform)
    return train_dataset, test_dataset

train_dataset, test_dataset = load_dataset("MNIST")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Reduce dataset size for faster runs
frac = 0.2
train_dataset, _ = random_split(train_dataset, [int(frac * len(train_dataset)),
                                               len(train_dataset) - int(frac * len(train_dataset))])
test_dataset,  _ = random_split(test_dataset,  [int(frac * len(test_dataset)),
                                               len(test_dataset)  - int(frac * len(test_dataset))])

###############################
#  CREATE MULTI-CLIENT SPLIT  #
###############################
def split_into_clients(dataset, num_clients=4):
    """
    Splits the dataset into 'num_clients' subsets.
    Returns a list of client datasets.
    """
    total_size = len(dataset)
    sizes = [total_size // num_clients]*num_clients
    # Adjust for any remainder
    for i in range(total_size % num_clients):
        sizes[i] += 1
    client_datasets = random_split(dataset, sizes)
    return client_datasets

client_datasets = split_into_clients(train_dataset, num_clients=4)
for i, cdata in enumerate(client_datasets):
    print(f"Client {i+1} dataset size: {len(cdata)}")

###############################
#       DP QNN MODEL          #
###############################
def clip_gradients(model, max_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def add_depolarizing_noise(probs, num_qubits, lambda_rate, num_layers):
    gamma = 1 - (1 - lambda_rate)**num_layers  # Effective noise strength
    probs = torch.clamp(probs, 0, 1)
    probs /= probs.sum(dim=0, keepdim=True)
    noise_prob = torch.ones_like(probs) / (2**num_qubits)
    noisy_probs = (1 - gamma) * probs + gamma * noise_prob
    noisy_probs = torch.clamp(noisy_probs, 0, 1)
    noisy_probs /= noisy_probs.sum(dim=0, keepdim=True)
    return noisy_probs


class QNN(nn.Module):
    def __init__(self, n, L, depol_noise=False, lambda_rate=0.0, shots=100, noise_factor=0.5):
        super().__init__()
        self.flatten = nn.Flatten()
        angles = torch.empty((L, n), dtype=torch.float64)
        nn.init.uniform_(angles, -0.01, 0.01)
        self.angles = nn.Parameter(angles)
        self.linear = nn.Linear(2**n, 10)

        self.depol_noise  = depol_noise
        self.lambda_rate  = lambda_rate
        self.n            = n
        self.L            = L
        self.shots        = shots
        self.noise_factor = noise_factor

    def forward(self, x):
        x = F.pad(x, (2, 2, 2, 2), "constant", 0)
        x = self.flatten(x)
        x /= torch.linalg.norm(x.clone(), ord=2, dim=1, keepdim=True)

        qc = quantum_circuit(num_qubits=self.n, state_vector=x.T)

        for l in range(self.L):
            qc.Ry_layer(self.angles[l].to(torch.cfloat))
            qc.cx_linear_layer()

            probs = torch.real(qc.probabilities())

            # Depolarizing noise
            if self.depol_noise:
                probs = add_depolarizing_noise(probs, self.n, self.lambda_rate, self.L)

            # Shot noise for DP
            if self.shots is not None and self.shots > 0:
                variance = probs * (1 - probs) / float(self.shots)
                std = torch.sqrt(variance + 1e-10)
                noise = torch.randn_like(probs) * std * self.noise_factor
                probs = probs + noise
                probs = torch.clamp(probs, 0.0, 1.0)
                probs = probs / probs.sum(dim=0, keepdim=True)

        x = self.linear(probs.T)
        return x

###############################
#    FEDERATED LEARNING       #
###############################
def performance_estimate(dataset, model, loss_fn, batch_size=64):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    total_loss, total_correct = 0.0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_correct += (pred.argmax(1) == y).sum().item()
            total_loss    += loss_fn(pred, y).item()
    accuracy = total_correct / len(dataset)
    avg_loss = total_loss / len(dataloader)
    return accuracy, avg_loss

def one_epoch(model, dataset, loss_fn, optimizer, batch_size=64, clip_value=1.0):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        out  = model(X)
        loss = loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()
        clip_gradients(model, clip_value)  # DP gradient clip
        optimizer.step()

def train_one_client(global_model, local_dataset, local_epochs=1, lr=1e-1, weight_decay=1e-10, batch_size=64, clip_value=1.0):
    """
    Train a copy of the global model on one client's data, then return:
    - Updated model state
    - Final train accuracy for this client
    """
    model = copy.deepcopy(global_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(local_epochs):
        one_epoch(model, local_dataset, loss_fn, optimizer, batch_size=batch_size, clip_value=clip_value)

    # Calculate train accuracy for this client
    train_acc, _ = performance_estimate(local_dataset, model, loss_fn, batch_size=batch_size)
    return model.state_dict(), train_acc

def fedavg(local_states):
    """
    Averages the parameter dicts from each local model to form a new global model state.
    """
    new_state = copy.deepcopy(local_states[0])
    for key in new_state.keys():
        for i in range(1, len(local_states)):
            new_state[key] += local_states[i][key]
        new_state[key] = new_state[key] / len(local_states)
    return new_state

def federated_training(client_datasets, test_dataset,
                       n=10, L=4, depol_noise=True, lambda_rate=0.2,
                       shots=10, noise_factor=2.0,
                       global_rounds=3, local_epochs=1, batch_size=64,
                       lr=1e-1, weight_decay=1e-10, clip_value=1.0):
    """
    Orchestrates Federated Learning across multiple clients using a QNN with DP noise.
    """
    # Initialize global model
    global_model = QNN(n=n, L=L,
                       depol_noise=depol_noise, lambda_rate=lambda_rate,
                       shots=shots, noise_factor=noise_factor).to(device)
    loss_fn = nn.CrossEntropyLoss()

    for round_idx in range(global_rounds):
        local_states = []
        train_accuracies = []

        # Train each client locally
        for client_idx, cdata in enumerate(client_datasets):
            local_state, train_acc = train_one_client(global_model, cdata,
                                                      local_epochs=local_epochs,
                                                      lr=lr, weight_decay=weight_decay,
                                                      batch_size=batch_size,
                                                      clip_value=clip_value)
            local_states.append(local_state)
            train_accuracies.append(train_acc)

        # FedAvg aggregation
        new_global_state = fedavg(local_states)
        global_model.load_state_dict(new_global_state)

        # Compute metrics
        avg_train_acc = np.mean(train_accuracies)
        acc_test, loss_test = performance_estimate(test_dataset, global_model, loss_fn, batch_size=batch_size)
        
        print(f"[Round {round_idx+1}/{global_rounds}] "
              f"Train Accuracy = {avg_train_acc:.4f} | "
              f"Test Accuracy = {acc_test:.4f} | Test Loss = {loss_test:.4f}")

    return global_model


###############################
#       RUN FEDERATED LE      #
###############################
n            = 10
L            = 4
global_rounds= 20
local_epochs = 5
batch_size   = 64
lr_          = 1e-1
weight_decay_= 1e-10
shots        = 50                           
noise_factor = 0.5
clip_value   = 1
delta        = 1e-2
lambda_depol = 0.2
client_data_size = 1800






print("\n====== Federated Learning with Quantum DP QNN ======\n")
global_model = federated_training(
    client_datasets   = client_datasets,
    test_dataset      = test_dataset,
    n                 = n,
    L                 = L,
    depol_noise       = True,
    lambda_rate       = lambda_depol,
    shots             = shots,
    noise_factor      = noise_factor,
    global_rounds     = global_rounds,
    local_epochs      = local_epochs,
    batch_size        = batch_size,
    lr                = lr_,
    weight_decay      = weight_decay_,
    clip_value        = clip_value
)



# Save the trained model with dynamic filename based on shots and lambda values
model_filename = f"QNN_shots{shots}_lambda{lambda_depol}.pt"
torch.save(global_model.state_dict(), model_filename)
print(f"Model saved as {model_filename}")





import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import glob
import re

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Placeholder for test_dataset and QNN (replace with your actual definitions)
# test_dataset = ... (e.g., MNIST dataset)
# class QNN(nn.Module): ... (your QNN model definition)

# Define common QNN parameters
qnn_params = {
    'n': 10,
    'L': 4,
    'depol_noise': True
}

# Find all .pt model files
model_files = glob.glob("QNN_shots*_lambda*_noise*.pt")

# Extract parameters from filename
def extract_params(filename):
    match = re.search(r"QNN_shots(\d+)_lambda([\d.]+)_noise([\d.]+).pt", filename)
    if match:
        shots = int(match.group(1))
        lambda_rate = float(match.group(2))
        noise_factor = float(match.group(3))
        return shots, lambda_rate, noise_factor
    return None, None, None

# Filter files based on parameters
def filter_files(shots_values=None, lambda_values=None, noise_factors=None):
    selected_files = []
    for file_path in model_files:
        filename = os.path.basename(file_path)
        shots, lambda_rate, noise_factor = extract_params(filename)
        if shots is None:
            continue
        shots_match = shots_values is None or shots in shots_values
        lambda_match = lambda_values is None or lambda_rate in lambda_values
        noise_match = noise_factors is None or noise_factor in noise_factors
        if shots_match and lambda_match and noise_match:
            selected_files.append(filename)
    return selected_files

# FGSM Attack Function (unchanged)
def fgsm_attack(model, images, labels, epsilon, loss_fn, max_grad_norm=0.1):
    images.requires_grad = True
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    model.zero_grad()
    loss.backward()
    images.grad = torch.clamp(images.grad, -max_grad_norm, max_grad_norm)
    perturbed_images = images + epsilon * images.grad.data.sign()
    perturbed_images = torch.clamp(perturbed_images, -1, 1).detach()
    return perturbed_images

# Evaluate Robustness Function (unchanged)
def evaluate_robustness(model, test_loader, epsilon_values, device):
    metrics = {'accuracy': [], 'success_rate': [], 'correct_confidence': []}
    loss_fn = nn.CrossEntropyLoss()
    for epsilon in epsilon_values:
        total = 0
        correct = 0
        success_count = 0
        original_correct = 0
        correct_conf = 0.0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                orig_outputs = model(images)
                orig_probs = F.softmax(orig_outputs, dim=1)
                orig_preds = orig_probs.argmax(dim=1)
            perturbed_data = fgsm_attack(model, images, labels, epsilon, loss_fn)
            with torch.no_grad():
                adv_outputs = model(perturbed_data)
                adv_probs = F.softmax(adv_outputs, dim=1)
                adv_preds = adv_probs.argmax(dim=1)
                adv_conf = adv_probs.gather(1, adv_preds.unsqueeze(1)).squeeze()
            batch_size = labels.size(0)
            total += batch_size
            correct += (adv_preds == labels).sum().item()
            orig_correct_mask = (orig_preds == labels)
            attack_success_mask = orig_correct_mask & (adv_preds != labels)
            success_count += attack_success_mask.sum().item()
            original_correct += orig_correct_mask.sum().item()
            correct_conf += adv_conf[adv_preds == labels].sum().item()
        accuracy = 100 * correct / total
        success_rate = 100 * success_count / original_correct if original_correct > 0 else 0
        avg_correct_conf = 100 * correct_conf / correct if correct > 0 else 0
        metrics['accuracy'].append(accuracy)
        metrics['success_rate'].append(success_rate)
        metrics['correct_confidence'].append(avg_correct_conf)
    return metrics

# Plot Metrics Function (unchanged)
def plot_robustness_metrics(all_metrics, labels, epsilon_values, constant_param='noise', save=False):
    if not all_metrics:
        print("No metrics to plot.")
        return
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'x']
    linestyles = ['-', '--', '-.', ':']
    style_counter = 0
    for metrics, label in zip(all_metrics, labels):
        marker = markers[style_counter % len(markers)]
        linestyle = linestyles[style_counter % len(linestyles)]
        axs[0].plot(epsilon_values, metrics['accuracy'], label=label, marker=marker, linestyle=linestyle, markersize=6)
        axs[1].plot(epsilon_values, metrics['success_rate'], label=label, marker=marker, linestyle=linestyle, markersize=6)
        axs[2].plot(epsilon_values, metrics['correct_confidence'], label=label, marker=marker, linestyle=linestyle, markersize=6)
        style_counter += 1
    axs[0].set_title('Classification Accuracy')
    axs[0].set_xlabel('Epsilon')
    axs[0].set_ylabel('Accuracy (%)')
    axs[0].grid(True)
    axs[0].legend()
    axs[1].set_title('Attack Success Rate')
    axs[1].set_xlabel('Epsilon')
    axs[1].set_ylabel('Success Rate (%)')
    axs[1].grid(True)
    axs[1].legend()
    axs[2].set_title('Confidence on Correct Predictions')
    axs[2].set_xlabel('Epsilon')
    axs[2].set_ylabel('Confidence (%)')
    axs[2].grid(True)
    axs[2].legend()
    fig.suptitle(f'Robustness Metrics ({constant_param.capitalize()} Constant)', fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig(f'robustness_metrics_{constant_param}_constant.png')
    plt.show()

# Main execution
shots_values = [60, 400, 1000]
lambda_values = [0.1, 0.15, 0.25, 0.3]
noise_factors = [0.5, 0.1]
epsilon_values = [0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35]

# Replace with your actual test dataset
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Keep lambda constant (lambda=0.15)
print("Evaluating models with lambda=0.15 constant:")
selected_files = filter_files(shots_values=shots_values, lambda_values=[0.1], noise_factors=[1])
all_metrics = []
labels = []
for file_name in selected_files:
    shots, lambda_rate, noise_factor = extract_params(file_name)
    # Include lambda_rate in the label even though it's constant
    label = f"Shots: {shots}, λ: {lambda_rate}, Noise: {noise_factor}"
    print(f"\nEvaluating {label}:")
    model_params = qnn_params.copy()
    model_params['shots'] = shots
    model_params['lambda_rate'] = lambda_rate
    model_params['noise_factor'] = noise_factor
    model = QNN(**model_params).to(device)
    model.load_state_dict(torch.load(file_name, map_location=device))
    model.eval()
    metrics = evaluate_robustness(model, test_loader, epsilon_values, device)
    all_metrics.append(metrics)
    labels.append(label)
plot_robustness_metrics(all_metrics, labels, epsilon_values, constant_param='lambda', save=True)
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import glob
import re
import pandas as pd

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Placeholder for test_dataset and QNN (replace with your actual definitions)
# test_dataset = ... (e.g., MNIST dataset)
# class QNN(nn.Module): ... (your QNN model definition)

# Define common QNN parameters
qnn_params = {
    'n': 10,
    'L': 4,
    'depol_noise': True
}

# Find all .pt model files
model_files = glob.glob("QNN_shots*_lambda*_noise*.pt")

# Extract parameters from filename
def extract_params(filename):
    match = re.search(r"QNN_shots(\d+)_lambda([\d.]+)_noise([\d.]+).pt", filename)
    if match:
        shots = int(match.group(1))
        lambda_rate = float(match.group(2))
        noise_factor = float(match.group(3))
        return shots, lambda_rate, noise_factor
    return None, None, None

# Filter files based on parameters
def filter_files(shots_values=None, lambda_values=None, noise_factors=None):
    selected_files = []
    for file_path in model_files:
        filename = os.path.basename(file_path)
        shots, lambda_rate, noise_factor = extract_params(filename)
        if shots is None:
            continue
        shots_match = shots_values is None or shots in shots_values
        lambda_match = lambda_values is None or lambda_rate in lambda_values
        noise_match = noise_factors is None or noise_factor in noise_factors
        if shots_match and lambda_match and noise_match:
            selected_files.append(filename)
    return selected_files

# FGSM Attack Function
def fgsm_attack(model, images, labels, epsilon, loss_fn, max_grad_norm=0.1):
    images.requires_grad = True
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    model.zero_grad()
    loss.backward()
    images.grad = torch.clamp(images.grad, -max_grad_norm, max_grad_norm)
    perturbed_images = images + epsilon * images.grad.data.sign()
    perturbed_images = torch.clamp(perturbed_images, -1, 1).detach()
    return perturbed_images

# Evaluate Robustness Function
def evaluate_robustness(model, test_loader, epsilon_values, device):
    metrics = {'accuracy': [], 'success_rate': [], 'correct_confidence': []}
    loss_fn = nn.CrossEntropyLoss()
    for epsilon in epsilon_values:
        total = 0
        correct = 0
        success_count = 0
        original_correct = 0
        correct_conf = 0.0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                orig_outputs = model(images)
                orig_probs = F.softmax(orig_outputs, dim=1)
                orig_preds = orig_probs.argmax(dim=1)
            perturbed_data = fgsm_attack(model, images, labels, epsilon, loss_fn)
            with torch.no_grad():
                adv_outputs = model(perturbed_data)
                adv_probs = F.softmax(adv_outputs, dim=1)
                adv_preds = adv_probs.argmax(dim=1)
                adv_conf = adv_probs.gather(1, adv_preds.unsqueeze(1)).squeeze()
            batch_size = labels.size(0)
            total += batch_size
            correct += (adv_preds == labels).sum().item()
            orig_correct_mask = (orig_preds == labels)
            attack_success_mask = orig_correct_mask & (adv_preds != labels)
            success_count += attack_success_mask.sum().item()
            original_correct += orig_correct_mask.sum().item()
            correct_conf += adv_conf[adv_preds == labels].sum().item()
        accuracy = 100 * correct / total
        success_rate = 100 * success_count / original_correct if original_correct > 0 else 0
        avg_correct_conf = 100 * correct_conf / correct if correct > 0 else 0
        metrics['accuracy'].append(accuracy)
        metrics['success_rate'].append(success_rate)
        metrics['correct_confidence'].append(avg_correct_conf)
    return metrics

# Plot Metrics Function
def plot_robustness_metrics(all_metrics, labels, epsilon_values, constant_param='noise', save=False):
    if not all_metrics:
        print("No metrics to plot.")
        return
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'x']
    linestyles = ['-', '--', '-.', ':']
    style_counter = 0
    for metrics, label in zip(all_metrics, labels):
        marker = markers[style_counter % len(markers)]
        linestyle = linestyles[style_counter % len(linestyles)]
        axs[0].plot(epsilon_values, metrics['accuracy'], label=label, marker=marker,
                    linestyle=linestyle, markersize=6)
        axs[1].plot(epsilon_values, metrics['success_rate'], label=label, marker=marker,
                    linestyle=linestyle, markersize=6)
        axs[2].plot(epsilon_values, metrics['correct_confidence'], label=label, marker=marker,
                    linestyle=linestyle, markersize=6)
        style_counter += 1

    axs[0].set_title('Classification Accuracy')
    axs[0].set_xlabel('Epsilon')
    axs[0].set_ylabel('Accuracy (%)')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].set_title('Attack Success Rate')
    axs[1].set_xlabel('Epsilon')
    axs[1].set_ylabel('Success Rate (%)')
    axs[1].grid(True)
    axs[1].legend()

    axs[2].set_title('Confidence on Correct Predictions')
    axs[2].set_xlabel('Epsilon')
    axs[2].set_ylabel('Confidence (%)')
    axs[2].grid(True)
    axs[2].legend()

    fig.suptitle(f'Robustness Metrics ({constant_param.capitalize()} Constant)', fontsize=16)
    plt.tight_layout()

    if save:
        plt.savefig(f'robustness_metrics_{constant_param}_constant.png')
    plt.show()

############################################################
#                   MAIN EXECUTION                         #
############################################################
shots_values = [60, 1000]
lambda_values = [0.1, 0.15, 0.25, 0.3]
noise_factors = [0.5, 0.1]
epsilon_values = [0.1, 0.15, 0.2, 0.25, 0.3]

# Replace with your actual test dataset
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Keep lambda constant (lambda=0.15)
print("Evaluating models with lambda=0.15 constant:")
selected_files = filter_files(
    shots_values=shots_values,
    lambda_values=[0.15, 0.1, 0.3],  # You can adjust this as needed
    noise_factors=[1.0, 0.5]
)

all_metrics = []
labels = []

# We'll store each row as a dict before converting to a DataFrame
csv_rows = []

for file_name in selected_files:
    shots, lambda_rate, noise_factor = extract_params(file_name)
    label = f"Shots: {shots}, λ: {lambda_rate}, Noise: {noise_factor}"
    print(f"\nEvaluating {label}:")
    model_params = qnn_params.copy()
    model_params['shots'] = shots
    model_params['lambda_rate'] = lambda_rate
    model_params['noise_factor'] = noise_factor
    model = QNN(**model_params).to(device)
    model.load_state_dict(torch.load(file_name, map_location=device))
    model.eval()

    metrics = evaluate_robustness(model, test_loader, epsilon_values, device)
    all_metrics.append(metrics)
    labels.append(label)

    # Gather results for CSV
    for i, eps in enumerate(epsilon_values):
        csv_rows.append({
            "Shots": shots,
            "Lambda": lambda_rate,
            "Noise": noise_factor,
            "Epsilon": eps,
            "Accuracy": metrics['accuracy'][i],
            "AttackSuccessRate": metrics['success_rate'][i],
            "CorrectConfidence": metrics['correct_confidence'][i]
        })

# Plot metrics
plot_robustness_metrics(all_metrics, labels, epsilon_values, constant_param='lambda', save=True)

# Save to CSV
df_csv = pd.DataFrame(csv_rows)
csv_filename = "robustness_metrics.csv"
df_csv.to_csv(csv_filename, index=False)
print(f"\nSaved CSV results to {csv_filename}")

import numpy as np
from scipy.stats import norm
import csv

# Function to calculate epsilon (unchanged from your provided code)
def calculate_epsilon_advanced(
    n, L, shots, lambda_rate, clip_value=1.0,
    delta=1e-5, noise_factor=1.0, T=1, local_epochs=1
):
    """
    Computes total epsilon using advanced composition for T rounds
    with correct variance and no over-reduction by client data size.
    """
    # 1) Effective noise variance for a SINGLE round
    c_gamma = 1 - (1 - lambda_rate)**L
    sigma_sq = (2**n / (2 * shots)) * c_gamma
    sigma = np.sqrt(sigma_sq) * noise_factor

    # 2) Single-round epsilon (Gaussian mechanism)
    Z = norm.ppf(1 - delta)
    epsilon_base = (clip_value / sigma) * Z

    # 3) Effective number of total steps
    T_effective = T

    # 4) Advanced Composition
    eps_AC = np.sqrt(2 * T_effective * np.log(1 / delta)) * epsilon_base \
             + T_effective * epsilon_base * (np.exp(epsilon_base) - 1)

    return round(eps_AC, 4)

# Fixed parameters from your setup
n = 8
L = 4
global_rounds = 50  # T
local_epochs = 3
clip_value = 0.6
delta = 10e-2  # 0.01

# Input lists for shots, lambda_rate, and noise_factor
shots_list = [30, 60, 150, 400, 1000]
lambda_list = [0.1, 0.15, 0.25, 0.3]
noise_list = [0.5, 1.0, 2.0]

# Generate all combinations and calculate epsilon
epsilon_data = []
for shots in shots_list:
    for lambda_rate in lambda_list:
        for noise_factor in noise_list:
            eps_est = calculate_epsilon_advanced(
                n=n,
                L=L,
                shots=shots,
                lambda_rate=lambda_rate,
                clip_value=clip_value,
                delta=delta,
                noise_factor=noise_factor,
                T=global_rounds,
                local_epochs=local_epochs
            )
            epsilon_data.append((shots, lambda_rate, noise_factor, eps_est))

# Sort by shots, lambda, noise for consistency
epsilon_data = sorted(epsilon_data, key=lambda x: (x[0], x[1], x[2]))

# Write to CSV
csv_filename = "epsilon_values.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write header
    writer.writerow(['Shots', 'Lambda', 'Noise', 'Epsilon'])
    # Write data rows
    for shots, lambda_rate, noise_factor, eps in epsilon_data:
        writer.writerow([shots, lambda_rate, noise_factor, eps])

print(f"Epsilon values have been saved to '{csv_filename}'")
print("\nSample of the data:")
print(f"{'Shots':<10} {'Lambda':<10} {'Noise':<10} {'Epsilon':<10}")
print("-" * 40)
for row in epsilon_data[:50]:  # Show first 5 rows as a sample
    print(f"{row[0]:<10} {row[1]:<10} {row[2]:<10} {row[3]:<10}")
if len(epsilon_data) > 50:
    print("...")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data
data = {
    'Shots': [30, 30, 30, 30, 60, 60, 60, 60, 150, 150, 150, 150, 400, 400, 400, 400, 1000, 1000, 1000, 1000],
    'Lambda': [0.1, 0.15, 0.25, 0.3, 0.1, 0.15, 0.25, 0.3, 0.1, 0.15, 0.25, 0.3, 0.1, 0.15, 0.25, 0.3, 0.1, 0.15, 0.25, 0.3],
    'Epsilon': [3.7727, 3.0124, 2.3785, 2.2214, 6.229, 4.8845, 3.7889, 3.5213, 
                 12.9427, 9.8528, 7.4209, 6.8406, 31.5407, 23.0593, 16.6835, 15.2075,
                 82.6142, 57.4623, 39.6018, 35.6167]
}

df = pd.DataFrame(data)
heatmap_data = df.pivot_table(values='Epsilon', index='Shots', columns='Lambda')

# Heatmap with distinct borders
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', cbar_kws={'label': 'Epsilon'}, linewidths=0.5, linecolor='black')
plt.gca().collections[0].colorbar.set_label('$\epsilon$', size=16)
plt.gca().collections[0].colorbar.ax.tick_params(labelsize=16)
plt.xlabel('Depolarization Factor', fontsize=16)
plt.ylabel('Shots', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('epsilon_heatmap.pdf')
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# Data
data = {
    'Shots': [30, 30, 30, 60, 60, 60, 150, 150, 150, 400, 400, 400, 1000, 1000, 1000],
    'Lambda': [0.1, 0.15, 0.25, 0.1, 0.15, 0.25, 0.1, 0.15, 0.25, 0.1, 0.15, 0.25, 0.1, 0.15, 0.25],
    'Epsilon': [3.7727, 3.0124, 2.3785, 6.229, 4.8845, 3.7889, 
                 12.9427, 9.8528, 7.4209, 31.5407, 23.0593, 16.6835,
                 82.6142, 57.4623, 39.6018]
}

df = pd.DataFrame(data)

# Line Plot for Epsilon vs Shots for Depolarizing Factors 0.1, 0.15, 0.25
plt.figure(figsize=(8, 6))
for lam in [0.1, 0.15, 0.25]:
    subset = df[df['Lambda'] == lam]
    plt.plot(subset['Shots'], subset['Epsilon'], marker='o', label=f"Depolarizing Factor ($\gamma$) = {lam}")

plt.xlabel('Shots ($M$)', fontsize=16)
plt.ylabel('$\epsilon$', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('epsilon_vs_shots_vs_depol.pdf')
plt.show()
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = {
    'Shots': [30, 30, 30, 30, 60, 60, 60, 60, 150, 150, 150, 150, 400, 400, 400, 400, 850, 850, 850, 850],
    'Lambda': [0.1, 0.15, 0.25, 0.3, 0.1, 0.15, 0.25, 0.3, 0.1, 0.15, 0.25, 0.3, 0.1, 0.15, 0.25, 0.3, 0.1, 0.15, 0.25, 0.3],
    'Epsilon': [6.0339, 5.2862, 4.8394, 4.776, 10.3774, 8.984, 8.161, 8.0448,
                 23.042, 19.5667, 17.5511, 17.269, 61.5465, 50.8749, 44.8434, 44.009,
                 142.364, 115.7109, 100.7043, 121.9797]
}

df = pd.DataFrame(data)

# Line Plot for Epsilon vs Shots for Depolarizing Factors 0.1, 0.15, 0.25
plt.figure(figsize=(10, 6))
for lam in [0.1, 0.15, 0.25]:
    subset = df[df['Lambda'] == lam]
    plt.plot(subset['Shots'], subset['Epsilon'], marker='o', label=f'Depolarizing Factor ($\gamma$) {lam}')

plt.xlabel('Shots ($M$)', fontsize=16)
plt.ylabel('$\epsilon$', fontsize=16)
plt.xticks(ticks=range(50, 850, 150), fontsize=16)
plt.yticks(fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('epsilon_vs_shots_vs_depol.pdf')
plt.show()


import numpy as np
import math
from scipy.stats import norm

def calculate_epsilon_qfl_single(
    n=10,               # number of qubits (D)
    L=4,                # number of PQC layers
    shots=100,          # number of measurement shots (M)
    lambda_rate=0.1,    # per-layer gate depolarizing rate (λ)
    clip_value=0.6,     # gradient clipping constant (C)
    delta=1e-2,         # DP delta
    noise_factor=1.0,   # optional extra scaling
    T=50,               # global rounds
    local_epochs=3,     # K local epochs
    learning_rate=0.01, # learning rate (η)
    dataset_size=1000,  # size of local dataset |D|
    num_clients=10,     # total number of clients (N)
    delta_prime=1e-6    # advanced composition δ′
):
    """
    calculate total ε for a single config using QFL differential privacy derivation.
    """
    # 1) depolarizing noise aggregation across L layers
    p = 1.0 - (1.0 - lambda_rate) ** L

    # 2) combined variance due to shot + depolarizing noise
    # c(p) = 1 - (1 - p)^2 (see QFL DP paper)
    c_gamma = 1.0 - (1.0 - p) ** 2
    sigma_sq = (2 ** n) / (2 * shots) * c_gamma
    sigma = np.sqrt(sigma_sq) * noise_factor

    # 3) sensitivity from QFL DP paper: Δ = η⋅K⋅(2C / |D|)
    sensitivity = local_epochs * (2.0 * clip_value / dataset_size)

    # 4) single-round epsilon εₙₜ from Gaussian mechanism:
    factor = np.sqrt(2.0 * np.log(1.25 / delta))
    epsilon_single = (sensitivity / sigma) * factor

    # 5) advanced composition over N clients and T rounds:
    term1 = np.sqrt(2 * num_clients * T * np.log(1.0 / delta_prime)) * epsilon_single
    term2 = (num_clients * T * epsilon_single * (math.exp(epsilon_single) - 1.0)) / 2.0
    epsilon_total = term1 + term2

    return round(epsilon_total, 6)

# ----- test with fixed parameters -----
eps_test = calculate_epsilon_qfl_single(
    n=8,
    L=5,
    shots=60,
    lambda_rate=0.05,
    clip_value=1,
    delta=1e-6,
    noise_factor=1.0,
    T=50,
    local_epochs=9,
    learning_rate=0.01,
    dataset_size=1500,
    num_clients=10
)

print(f"[🚀] QFL Differential Privacy ε (total): {eps_test}")


import numpy as np
import csv

def calculate_epsilon_new(n, L, shots, lambda_rate, eta, K, B, D_n_t, N, T, C, delta):
    """
    Computes total epsilon using the document's formulation for differential privacy
    in quantum federated learning with advanced composition over N clients and T rounds.
    """
    # Compute depolarizing noise probability p
    p = 1 - (1 - lambda_rate) ** L
    c_p = 1 - (1 - p) ** 2  # c(p) = 1 - (1 - p)^2
    
    # Variance of total noise per component
    Var_xi_total_d = eta ** 2 * K * (B ** 2 * 2 ** n * c_p) / (2 * shots * D_n_t)
    sigma = np.sqrt(Var_xi_total_d)
    
    # Sensitivity
    Delta_n_t = eta * K * (2 * C / D_n_t)
    
    # Per-client per-round epsilon
    epsilon_n_t = (Delta_n_t / sigma) * np.sqrt(2 * np.log(1.25 / delta))
    
    # Total epsilon via advanced composition for N*T mechanisms
    delta_prime = delta  # Simplification; could adjust based on delta_total
    epsilon_total = (
        np.sqrt(2 * N * T * np.log(1 / delta_prime)) * epsilon_n_t
        + (N * T * epsilon_n_t * (np.exp(epsilon_n_t) - 1)) / 2
    )
    
    return round(epsilon_total, 4)

# Fixed parameters
n = 10              # Number of qubits
L = 4               # Number of layers
global_rounds = 50  # T: Number of global rounds
local_epochs = 3   # K: Number of local epochs
clip_value = 0.6    # C: Clipping bound
delta = 0.0001        # Delta for DP
eta = 0.01           # Learning rate (example)
B = 1.0             # Bound on partial derivative (example)
D_n_t = 3500         # Local dataset size (example)
N = 10              # Number of clients (example)

# Input lists for varying parameters
shots_list = [30, 60, 150, 400,600, 800, 1000, 2000, 5000, 10000]
lambda_list = [ 0.03, 0.05, 0.1]

# Generate all combinations and calculate epsilon
epsilon_data = []
for shots in shots_list:
    for lambda_rate in lambda_list:
        eps_est = calculate_epsilon_new(
            n=n,
            L=L,
            shots=shots,
            lambda_rate=lambda_rate,
            eta=eta,
            K=local_epochs,
            B=B,
            D_n_t=D_n_t,
            N=N,
            T=global_rounds,
            C=clip_value,
            delta=delta
        )
        epsilon_data.append((shots, lambda_rate, eps_est))

# Sort by shots and lambda_rate
epsilon_data = sorted(epsilon_data, key=lambda x: (x[0], x[1]))

# Write to CSV
csv_filename = "epsilon_values_new.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Shots', 'Lambda', 'Epsilon'])
    for shots, lambda_rate, eps in epsilon_data:
        writer.writerow([shots, lambda_rate, eps])

# Print confirmation and sample data
print(f"Epsilon values have been saved to '{csv_filename}'")
print("\nSample of the data:")
print(f"{'Shots':<10} {'Lambda':<10} {'Epsilon':<10}")
print("-" * 30)
for row in epsilon_data[:10]:
    print(f"{row[0]:<10} {row[1]:<10} {row[2]:<10}")
if len(epsilon_data) > 10:
    print("...")
import matplotlib.pyplot as plt
import pandas as pd

# Final updated dataset
data = {
    'Shots': [
        30, 30, 30,
        60, 60, 60,
        150, 150, 150,
        400, 400, 400,
        600, 600, 600,
        800, 800, 800,
        1000, 1000, 1000
    ],
    'Lambda': [
        0.03, 0.05, 0.1,
        0.03, 0.05, 0.1,
        0.03, 0.05, 0.1,
        0.03, 0.05, 0.1,
        0.03, 0.05, 0.1,
        0.03, 0.05, 0.1,
        0.03, 0.05, 0.1
    ],
    'Epsilon': [
        9.2649, 7.157, 5.3113,
        14.1196, 10.7626, 7.8835,
        25.6768, 19.1095, 13.6686,
        52.2356, 37.5258, 25.8968,
        71.9702, 50.7774, 34.3989,
        91.3681, 63.554, 42.4295,
        110.7244, 76.108, 50.1924
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(8, 5))
for lam in sorted(df['Lambda'].unique()):
    subset = df[df['Lambda'] == lam]
    plt.plot(subset['Shots'], subset['Epsilon'], marker='o', label=f'$\lambda$= {lam}')

plt.xlabel('Shots ($M$)', fontsize=18)
plt.ylabel('$\epsilon_{total}$', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.savefig('epsilon_vs_shots_final.pdf')
plt.show()
