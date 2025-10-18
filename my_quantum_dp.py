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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

def load_dataset(name):
    if name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(root="FashionMNIST", train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root="FashionMNIST", train=False, download=True, transform=transform)
    elif name == "MNIST":
        train_dataset = datasets.MNIST(root="MNIST", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="MNIST", train=False, download=True, transform=transform)
    return train_dataset, test_dataset

train_dataset, test_dataset = load_dataset("MNIST")

device = "cuda" if torch.cuda.is_available() else "cpu"

fraction = 0.135
subset_count = int(len(train_dataset) * fraction)
train_subset, _ = random_split(train_dataset, [subset_count, len(train_dataset) - subset_count])

client_sizes = [subset_count // 4] * 4
client_sizes[-1] += subset_count - sum(client_sizes)

client_datasets = random_split(train_subset, client_sizes)

class QNN(nn.Module):
    def __init__(self, n, L):
        super().__init__()
        self.flatten = nn.Flatten()
        angles = torch.empty((L, n), dtype=torch.float64)
        nn.init.uniform_(angles, -0.01, 0.01)
        self.angles = nn.Parameter(angles)
        self.linear = nn.Linear(2**n, 10)

    def forward(self, x):
        x = F.pad(x, (2, 2, 2, 2), "constant", 0)
        x = self.flatten(x)
        x /= torch.linalg.norm(x.clone(), ord=2, dim=1, keepdim=True)
        qc = quantum_circuit(num_qubits=self.angles.shape[1], state_vector=x.T)
        for l in range(self.angles.shape[0]):
            qc.Ry_layer(self.angles[l].to(torch.cfloat))
            qc.cx_linear_layer()
        x = torch.real(qc.probabilities())
        x = self.linear(x.T)
        return x

def performance_estimate(dataset, model, loss_fn):
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)
    model.eval()
    loss, accuracy = 0.0, 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            accuracy += (pred.argmax(1) == y).sum().item()
            loss += loss_fn(pred, y).item()
    accuracy /= len(dataset)
    loss /= len(dataloader)
    return accuracy, loss

def fedavg(local_models):
    avg_state = copy.deepcopy(local_models[0])
    for key in avg_state.keys():
        for i in range(1, len(local_models)):
            avg_state[key] += local_models[i][key]
        avg_state[key] = avg_state[key] / len(local_models)
    return avg_state

def train_one_client(model, dataset, epochs=5, lr=1e-1, weight_decay=1e-10):
    model = copy.deepcopy(model)
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

    for _ in range(epochs):
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def federated_training(client_datasets, test_dataset, L=4, global_rounds=5, local_epochs=5, lr=1e-1, wd=1e-10, n=10):
    global_model = QNN(n=n, L=L).to(device)
    loss_fn = nn.CrossEntropyLoss()

    for round_idx in range(global_rounds):
        local_states = [train_one_client(global_model, c_data, epochs=local_epochs, lr=lr, weight_decay=wd).state_dict() for c_data in client_datasets]
        avg_state = fedavg(local_states)
        global_model.load_state_dict(avg_state)
    
        acc_test, loss_test = performance_estimate(test_dataset, global_model, loss_fn)
        print(f"Round {round_idx + 1}: Accuracy = {acc_test:.4f}, Loss = {loss_test:.4f}")

    return global_model

n = 10
L = 4
global_rounds = 50
local_epochs = 5
lr_ = 1e-1
weight_decay_ = 1e-10

print(f"Using device: {device}")
print(f"Number of qubits = {n}")
print(f"Number of quantum layers = {L}")
print(f"Number of angles = {n*L}")
print(f"Total training set size for federation = {len(train_subset)}")

for i, c in enumerate(client_datasets):
    print(f"Client {i + 1} dataset size: {len(c)}")

global_model = federated_training(
    client_datasets=client_datasets,
    test_dataset=test_dataset,
    n=n,
    L=L,
    global_rounds=global_rounds,
    local_epochs=local_epochs,
    lr=lr_,
    wd=weight_decay_
)

print("~~~~~ Federated Training Complete ~~~~~")
