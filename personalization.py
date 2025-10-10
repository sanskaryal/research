#!/usr/bin/env python3
"""
personalization.py

This script runs a federated learning simulation with client-side personalization.
It imports the CapsNet model, data loading, and federated learning components
from federated_caps.py and adds a personalization step.

The personalization is achieved by creating a weighted average of the global
model's weights and the client's own weights from the previous round. This allows
each client to maintain a degree of specialization while still benefiting from
the aggregated knowledge of the federated network.

A new configuration parameter, `personalization_weight`, is introduced to control
the balance between the global model and the client's local model. A value of 1.0
means the client will only use the global model, while a value of 0.0 means it
will only use its own model from the previous round.

To run the simulation, execute this script from your terminal:
    python personalization.py
"""

import time
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Import necessary components from federated_caps.py
from federated_caps import (
    CONFIG,
    N_CLASSES,
    CapsNet,
    device,
    train_local_caps,
    evaluate_caps,
    fed_avg,
    client_datasets,
    valid_loader,
    test_loader,
    num_clients,
)

# =============================
# Personalization Configuration
# =============================
# Add the personalization weight to the configuration
CONFIG["personalization_weight"] = 0.5  # 0.0 = local only, 1.0 = global only

# =============================
# File Naming for Personalization
# =============================
def get_filenames_perso():
    """Create descriptive filenames for personalization outputs."""
    iid_str = "iid" if CONFIG["iid"] else f"niid_{CONFIG['dirichlet_alpha']}"
    base = f"fed_caps_{iid_str}_{CONFIG['num_clients']}clients_perso"
    
    if CONFIG["data_frac"] < 1.0:
        base += f"_frac{CONFIG['data_frac']}"
    
    csv_name = Path("results") / f"{base}.csv"
    model_name = Path("trained_models") / f"{base}.pth"
    
    csv_name.parent.mkdir(parents=True, exist_ok=True)
    model_name.parent.mkdir(parents=True, exist_ok=True)
    
    return csv_name, model_name

CSV_PATH, MODEL_PATH = get_filenames_perso()
print(f"Personalization results will be saved to: {CSV_PATH}")
print(f"Personalized model will be saved to: {MODEL_PATH}")

# =============================
# Federated Training with Personalization
# =============================
def run_federated_personalization():
    """
    Runs the federated learning simulation with client-side personalization.
    """
    global_model = CapsNet(img_size=CONFIG["image_size"], num_classes=N_CLASSES).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in global_model.parameters() if p.requires_grad):,}")

    # Store last sent weights for each client
    client_last_weights = {i: None for i in range(num_clients)}

    rounds = CONFIG["rounds"]
    frac = CONFIG["frac_clients"]
    local_epochs = CONFIG["local_epochs"]

    best_val = -1.0
    t0 = time.time()
    results_log = []

    for r in range(1, rounds + 1):
        start = time.time()
        base_state = deepcopy(global_model.state_dict())

        m = max(1, int(frac * num_clients))
        selected = np.random.default_rng(CONFIG["seed"] + r).choice(num_clients, size=m, replace=False)

        updates, weights = [], []
        round_train_losses, round_train_accs = [], []
        for cid in selected:
            client_model = deepcopy(global_model).to(device)
            
            # Personalization step
            if client_last_weights[cid] is not None:
                p_weight = CONFIG["personalization_weight"]
                perso_state = deepcopy(base_state)
                for key in perso_state.keys():
                    perso_state[key] = (p_weight * base_state[key]) + ((1 - p_weight) * client_last_weights[cid][key])
                client_model.load_state_dict(perso_state)
            else:
                client_model.load_state_dict(base_state)

            sd, n, loss, acc = train_local_caps(client_model, client_datasets[cid], epochs=local_epochs, lr=CONFIG["lr"])
            
            client_last_weights[cid] = deepcopy(sd)
            
            updates.append(sd)
            weights.append(n)
            round_train_losses.append(loss)
            round_train_accs.append(acc)

        new_state = fed_avg(updates, weights)
        global_model.load_state_dict(new_state)

        val_metrics = evaluate_caps(global_model, valid_loader)
        test_metrics = evaluate_caps(global_model, test_loader)

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

    global_model.load_state_dict(torch.load(MODEL_PATH))
    final_test_metrics = evaluate_caps(global_model, test_loader)
    print(f"\nFinal Test Metrics on Best Model: {final_test_metrics}")

    final_test_metrics["round"] = "best_model_test"
    df = pd.concat([df, pd.DataFrame([final_test_metrics])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

if __name__ == "__main__":
    run_federated_personalization()