# CapsNet for OrganMNIST

This project implements a Capsule Network (CapsNet) to classify 3D medical images from the [MedMNIST OrganMNIST dataset](https://medmnist.com/). It is designed to be a self-contained research experiment.

## How It Works

The project is split into two main files:

1.  `data_downloader.py`: A utility script to download the necessary OrganMNIST datasets (Axial, Coronal, and Sagittal views).
2.  `caps_initial.py`: The main script that defines, trains, and evaluates the CapsNet model on one of the downloaded datasets.
3.  `federated_caps.py`: A script that trains the CapsNet model in a federated learning setting.


## Requirements

All required Python packages are listed in `requirements.txt`. Install them using pip:

```bash
pip install -r requirements.txt
```

## Usage

Follow these steps to run the project:

### 1. Download the Datasets

First, run the data downloader script. This will download all three OrganMNIST view datasets into a `data/` directory.

```bash
python data_downloader.py
```

### 2. Run the Training Script

Once the data is downloaded, you can train the CapsNet model by running the main script:

```bash
python caps_initial.py
```

The script will:
- Use the 'axial' view dataset by default.
- Set up the CapsNet model.
- Train the model for 30 epochs.
- Print the validation accuracy after each epoch.
- Report the final test accuracy after training is complete.

### 3. Run the Federated Training Script

To train the CapsNet model using federated learning, run the `federated_caps.py` script:

```bash
python federated_caps.py
```

## Configuration

You can change hyperparameters and settings directly in the `CONFIG` dictionary at the top of the `caps_initial.py` file.

```python
# caps_initial.py

# ...
CONFIG = {
    "view": "axial",          # 'axial', 'coronal', or 'sagittal'
    "epochs": 30,
    "batch_size": 128,
    "lr": 2e-4,
    "seed": 42,
    "image_size": 28,
    "num_workers": 0,
}
# ...
```

To train on a different view, simply change the `"view"` parameter to `"coronal"` or `"sagittal"`.
