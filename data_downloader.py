import os
import medmnist

def _choose_dataclass(view: str):
    """Pick the MedMNIST OrganMNIST class robustly across versions."""
    # In newer versions, classes are in a submodule
    try:
        from medmnist import dataset
        search_module = dataset
    except ImportError:
        search_module = medmnist

    view = view.lower()
    # Most likely to least likely names
    class_map = {
        "axial": ["OrganMNISTAxial", "OrganMNISTAxi"],
        "coronal": ["OrganMNISTCoronal", "OrganMNISTCor"],
        "sagittal": ["OrganMNISTSagittal", "OrganMNISTSag"],
    }
    for name in class_map.get(view, []):
        cls = getattr(search_module, name, None)
        if cls is not None:
            return cls
            
    # Fallback for very old/new versions
    for attr in dir(search_module):
        if "Organ" in attr and "MNIST" in attr and view in attr.lower():
            cls = getattr(search_module, attr, None)
            if cls is not None:
                return cls
                
    raise RuntimeError(f"Could not locate an OrganMNIST class for view '{view}' in medmnist library.")

def download_data(data_path='data/'):
    """Downloads all OrganMNIST datasets."""
    os.makedirs(data_path, exist_ok=True)
    
    views = ["axial", "coronal", "sagittal"]
    
    for view in views:
        print(f"Downloading OrganMNIST-{view}...")
        DataClass = _choose_dataclass(view)
        
        # Download train, validation, and test sets
        for split in ['train', 'val', 'test']:
            _ = DataClass(split=split, download=True, root=data_path)
    
    print("\nAll datasets downloaded and saved to 'data/' directory.")

if __name__ == "__main__":
    download_data()
