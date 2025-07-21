import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf
from pathlib import Path
from torch.utils.data import random_split

def normalize(data):
    data_norm = max(max(data), abs(min(data)))
    return data / data_norm

def load_data(path):
    
    input_dir = Path(path)
    print("Looking in:", input_dir.resolve())

    input_files = sorted(input_dir.glob("dummy_input_*.txt"))
    target_files = sorted(input_dir.glob("dummy_target_*.txt"))

    print(f"Found {len(input_files)} input files.")
    print(f"Found {len(target_files)} target files.")

    assert len(input_files) > 0 , "No input files have been found."
    assert len(input_files) == len(target_files), "Mismatched input and target files."

    X_list = []
    y_list = []

    for input_path, target_path in zip(input_files, target_files):
        x = np.loadtxt(input_path)
        y = np.loadtxt(target_path)

        assert x.shape == y.shape, f"Shape mismatch: {input_path.name}"

        X_list.append(x)
        y_list.append(y)

        X_array = np.array(X_list, dtype=np.float32)
        y_array = np.array(y_list, dtype=np.float32)

    return torch.tensor(X_array, dtype=torch.float32), torch.tensor(y_array, dtype=torch.float32)

def split_data(tensor, test_split):
    total_len = tensor.size(0)
    test_len = int(total_len * test_split)
    return tensor[:-test_len], tensor[-test_len:]

def main():
    # Load the configuration file
    cfg = OmegaConf.load("params.yaml")

    X_all, y_all = load_data(cfg.preprocess.input_path) # Replace this with your own data loading function
    print("Data loaded and normalized.")

    X_training, X_testing = split_data(X_all, cfg.preprocess.test_split)
    y_training, y_testing = split_data(y_all, cfg.preprocess.test_split)
    print("Data split into training and testing sets.")
  
    output_file_path = Path(cfg.preprocess.output_file_path) # output path differs for runs with dvc (hydra) and file runs (params.yaml)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'X_training': X_training,
        'y_training': y_training,
        'X_testing': X_testing,
        'y_testing': y_testing
    }, output_file_path)
    print("Preprocessing done and data saved.")

if __name__ == "__main__":
    main()