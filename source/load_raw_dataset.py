from datasets import load_dataset
from omegaconf import OmegaConf

# Load Configuration
cfg = OmegaConf.load("params.yaml")

sampling_rate = cfg.dataset.sampling_rate

dataset_key = cfg.dataset.dataset_key
dataset_split = cfg.dataset.dataset_split

raw_dataset_dir = cfg.dataset.raw_dataset_dir
output_path = raw_dataset_dir + dataset_key

# Load raw dataset
print("Load dataset", flush=True)
raw_dataset = load_dataset("DBD-research-group/BirdSet", dataset_key, split=dataset_split, trust_remote_code=True)

# Save raw dataset
raw_dataset.save_to_disk(output_path)
print(f"Saved raw dataset {dataset_key} to {raw_dataset_dir}")

