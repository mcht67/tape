from datasets import load_dataset, Dataset, Audio, Features, Sequence, Value, load_from_disk, concatenate_datasets
from pathlib import Path
import numpy as np
import random
import os
import psutil
from collections import defaultdict
import copy
import gc
import tempfile
import shutil


def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

def create_index_map(num_indices):
    '''
            Creates dictionary of integers and booleans corresponding to used and unused indices initialized to False. 
    
            :param num_indices: Number of indices of the dict to create
            :type num_indices: int
            
            :return: Dictionary with indices as keys and booleans as values
            :rtype: dict of int: bool
    '''
    # Setup random indexing
    indices = list(range( num_indices ))
    random.shuffle(indices)
    index_map = dict(zip(indices, [False] * len(indices)))
    return index_map

def create_index_map_from_range(range):
    '''
            Creates dictionary of integers and booleans corresponding to used and unused indices initialized to False. 
    
            :param num_indices: Number of indices of the dict to create
            :type num_indices: int
            
            :return: Dictionary with indices as keys and booleans as values
            :rtype: dict of int: bool
    '''
    # Setup random indexing
    indices = list(range)
    random.shuffle(indices)
    index_map = dict(zip(indices, [False] * len(indices)))
    return index_map

def pop_random_index(index_map):
    '''
            Gets next false key from index map and sets it to True. 
            
            :return: Pseudo random index
            :rtype: int
    '''
    # Find the keys where the value is False
    false_keys = [key for key, value in index_map.items() if value is False]

    first_key = false_keys[0]

    index_map[first_key] = True

    return first_key

def reset_index_map(index_map):
    '''
            Resets the index map by setting all values to False
    '''
    for key, value in index_map.items():
        index_map[key] = False

def flatten_raw_examples(raw_examples):
    flattened = defaultdict(list)
    for ex in raw_examples:
        for key, val in ex.items():
            flattened[f"raw_files_{key}"].append(val)
    return dict(flattened)


def flatten_features(prefix: str, features: Features) -> Features:
    flat = {}
    for key, value in features.items():
        flat[f"{prefix}_{key}"] = Sequence(value)
    return Features(flat)

def generate_mix_examples(raw_data, max_polyphony_degree, segment_duration, sampling_rate):

    # Create polyphony degree map and get initial value
    polyphony_map = create_index_map_from_range(range(1, max_polyphony_degree + 1))
    polyphony_degree = pop_random_index(polyphony_map)

    # init containers
    raw_signals = []
    raw_data_list = []
    ebird_code_multilabel = []

    mix_id = 0

    for example in raw_data:
        # Collect signals up to polyphony degree
        audio = example["audio"]
        raw_signal = audio["array"]  # This is a float32 NumPy array
        filename = Path(audio["path"]).name

        # Check sampling rate
        if not audio['sampling_rate'] == sampling_rate:
            print("ups")
            continue

        # If signal length below chosen segment duration in seconds, skip it
        if raw_signal.size < segment_duration * sampling_rate:
            print(f'Skipping {filename} due to insufficient length.')
            continue

        # If stereo, sum to mono
        if raw_signal.ndim > 1:  
            raw_signal = np.mean(raw_signal, axis=0)
        raw_signals.append(raw_signal)

        raw_data_list.append(example)

        ebird_code = example['ebird_code']
        ebird_code_multilabel.append(ebird_code)

        # Mix colleted signals
        if len(ebird_code_multilabel) == polyphony_degree:  

            # Trim/pad to the same length
            min_len = min(len(a) for a in raw_signals)
            raw_signals = [a[:min_len] for a in raw_signals]#

            # Mix (sum all waveforms)
            mixed_signal = np.sum(raw_signals, axis=0)

            # Normalize to prevent clipping
            mixed_signal = mixed_signal / np.max(np.abs(mixed_signal))

            flattened_raw = flatten_raw_examples(raw_data_list)

            mix_example = {
                "id": str(mix_id),
                "mixed_signal": mixed_signal.copy(),
                "polyphony_degree": int(polyphony_degree),
                "ebird_code_multilabel": ebird_code_multilabel[:],
                **flattened_raw.copy()
            }
            yield mix_example

            # Reset mixing stage
            mix_id += 1

            raw_signals = []
            raw_data_list = []
            ebird_code_multilabel = []

            # Check if all polyphony degrees have been used
            if (all(polyphony_map.values())):
                reset_index_map(polyphony_map)
            
            polyphony_degree = pop_random_index(polyphony_map)

def generate_batches(raw_data, max_polyphony_degree, segment_duration, sampling_rate, batch_size=100):
    batch = []
    for example in generate_mix_examples(raw_data, max_polyphony_degree, segment_duration, sampling_rate):
        batch.append(example)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

relative_path = "../data/datasets/"
abs_path = os.path.abspath(relative_path)
print(abs_path)

# Configuration 
# TODO: make these arguments
random.seed(42)

sampling_rate = 32000

segment_duration = 5  # in seconds
max_polyphony_degree = 7

dataset_key = "HSN_xc"
dataset_split = "train"

output_path = "../data/datasets/" + dataset_key + "_mix/"
os.makedirs(output_path, exist_ok=True)

# Load raw dataset
raw_data = load_dataset("DBD-research-group/BirdSet", dataset_key, split=dataset_split)
raw_data = raw_data.cast_column("audio", Audio(sampling_rate=sampling_rate))

# Setup features
raw_features = raw_data.features
flattened_raw_features = flatten_features("raw_files", raw_features)

mix_features = Features({
    "id": Value("string"),
    "mixed_signal": Sequence(Value("float32")),
    "polyphony_degree": Value("int32"),
    "ebird_code_multilabel": Sequence(raw_features["ebird_code_multilabel"].feature),
    **flattened_raw_features
})

# Generate batches
temp_dirs = []

for i, batch in enumerate(generate_batches(raw_data.select(range(100)), max_polyphony_degree, segment_duration, sampling_rate)):
    ds = Dataset.from_list(batch, features=mix_features)
    print(f'finished batch {i}')
    
    tmp_dir = tempfile.mkdtemp(prefix=f"mix_batch_{i}_")
    ds.save_to_disk(tmp_dir)
    temp_dirs.append(tmp_dir)

# Concatenate batches
datasets = [load_from_disk(d) for d in temp_dirs]
full_dataset = concatenate_datasets(datasets)

# Save final dataset
full_dataset.save_to_disk(output_path)
print(f'Saved to {output_path}')

# Load dataset
new_dataset = load_from_disk(output_path)
print(f'New Dataset saved to disk: {new_dataset}')

# Remove tmp files
for d in temp_dirs:
    shutil.rmtree(d)