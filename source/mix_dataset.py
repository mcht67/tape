# Very first line in source/dataset.py
print("=== SCRIPT STARTED ===", flush=True)
from datasets import load_dataset, Dataset, Audio, Features, Sequence, Value, load_from_disk, concatenate_datasets
from pathlib import Path
import numpy as np
import os
from collections import defaultdict
import tempfile
import shutil
from omegaconf import OmegaConf
from utils import config
from utils.dsp import resample_audio
from utils.general import create_index_map_from_range, pop_random_index, reset_index_map
print('Imports done', flush=True)

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

def generate_mix_examples(raw_data, max_polyphony_degree, segment_duration, sampling_rate, random_seed=None):

    # Create polyphony degree map and get initial value
    polyphony_map = create_index_map_from_range(range(1, max_polyphony_degree + 1), random_state=random_seed)
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

        # Resample if necessary
        audio = resample_audio(audio, audio['sampling_rate'], sampling_rate)

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
                "audio": mixed_signal.copy(),
                "sampling_rate": int(sampling_rate),
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

def generate_batches(raw_data, max_polyphony_degree, segment_duration, sampling_rate, batch_size=100, random_seed=None):
    batch = []
    for example in generate_mix_examples(raw_data, max_polyphony_degree, segment_duration, sampling_rate, random_seed):
        batch.append(example)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    print("Start dataset creation", flush=True)

    # Load Configuration
    cfg = OmegaConf.load("params.yaml")

    random_seed = cfg.general.random_seed
    sampling_rate = cfg.dataset.sampling_rate

    segment_duration = cfg.dataset.segment_duration # in seconds
    max_polyphony_degree = cfg.dataset.max_polyphony_degree

    dataset_key = cfg.dataset.dataset_key
    #dataset_split = cfg.dataset.dataset_split

    raw_dataset_dir = cfg.dataset.raw_dataset_dir
    raw_dataset_path = raw_dataset_dir + dataset_key

    mixed_dataset_dir = cfg.dataset.mixed_dataset_dir
    output_path = mixed_dataset_dir + dataset_key + "_mixed/"
    os.makedirs(output_path, exist_ok=True)

    # Load raw dataset
    print("Load dataset", flush=True)
    #raw_data = load_dataset("DBD-research-group/BirdSet", dataset_key, split=dataset_split, trust_remote_code=True)
    print(raw_dataset_path)
    raw_data = load_from_disk(raw_dataset_path)
    raw_data = raw_data.cast_column("audio", Audio(sampling_rate=sampling_rate))

    # Setup features
    raw_features = raw_data.features
    flattened_raw_features = flatten_features("raw_files", raw_features)

    mix_features = Features({
        "id": Value("string"),
        "audio": Sequence(Value("float32")),
        "sampling_rate": Value("int32"),
        "polyphony_degree": Value("int32"),
        "ebird_code_multilabel": Sequence(raw_features["ebird_code_multilabel"].feature),
        **flattened_raw_features
    })

    # Generate batches
    print("Mix audio in batches", flush=True)

    temp_dirs = []
    for i, batch in enumerate(generate_batches(raw_data.select(range(100)), 
                                               max_polyphony_degree, segment_duration, 
                                               sampling_rate, random_seed=random_seed)):
        ds = Dataset.from_list(batch, features=mix_features)
        print(f'finished batch {i}')
        
        tmp_dir = tempfile.mkdtemp(prefix=f"mix_batch_{i}_")
        ds.save_to_disk(tmp_dir)
        temp_dirs.append(tmp_dir)

    # Concatenate batches
    print("Concatenate batches", flush=True)
    datasets = [load_from_disk(d) for d in temp_dirs]
    full_dataset = concatenate_datasets(datasets)

    # Save final dataset
    full_dataset.save_to_disk(output_path)
    print(f'Saved dataset to {output_path}', flush=True)

    # Load dataset
    new_dataset = load_from_disk(output_path)
    print(f'New Dataset saved to disk: {new_dataset}', flush=True)

    # Remove tmp files
    for d in temp_dirs:
        shutil.rmtree(d)

if __name__ == "__main__":
    main()