from perch_hoplite.zoo import model_configs
from omegaconf import OmegaConf
from datasets import load_from_disk, Dataset
import datasets
import numpy as np
from functools import partial
import shutil
from utils.dsp import resample_audio
import os
import tempfile

def load_model_by_key(model_key):
    model_config_name = model_configs.ModelConfigName(model_key)
    preset_info = model_configs.get_preset_model_config(model_config_name)
    model = preset_info.load_model()

    return model, preset_info

def embed_example(example, model, feature_key, new_feature_key, sampling_rate):

    audio = example[feature_key]
    audio = resample_audio(audio, example['sampling_rate'], sampling_rate)

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    # Get embedding
    outputs = model.embed(audio)
    example[new_feature_key] = outputs.embeddings

    return example

def add_embeddings(model_keys, feature_key, dataset, cache_dir):
    modified = False
    for model_key in model_keys:
        new_feature_key = model_key + "_embeddings"
        if new_feature_key not in dataset.features:
            model, preset_info = load_model_by_key(model_key)
            sampling_rate = preset_info.model_config["sample_rate"]
            embedding_fn = partial(
                embed_example,
                model=model,
                feature_key=feature_key,
                new_feature_key=new_feature_key,
                sampling_rate=sampling_rate,
            )
            cache_file = os.path.join(cache_dir, f"{model_key}_cache.arrow")
            dataset = dataset.map(embedding_fn, cache_file_name=cache_file)
            modified = True
    return dataset, modified

def add_embeddings_batchwise(model_keys, feature_key, dataset, cache_dir, batch_size=100):
    modified = False
    
    for model_key in model_keys:
        new_feature_key = model_key + "_embeddings"
        if new_feature_key not in dataset.features:
            print(f"Processing {model_key} embeddings in batches of {batch_size}...")
            
            model, preset_info = load_model_by_key(model_key)
            sampling_rate = preset_info.model_config["sample_rate"]
            
            embedding_fn = partial(
                embed_example,
                model=model,
                feature_key=feature_key,
                new_feature_key=new_feature_key,
                sampling_rate=sampling_rate,
            )
            
            # Process in batches
            processed_datasets = []
            total_samples = len(dataset)
            
            for i in range(0, total_samples, batch_size):
                end_idx = min(i + batch_size, total_samples)
                print(f"Processing batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}")
                
                # Select batch
                batch_dataset = dataset.select(range(i, end_idx))
                
                # Process batch
                cache_file = os.path.join(cache_dir, f"{model_key}_batch_{i}_{end_idx}_cache.arrow")
                batch_processed = batch_dataset.map(embedding_fn, cache_file_name=cache_file)
                
                processed_datasets.append(batch_processed)
            
            # Concatenate all processed batches
            print(f"Concatenating {len(processed_datasets)} batches...")
            dataset = datasets.concatenate_datasets(processed_datasets)
            modified = True
    
    return dataset, modified

def main():

    with tempfile.TemporaryDirectory() as temp_cache_dir:
        # Set HuggingFace cache to this temporary directory
        datasets.config.HF_DATASETS_CACHE = temp_cache_dir
        # Configuration
        cfg = OmegaConf.load("params.yaml")

        model_keys = cfg.embeddings.models
        feature_key = cfg.embeddings.feature

        polyphonic_dataset_path = cfg.paths.polyphonic_dataset
        preprocessed_dataset_path = cfg.paths.preprocessed_dataset

        # Load polyphonic dataset to satisfy dvcs dependency tracking
        dataset = load_from_disk(polyphonic_dataset_path)
        
        # Load Dataset depending on state of preprocessing dataset
        if not os.path.exists(preprocessed_dataset_path):
            os.makedirs(preprocessed_dataset_path, exist_ok=True)
        else:
            dataset = load_from_disk(preprocessed_dataset_path)
        
        # Preprocessing
        dataset_was_modified = False

        # Embeddings
        modified_dataset, modified = add_embeddings_batchwise(model_keys, feature_key, dataset, temp_cache_dir)

        if modified:
            dataset = modified_dataset
            dataset_was_modified = True
        else:
            print("No changes in embeddings.")

        # Save dataset
        if not dataset_was_modified:
            print('Preprocessed Dataset was not modified.')
        
        else:
            # Save to temporary location
            temp_path = preprocessed_dataset_path + "_temp"
            os.makedirs(temp_path, exist_ok=True)
            dataset.save_to_disk(temp_path)

            # Move old data to backup
            backup_path = preprocessed_dataset_path + "_backup"
            if os.path.exists(preprocessed_dataset_path):
                shutil.move(preprocessed_dataset_path, backup_path)

            # Move temp data into place
            shutil.move(temp_path, preprocessed_dataset_path)

            # Optionally remove backup
            shutil.rmtree(backup_path)

if __name__ == "__main__":
    main()