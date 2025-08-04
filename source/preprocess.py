from perch_hoplite.zoo import model_configs
from omegaconf import OmegaConf
from datasets import load_from_disk
import numpy as np
from functools import partial
import shutil
from utils.dsp import resample_audio
import os

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

def add_embeddings(model_keys, feature_key, dataset):
    modified = False
    for model_key in model_keys:
        new_feature_key = model_key + "_embedding"
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
            dataset = dataset.map(embedding_fn)
            modified = True
    return dataset, modified

def main():

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
    modified_dataset, modified = add_embeddings(model_keys, feature_key, dataset)

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