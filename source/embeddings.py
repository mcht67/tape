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

# Available models
# BIRDNET_V2_1 = 'birdnet_V2.1'
# BIRDNET_V2_2 = 'birdnet_V2.2'
# BIRDNET_V2_3 = 'birdnet_V2.3'
# PERCH_8 = 'perch_8'
# SURFPERCH = 'surfperch'
# VGGISH = 'vggish'
# YAMNET = 'yamnet'
# HUMPBACK = 'humpback'
# MULTISPECIES_WHALE = 'multispecies_whale'
# BEANS_BASELINE = 'beans_baseline'
# AVES = 'aves'
# PLACEHOLDER = 'placeholder'

def main():
    # Configuration

    # Load Configuration
    cfg = OmegaConf.load("params.yaml")

    model_keys = cfg.embeddings.models
    feature_key = cfg.embeddings.feature

    mixed_dataset_path = cfg.paths.mixed_dataset
    preprocessed_dataset_path = cfg.paths.preprocessed_dataset
    #temp_path = "temp/" + preprocessed_dataset_path

    os.makedirs(preprocessed_dataset_path, exist_ok=True)
   # os.makedirs(temp_path, exist_ok=True)

     # Load Original Dataset
    dataset = load_from_disk(mixed_dataset_path)

    for model_key in model_keys:

        new_feature_key = model_key + "_embedding"

        if not new_feature_key in list(dataset.features.keys()): # Risky, does not overwrite if embedding has to change, but key exists

            # Load model
            model, preset_info = load_model_by_key(model_key)
            sampling_rate = preset_info.model_config["sample_rate"]

            # Embed signals
            embedding_fn = partial(embed_example, model=model, feature_key=feature_key, new_feature_key=new_feature_key, sampling_rate=sampling_rate)
            dataset = dataset.map(embedding_fn)

    # Save to temporary location
    dataset.save_to_disk(preprocessed_dataset_path)

    # Remove original and move temp
    # shutil.rmtree(preprocessed_dataset_path)
    # shutil.move(temp_path, preprocessed_dataset_path)

if __name__ == "__main__":
    main()