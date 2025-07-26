from perch_hoplite.zoo import model_configs
from omegaconf import OmegaConf
from datasets import load_from_disk
import numpy as np
from functools import partial
import shutil
from utils.dsp import resample_audio

def load_model_by_key(model_key):
    model_config_name = model_configs.ModelConfigName(model_key)
    preset_info = model_configs.get_preset_model_config(model_config_name)
    model = preset_info.load_model()

    return model, preset_info

def embed_example(example, model, signal_key, new_feature_key, sampling_rate):

    audio = example[signal_key]
    audio = resample_audio(audio, audio['sampling_rate'], sampling_rate)

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

    model_keys = ['perch_8','yamnet']
    signal_key = 'mixed_signal'

    dataset_key = cfg.dataset.dataset_key
    dataset_dir = cfg.dataset.output_dir

    original_dataset_path = dataset_dir + dataset_key + "_mix/"
    temp_path = dataset_dir + "temp/" + dataset_key + "_mix/"

     # Load Original Dataset
    dataset = load_from_disk(original_dataset_path)

    for model_key in model_keys:

        new_feature_key = model_key + "_embedding"

        if not new_feature_key in list(dataset.features.keys()):

            # Load model
            model, preset_info = load_model_by_key(model_key)
            sampling_rate = preset_info.model_config["sample_rate"]

            # Embed signals
            embedding_fn = partial(embed_example, model=model, signal_key=signal_key, new_feature_key=new_feature_key, sampling_rate=sampling_rate)
            dataset = dataset.map(embedding_fn)

    # Save to temporary location
    dataset.save_to_disk(temp_path)

    # Remove original and rename temp
    shutil.rmtree(original_dataset_path)
    shutil.move(temp_path, original_dataset_path)

if __name__ == "__main__":
    main()