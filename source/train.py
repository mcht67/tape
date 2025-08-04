from datasets import load_from_disk
from omegaconf import OmegaConf
from tensorflow.keras import layers, models
import numpy as np
from utils.general import reshape_tensor_data

# Configuration
cfg = OmegaConf.load("params.yaml")

random_seed = cfg.general.random_seed

input_dim = 1280
epochs = 20

features = 'perch_8_embedding'
labels = 'polyphony_degree'
suffix = "_reshaped"

reshaped_features = features + suffix

dataset_key = cfg.dataset.dataset_key
split_dataset_dir = cfg.dataset.split_dataset_dir

dataset_path = split_dataset_dir + dataset_key + "_mixed_split/"

# Load dataset
dataset = load_from_disk(dataset_path)

# Reshape features for training
for key, subset in dataset.items():
    dataset[key] = subset.map(
        lambda x: reshape_tensor_data(x, features, (input_dim,), "mean", suffix),
        desc=f"Reshaping {key} embeddings"
    )

# Get tensorflow datasets
train_dataset = dataset['train'].to_tf_dataset(
    columns=reshaped_features,
    label_cols=labels, 
    batch_size=32,
    shuffle=True,
    prefetch=True
)

test_dataset = dataset['test'].to_tf_dataset(
    columns=reshaped_features,
    label_cols=labels,
    batch_size=32,
    shuffle=False,
    prefetch=True
)

val_dataset = dataset['validation'].to_tf_dataset(
    columns=reshaped_features,
    label_cols=labels,
    batch_size=32,
    shuffle=False,
    prefetch=True
)

# Define model
model = models.Sequential([
    layers.Input(shape=(input_dim,)), 
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dense(1)  # Regression output: total polyphony degree
])

# Train model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs)
model.save('polyReg.keras')
