from datasets import load_from_disk, DatasetDict, Dataset
from transformers import DefaultDataCollator
from omegaconf import OmegaConf
from tensorflow.keras import layers, models
import numpy as np
from utils.general import reshape_tensor_data
from functools import partial
import tensorflow as tf
import tempfile
import os
from utils.logs import return_tensorboard_path
from tensorflow.keras.callbacks import TensorBoard


def split_dataset(test_split, val_split, dataset, random_seed):
    # Split into train/test first (e.g., 90/10) -> test size = 0.1 * number of items
    train_test = dataset.train_test_split(test_size=test_split, shuffle=True, seed=random_seed)

    # Split the training set further to create validation (e.g., 80/10/10) 0.1 * number of items = x * 0.9 * number of items -> x = 0.1 / 0.9 = 0.11
    val_split_factor = val_split / (1 - test_split) 
    train_val = train_test['train'].train_test_split(test_size=val_split_factor, shuffle=True, seed=random_seed)  # 0.11 * 0.9 = 0.1 of total

    # TODO: Add stratified splitting

    # Create the final dataset dictionary
    return DatasetDict({
        'train': train_val['train'],      
        'validation': train_val['test'],   
        'test': train_test['test']        
    })

def get_tf_datasets(dataset, features, labels, batch_size):
    train_dataset = dataset['train'].to_tf_dataset(
        columns=features,
        label_cols=labels, 
        batch_size=batch_size,
        shuffle=True,
        prefetch=True
    )

    test_dataset = split_dataset['test'].to_tf_dataset(
        columns=reshaped_features,
        label_cols=labels,
        batch_size=batch_size,
        shuffle=False,
        prefetch=True
    )

    val_dataset = split_dataset['validation'].to_tf_dataset(
        columns=features,
        label_cols=labels,
        batch_size=batch_size,
        shuffle=False,
        prefetch=True
    )

    return train_dataset, test_dataset, val_dataset

# Configuration
cfg = OmegaConf.load("params.yaml")

random_seed = cfg.general.random_seed

test_split = cfg.train.test_split
val_split = cfg.train.val_split
epochs = cfg.train.epochs

features = cfg.train.features 
labels = cfg.train.labels # 'polyphony_degree'
suffix = "_reshaped"
pooling_strategy = cfg.train.pooling_strategy

reshaped_features = features + suffix

batch_size = cfg.train.batch_size

preprocessed_dataset_path =  cfg.paths.preprocessed_dataset

# Setup tensorboard
tensorboard_path = return_tensorboard_path()

# Load dataset
preprocessed_dataset = load_from_disk(preprocessed_dataset_path)

# Split dataset
split_dataset = split_dataset(test_split, val_split, preprocessed_dataset, random_seed)

# Reshape features for training
embedding_length = np.array(split_dataset['train'][0][features]).shape[-1]
input_dim = (embedding_length,)

for key, subset in split_dataset.items():
    split_dataset[key] = subset.map(
        lambda x: reshape_tensor_data(x, features, input_dim, pooling_strategy, suffix),
        desc=f"Reshaping {key} embeddings"
    )

# Get tensorflow datasets
train_dataset, test_dataset, val_dataset = get_tf_datasets(split_dataset, reshaped_features, labels, batch_size)

# Define model
model = models.Sequential([
    layers.Input(shape=input_dim), 
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dense(1)  # Regression output: total polyphony degree
])

# Callbacks
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

# Train model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs, callbacks=[tensorboard_callback])
model.save('polyReg.keras')

split_dataset.cleanup_cache_files()

print(f"Launch TensorBoard with: tensorboard --logdir={tensorboard_path}")