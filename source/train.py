from datasets import load_from_disk, DatasetDict, concatenate_datasets
from transformers import DefaultDataCollator
from omegaconf import OmegaConf
from tensorflow.keras import layers, models
import numpy as np
from utils.general import reshape_tensor_data
from functools import partial
import tensorflow as tf
import tempfile
import os
from utils.logs import return_tensorboard_path, plot_confusion_matrix
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import io
from utils.config import set_random_seeds

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

    test_dataset = dataset_splits['test'].to_tf_dataset(
        columns=features,
        label_cols=labels,
        batch_size=batch_size,
        shuffle=False,
        prefetch=True
    )

    val_dataset = dataset_splits['validation'].to_tf_dataset(
        columns=features,
        label_cols=labels,
        batch_size=batch_size,
        shuffle=False,
        prefetch=True
    )

    return train_dataset, test_dataset, val_dataset

def apply_batched_reshape(dataset, features, input_dim, pooling_strategy, suffix, batch_size=100):
    """Apply reshape_tensor_data in batches to avoid PyArrow offset overflow"""
    
    # Process in batches
    processed_datasets = []
    total_samples = len(dataset)
    
    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        print(f"Processing reshape batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}")
        
        # Select batch
        batch_dataset = dataset.select(range(i, end_idx))
        
        # Process batch
        batch_processed = batch_dataset.map(
            lambda x: reshape_tensor_data(x, features, input_dim, pooling_strategy, suffix),
            desc=f"Reshaping batch {i//batch_size + 1}"
        )
        
        processed_datasets.append(batch_processed)
    
    # Concatenate all processed batches
    print(f"Concatenating {len(processed_datasets)} reshape batches...")
    return concatenate_datasets(processed_datasets)

def get_predictions_and_true_labels(model, dataset):
    y_pred = []
    y_true = []

    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch, verbose=0)
        y_pred.append(preds)
        y_true.append(y_batch.numpy())

    return y_pred, y_true

def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    plt.close(figure)
    return image

# Configuration
cfg = OmegaConf.load("params.yaml")

random_seed = cfg.general.random_seed
set_random_seeds(random_seed)

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


for version in ["_epochs100"]:
    # Setup tensorboard
    tensorboard_path = './logs/' + features + version #return_tensorboard_path()
    os.makedirs(tensorboard_path, exist_ok=True)
    print(tensorboard_path)

    # Load dataset
    preprocessed_dataset = load_from_disk(preprocessed_dataset_path)

    # Split dataset
    dataset_splits = split_dataset(test_split, val_split, preprocessed_dataset, random_seed)

    # Get input dim
    # embedding_length = np.array(split_dataset['train'][0][features]).shape[-1]
    # input_dim = (embedding_length,)
    input_dim = np.array(dataset_splits['train'][0][features]).shape

    # # Reshape features for training
    # for key, subset in split_dataset.items():
    #     print(f"Reshaping {key} embeddings...")
    #     split_dataset[key] = apply_batched_reshape(
    #         subset, features, input_dim, pooling_strategy, suffix, batch_size=100
    #     )

    # Get tensorflow datasets
    train_dataset, test_dataset, val_dataset = get_tf_datasets(dataset_splits, features, labels, batch_size)

    # Define model
    model = models.Sequential([
        layers.Input(shape=input_dim), 
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dense(1)  # Regression output: total polyphony degree
    ])

    # Callbacks
    tensorboard_callback = TensorBoard(log_dir=tensorboard_path, histogram_freq=1)

    # Train model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=[tensorboard_callback])
    model.save('polyReg.keras')

    # Plot Confusion Matrix
    y_pred, y_true = get_predictions_and_true_labels(model, val_dataset)

    figure = plot_confusion_matrix(y_pred, y_true)
    image = plot_to_image(figure)

    writer = tf.summary.create_file_writer(tensorboard_path)
    with writer.as_default():
        tf.summary.image("Confusion Matrix", image, step=0)
    writer.close()

    dataset_splits.cleanup_cache_files()