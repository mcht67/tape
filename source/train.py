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
#tensorboard_path = return_tensorboard_path()

# Load dataset
preprocessed_dataset = load_from_disk(preprocessed_dataset_path)

# Split dataset
split_dataset = split_dataset(test_split, val_split, preprocessed_dataset, random_seed)

# Reshape features for training
embedding_length = np.array(split_dataset['train'][0][features]).shape[-1]
input_dim = (embedding_length,)

# for key, subset in split_dataset.items():
#     split_dataset[key] = subset.map(
#         lambda x: reshape_tensor_data(x, features, input_dim, pooling_strategy, suffix),
#         desc=f"Reshaping {key} embeddings"
#     )

for key, subset in split_dataset.items():
    print(f"Reshaping {key} embeddings...")
    split_dataset[key] = apply_batched_reshape(
        subset, features, input_dim, pooling_strategy, suffix, batch_size=100
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

# Extract predictions and ground truth
y_true = []
y_pred = []

for x_batch, y_batch in test_dataset:
    y_true.append(y_batch.numpy())
    preds = model.predict(x_batch, verbose=0)
    y_pred.append(preds)

print(y_true)
print(y_pred)

# Convert to flat NumPy arrays
y_true =  np.concatenate(y_true, axis=0) #np.array(y_true)
y_pred = np.concatenate(y_pred, axis=0) #np.array(y_pred)

print(y_true)
print(y_pred)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.patches as patches

# Assuming last column is total polyphony degree
y_true_total = np.round(y_true).astype(int)
y_pred_total = np.round(y_pred).astype(int).flatten()

print(y_true_total)
print(y_pred_total)
# Get all unique polyphony degrees in true total counts
unique_total_classes = np.unique(np.concatenate([y_true_total, y_pred_total]))

# Compute confusion matrix for total polyphony degrees
cm_total = confusion_matrix(y_true_total, y_pred_total, labels=unique_total_classes)

# Calculate percentages per true class (row-wise)
with np.errstate(all='ignore'):
    cm_total_percent = cm_total / cm_total.sum(axis=1, keepdims=True) * 100

# Create annotation strings (count + percentage)
annot_total = np.empty_like(cm_total).astype(str)
for r in range(cm_total.shape[0]):
    for c in range(cm_total.shape[1]):
        count = cm_total[r, c]
        pct = cm_total_percent[r, c]
        annot_total[r, c] = f"{count}\n({pct:.1f}%)"


# Plot heatmap for total polyphony degree
figure = plt.figure(figsize=(8, 6))
ax = sns.heatmap(cm_total, annot=annot_total, fmt='', cmap='Blues', cbar=True,
                 xticklabels=unique_total_classes,
                 yticklabels=unique_total_classes,
                 annot_kws={"fontsize": 10})

# Highlight diagonal cells with a red rectangle
for i in range(len(unique_total_classes)):
    ax.add_patch(patches.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=3))

plt.title("Confusion Matrix for Total Polyphony Degree")
plt.xlabel("Predicted Total Polyphony Degree")
plt.ylabel("True Total Polyphony Degree")
plt.tight_layout()
# plt.show()

import io
def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    plt.close(figure)
    return image

image = plot_to_image(figure)

writer = tf.summary.create_file_writer('./logs')
with writer.as_default():
    tf.summary.image("Confusion Matrix", image, step=0)
writer.close()

split_dataset.cleanup_cache_files()

# print(f"Launch TensorBoard with: tensorboard --logdir={tensorboard_path}")