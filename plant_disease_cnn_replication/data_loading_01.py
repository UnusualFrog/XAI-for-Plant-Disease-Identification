"""
Replication - Data Loading and Preprocessing
"""

import os
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

layers = tf.keras.layers
models = tf.keras.models
Model = tf.keras.Model

# Use HDD for TFDS
TFDS_DATA_DIR = "/mnt/HDD T4T/tensorflow_datasets"
os.environ["TFDS_DATA_DIR"] = TFDS_DATA_DIR

# Global Random Seed for Reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Global Hyperparameters
IMG_SIZE = 256
# Note: batch size absent from paper, default of 16 used
BATCH_SIZE = 16
EPOCHS = 50
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

# Dataset-specific class counts
NUM_CLASSES = {
    "plantvillage": 17,
    "rice": 4,
    "cassava": 5,
}

# =============================================================================
# PLANTVILLAGE DATASET
# Source: tensorflow_datasets — 'plant_village'
# Only corn, potato, and tomato subsets are used (17 classes)
#
# NOTE:
# tensorflow-datasets 4.9.9 is incompatible with protobuf 7.x (required by
# TF 2.21). The TFDS builder crashes when reading dataset_info.json from the
# local cache. To address this, on each run, check for local cache and use if
# present, otherwise download dataset
# =============================================================================

# Labels in TFDS that correspond to the paper's 17 corn/potato/tomato classes
PLANTVILLAGE_KEEP_LABELS = [
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Potato___Early_Blight",
    "Potato___Late_Blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_Blight",
    "Tomato___Late_Blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Helper image preprocessing function which resizes and normalizes to [0, 1]
def preprocess_image(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Path to the downloaded PlantVillage tfrecords
PLANTVILLAGE_CACHE_DIR = os.path.join(TFDS_DATA_DIR, "plant_village", "1.0.2")

# Check for existence of cached dataset to prevent crash (See above note)
def _plantvillage_cache_exists():
    if not os.path.exists(PLANTVILLAGE_CACHE_DIR):
        return False
    files = [f for f in os.listdir(PLANTVILLAGE_CACHE_DIR)
             if f.endswith(".tfrecord-00000-of-00008")]
    return len(files) > 0

# Load plantvillage dataset from local cache (See above note)
def _load_plantvillage_from_tfrecords():
    # Read label names from the cache
    labels_path = os.path.join(PLANTVILLAGE_CACHE_DIR, "label.labels.txt")
    with open(labels_path, "r") as f:
        label_names = [line.strip() for line in f.readlines()]

    # Get indices of the 17 classes of the subset
    keep_indices = [
        i for i, name in enumerate(label_names)
        if name in PLANTVILLAGE_KEEP_LABELS
    ]
    keep_indices_tensor = tf.constant(keep_indices, dtype=tf.int64)

    # Remap original label indices to 0-indexed range for 17-class subset
    old_to_new = {old: new for new, old in enumerate(keep_indices)}

    # tfrecord feature description matching plant_village schema
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    # Parse local record data into image and label
    def parse_example(serialized):
        features = tf.io.parse_single_example(serialized, feature_description)
        image = tf.image.decode_jpeg(features["image"], channels=3)
        label = features["label"]
        return image, label

    # Filter images to include only labels from the subset
    def filter_fn(image, label):
        return tf.reduce_any(tf.equal(label, keep_indices_tensor))

    # Fix label index on filtered samples
    def remap_label(image, label):
        new_label = tf.py_function(
            lambda l: old_to_new[int(l)], [label], tf.int64
        )
        new_label.set_shape(())
        return image, new_label

    # Load all 8 tfrecord shards from cache
    tfrecord_files = sorted(tf.io.gfile.glob(
        os.path.join(PLANTVILLAGE_CACHE_DIR, "plant_village-train.tfrecord*")
    ))

    # Load full dataset from shards
    ds = tf.data.TFRecordDataset(tfrecord_files)
    # Parse imgs and lbls
    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    # Filter to subset
    ds = ds.filter(filter_fn)
    # Fix label index for new subset count
    ds = ds.map(remap_label, num_parallel_calls=tf.data.AUTOTUNE)
    # Preprocess images
    ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    return ds, label_names

# Loads the raw plantvillage dataset without batching or shuffling
# Attempts to load from cache; if no cache, falls back to download through TFDS
def load_plantvillage_full():
    print("Loading PlantVillage dataset (full)...")

    if _plantvillage_cache_exists():
        print("  Cache found — reading tfrecords directly")
        ds, label_names = _load_plantvillage_from_tfrecords()

        KNOWN_FILTERED_COUNT = 15403
        print(f"  Using known sample count: {KNOWN_FILTERED_COUNT}")
        total = KNOWN_FILTERED_COUNT

    else:
        print("  No cache found — downloading via TFDS...")
        ds, info = tfds.load(
            "plant_village",
            split="train",
            with_info=True,
            as_supervised=True,
            shuffle_files=True,
        )

        label_names = info.features["label"].names

        # Get indices of subset classes
        keep_indices = [
            i for i, name in enumerate(label_names)
            if name in PLANTVILLAGE_KEEP_LABELS
        ]
        keep_indices_tensor = tf.constant(keep_indices, dtype=tf.int64)

        # Remap indices to subset of 17 classes
        old_to_new = {old: new for new, old in enumerate(keep_indices)}

        # Filter out any classes outside the subset
        def filter_fn(image, label):
            return tf.reduce_any(tf.equal(label, keep_indices_tensor))

        # Remap old indexes to new range
        def remap_label(image, label):
            new_label = tf.py_function(
                lambda l: old_to_new[int(l)], [label], tf.int64
            )
            new_label.set_shape(())
            return image, new_label

        # Filter out classes outside subset, update indices and preprocess samples
        ds = (
            ds
            .filter(filter_fn)
            .map(remap_label, num_parallel_calls=tf.data.AUTOTUNE)
            .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        )

        # Get count of total samples
        total = sum(1 for _ in ds)

    print(f"  PlantVillage FULL dataset size: {total}")
    return ds, total

# Loads plantvillage dataset and performs shuffled train/validate split
def load_plantvillage():
    # Load raw dataset
    ds, total = load_plantvillage_full()

    # Get train/validate size
    train_size = int(total * TRAIN_SPLIT)

    # Create and shuffle with a stable ordering for clean take/skip split
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=total, seed=SEED, reshuffle_each_iteration=False)

    # Split dataset into train and validate sets
    # Inner shuffle on reorders training samples each epoch independently
    train_ds = (
        ds.take(train_size)
        .shuffle(buffer_size=train_size, seed=SEED, reshuffle_each_iteration=True)
        .repeat()
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        ds.skip(train_size)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    print(f"  PlantVillage — Total: {total} | Train: {train_size} | Val: {total - train_size}")
    return train_ds, val_ds


# =============================================================================
# RICE DATASET
# Source: https://data.mendeley.com/datasets/fwcj7stb8r/1
# Expected folder structure:
#   rice_data/
#     bacterial_blight/   (1584 images)
#     blast/              (1440 images)
#     brown_spot/         (1600 images)
#     tungro/             (1308 images)
# =============================================================================

RICE_DATA_DIR = "./data/rice_data"

# Load rice dataset locally
def load_rice():
    if not os.path.exists(RICE_DATA_DIR):
        raise FileNotFoundError(
            f"Rice dataset not found at '{RICE_DATA_DIR}'.\n"
        )

    print("Loading Rice disease dataset from disk...")
    full_ds = tf.keras.utils.image_dataset_from_directory(
        RICE_DATA_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=None,
        shuffle=True,
        seed=SEED,
        label_mode="int",
    )

    # Normalize pixel values to [0, 1]
    def normalize(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    # Materialize and shuffle with a stable ordering for clean take/skip split
    full_ds = full_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    full_ds = full_ds.cache()

    # Get total samples
    total = sum(1 for _ in full_ds)
    # Calculate train/validate split
    train_size = int(total * TRAIN_SPLIT)
    # Shuffle dataset
    full_ds = full_ds.shuffle(buffer_size=total, seed=SEED, reshuffle_each_iteration=False)

    # Train/validation split of dataset
    # Inner shuffle reorders training samples each epoch independently
    train_ds = (
        full_ds.take(train_size)
        .shuffle(buffer_size=train_size, seed=SEED, reshuffle_each_iteration=True)
        .repeat()
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = full_ds.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print(f"  Rice — Total: {total} | Train: {train_size} | Val: {total - train_size}")
    return train_ds, val_ds


# =============================================================================
# CASSAVA DATASET
# Source: https://www.kaggle.com/competitions/cassava-disease/data
# Expected folder structure:
#   cassava_data/
#     Healthy/                        (316 images)
#     Cassava_Bacterial_Blight/       (466 images)
#     Cassava_Brown_Streak_Disease/   (1443 images)
#     Cassava_Green_Mite/             (773 images)
#     Cassava_Mosaic_Disease/         (2658 images)
# =============================================================================

CASSAVA_DATA_DIR = "./data/cassava_data"

# Load cassava dataset locally
def load_cassava():
    if not os.path.exists(CASSAVA_DATA_DIR):
        raise FileNotFoundError(
            f"Cassava dataset not found at '{CASSAVA_DATA_DIR}'.\n"
        )

    print("Loading Cassava disease dataset from disk...")
    full_ds = tf.keras.utils.image_dataset_from_directory(
        CASSAVA_DATA_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=None,
        shuffle=True,
        seed=SEED,
        label_mode="int",
    )

    # Normalize pixel values to [0, 1]
    def normalize(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    # Materialize and shuffle with a stable ordering for clean take/skip split
    full_ds = full_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    full_ds = full_ds.cache()
    # Calculate total samples
    total = sum(1 for _ in full_ds)
    # Calculate train/validate split
    train_size = int(total * TRAIN_SPLIT)
    # Shuffle dataset
    full_ds = full_ds.shuffle(buffer_size=total, seed=SEED, reshuffle_each_iteration=False)

    # Train/validation split of dataset
    # Inner shuffle on train re-orders training samples each epoch independently
    train_ds = (
        full_ds.take(train_size)
        .shuffle(buffer_size=train_size, seed=SEED, reshuffle_each_iteration=True)
        .repeat()
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = full_ds.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print(f"  Cassava — Total: {total} | Train: {train_size} | Val: {total - train_size}")
    return train_ds, val_ds

# Load data with second fold shuffle for fast validation
def load_plantvillage_fold2():
    # Load dataset
    ds, total = load_plantvillage_full()

    # Get train/val split size
    train_size = int(total * TRAIN_SPLIT)

    # Cache and suffle dataset
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=total, seed=SEED + 1, reshuffle_each_iteration=False)

    # Split into train/validation sets
    train_ds = (
        ds.take(train_size)
        .shuffle(buffer_size=train_size, seed=SEED + 1, reshuffle_each_iteration=True)
        .repeat()
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = ds.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print(f"  PlantVillage Fold 2 — Total: {total} | Train: {train_size} | Val: {total - train_size}")
    return train_ds, val_ds

# Load data with second fold shuffle for fast validation
def load_rice_fold2():
    # Load dataset
    full_ds = tf.keras.utils.image_dataset_from_directory(
        RICE_DATA_DIR, image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=None, shuffle=True, seed=SEED + 1, label_mode="int",
    )

    # Normalize image to 0,1
    def normalize(image, label):
        return tf.cast(image, tf.float32) / 255.0, label
    
    full_ds = full_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    full_ds = full_ds.cache()

    # Train/Validate split
    total = sum(1 for _ in full_ds)
    train_size = int(total * TRAIN_SPLIT)
    # Shuffle dataset
    full_ds = full_ds.shuffle(buffer_size=total, seed=SEED + 1, reshuffle_each_iteration=False)

    train_ds = (
        full_ds.take(train_size)
        .shuffle(buffer_size=train_size, seed=SEED + 1, reshuffle_each_iteration=True)
        .repeat()
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = full_ds.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print(f"  Rice Fold 2 — Total: {total} | Train: {train_size} | Val: {total - train_size}")
    return train_ds, val_ds

# Load data with second fold shuffle for fast validation
def load_cassava_fold2():
    # Load dataset
    full_ds = tf.keras.utils.image_dataset_from_directory(
        CASSAVA_DATA_DIR, image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=None, shuffle=True, seed=SEED + 1, label_mode="int",
    )
    # Normalize values to range of 0 to 1
    def normalize(image, label):
        return tf.cast(image, tf.float32) / 255.0, label
    full_ds = full_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    full_ds = full_ds.cache()

    # Train/validate split
    total = sum(1 for _ in full_ds)
    train_size = int(total * TRAIN_SPLIT)
    full_ds = full_ds.shuffle(buffer_size=total, seed=SEED + 1, reshuffle_each_iteration=False)
    train_ds = (
        full_ds.take(train_size)
        .shuffle(buffer_size=train_size, seed=SEED + 1, reshuffle_each_iteration=True)
        .repeat()
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = full_ds.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print(f"  Cassava Fold 2 — Total: {total} | Train: {train_size} | Val: {total - train_size}")
    return train_ds, val_ds

# Validate dataset integrity
def validate_dataset(dataset, name, num_classes):
    for images, labels in dataset.take(1):
        print(f"\n[{name}] Batch check:")
        print(f"  Image batch shape: {images.shape}")
        print(f"  Label batch shape: {labels.shape}")
        print(f"  Pixel range: [{images.numpy().min():.2f}, {images.numpy().max():.2f}]")
        print(f"  Label range: [{labels.numpy().min()}, {labels.numpy().max()}]")
        assert images.shape[1:] == (IMG_SIZE, IMG_SIZE, 3), "Unexpected image shape!"
        assert labels.numpy().max() < num_classes, "Label index out of range!"
        print(f"SUCCESS: {name} is valid.")

# Main Block
if __name__ == "__main__":
    print("=" * 60)
    print("TensorFlow version:", tf.__version__)
    print("GPU available:", tf.config.list_physical_devices("GPU"))
    print("=" * 60)

    # Load PlantVillage
    pv_train, pv_val = load_plantvillage()
    validate_dataset(pv_train, "PlantVillage", NUM_CLASSES["plantvillage"])

    # Load Rice disease
    rice_train, rice_val = load_rice()
    validate_dataset(rice_train, "Rice", NUM_CLASSES["rice"])

    # Load Cassava
    cassava_train, cassava_val = load_cassava()
    validate_dataset(cassava_train, "Cassava", NUM_CLASSES["cassava"])

    print("\nSUCCESS: Data loading step complete. Proceed to model_02.py")


# =============================================================================
# REPLICATION GAPS
# =============================================================================
#
# 1. Batch size not stated in the paper, using 16 as default
#
# 2. Paper states Adam but gives no learning rate, decay, or schedule, default
#    starting value of 1e-3
#
# 3. Droout mentioned in the paper but never quantified, 0.5 assumed as default
#
# 4. The feature keys "image" and "label" are inferred from the standard TFDS plant_village schema.
#    If parsing fails on a fresh download, inspect the tfrecord with:
#       import tensorflow as tf
#       raw = next(iter(tf.data.TFRecordDataset([<path_to_shard>])))
#       print(tf.train.Example.FromString(raw.numpy()))
#    and update feature_description in _load_plantvillage_from_tfrecords().
#
# =============================================================================