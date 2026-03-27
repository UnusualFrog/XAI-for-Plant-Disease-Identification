"""
Replication - Data Loading
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
import tensorflow_datasets as tfds

layers = tf.keras.layers
models = tf.keras.models
Model = tf.keras.Model
Adam = tf.keras.optimizers.Adam
to_categorical = tf.keras.utils.to_categorical

# Use HDD for TFDS
TFDS_DATA_DIR = "/mnt/HDD T4T/tensorflow_datasets"
os.environ["TFDS_DATA_DIR"] = TFDS_DATA_DIR

# Global Random Seed for Reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# Global Hyperparameters
IMG_SIZE = 256
BATCH_SIZE = 32       # GAP: Not stated in paper, 32 is a standard default
EPOCHS = 50
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2
N_FOLDS = 5

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
# NOTE ON LOADING STRATEGY:
# tensorflow-datasets 4.9.9 is incompatible with protobuf 7.x (required by
# TF 2.21). The TFDS builder crashes when reading dataset_info.json from the
# local cache. To work around this, we detect whether the tfrecords are already
# downloaded and parse them directly via tf.data, bypassing the TFDS builder
# entirely. On a fresh environment with no cache, we fall back to tfds.load
# which will download and write the cache once.
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

# Path to the downloaded PlantVillage tfrecords
PLANTVILLAGE_CACHE_DIR = os.path.join(TFDS_DATA_DIR, "plant_village", "1.0.2")


def _plantvillage_cache_exists():
    """Returns True if the tfrecord shards are already on disk."""
    if not os.path.exists(PLANTVILLAGE_CACHE_DIR):
        return False
    files = [f for f in os.listdir(PLANTVILLAGE_CACHE_DIR)
             if f.endswith(".tfrecord-00000-of-00008")]
    return len(files) > 0


def _load_plantvillage_from_tfrecords():
    """
    Reads PlantVillage directly from tfrecord shards, bypassing the TFDS
    builder and its protobuf-incompatible metadata reader.

    The tfrecord feature schema for plant_village is:
        image/encoded: raw JPEG bytes
        image/filename: string
        label: int64
    Label names are read from label.labels.txt in the cache directory.
    """
    # Read label names from the cache — this is a plain text file, no protobuf
    labels_path = os.path.join(PLANTVILLAGE_CACHE_DIR, "label.labels.txt")
    with open(labels_path, "r") as f:
        label_names = [line.strip() for line in f.readlines()]

    # Get indices of the 17 classes we want to keep
    keep_indices = [
        i for i, name in enumerate(label_names)
        if name in PLANTVILLAGE_KEEP_LABELS
    ]
    keep_indices_tensor = tf.constant(keep_indices, dtype=tf.int64)

    # Remap original label indices to 0-indexed range for our 17-class subset
    old_to_new = {old: new for new, old in enumerate(keep_indices)}

    # tfrecord feature description matching plant_village schema
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    def parse_example(serialized):
        features = tf.io.parse_single_example(serialized, feature_description)
        image = tf.image.decode_jpeg(features["image"], channels=3)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = tf.cast(image, tf.float32) / 255.0
        label = features["label"]
        return image, label

    def filter_fn(image, label):
        return tf.reduce_any(tf.equal(label, keep_indices_tensor))

    def remap_label(image, label):
        new_label = tf.py_function(
            lambda l: old_to_new[int(l)], [label], tf.int64
        )
        new_label.set_shape(())
        return image, new_label

    # Load all 8 tfrecord shards
    tfrecord_files = sorted(tf.io.gfile.glob(
        os.path.join(PLANTVILLAGE_CACHE_DIR, "plant_village-train.tfrecord*")
    ))

    ds = tf.data.TFRecordDataset(tfrecord_files)
    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(filter_fn)
    ds = ds.map(remap_label, num_parallel_calls=tf.data.AUTOTUNE)

    return ds, label_names


def load_plantvillage():
    print("Loading PlantVillage dataset...")

    if _plantvillage_cache_exists():
        # Fast path: read directly from tfrecords, skip broken TFDS builder
        print("  Cache found — reading tfrecords directly (bypassing TFDS builder)...")
        ds, label_names = _load_plantvillage_from_tfrecords()

        # Count filtered samples for splitting
        print("  Counting filtered samples (corn/potato/tomato only)...")
        total = sum(1 for _ in ds)

    else:
        # First-run path: no cache yet, use tfds.load to download
        print("  No cache found — downloading via TFDS (first run only)...")
        ds, info = tfds.load(
            "plant_village",
            split="train",
            with_info=True,
            as_supervised=True,
            shuffle_files=True,
        )
        label_names = info.features["label"].names
        keep_indices = [
            i for i, name in enumerate(label_names)
            if name in PLANTVILLAGE_KEEP_LABELS
        ]
        keep_indices_tensor = tf.constant(keep_indices, dtype=tf.int64)
        old_to_new = {old: new for new, old in enumerate(keep_indices)}

        def filter_fn(image, label):
            return tf.reduce_any(tf.equal(label, keep_indices_tensor))

        def remap_label(image, label):
            new_label = tf.py_function(
                lambda l: old_to_new[int(l)], [label], tf.int64
            )
            new_label.set_shape(())
            return image, new_label

        def preprocess(image, label):
            image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
            image = tf.cast(image, tf.float32) / 255.0
            return image, label

        ds = ds.filter(filter_fn).map(remap_label).map(
            preprocess, num_parallel_calls=tf.data.AUTOTUNE
        )
        total = info.splits["train"].num_examples

    train_size = int(total * TRAIN_SPLIT)

    ds = ds.shuffle(buffer_size=total, seed=SEED)
    train_ds = ds.take(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = ds.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

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

    def normalize(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    full_ds = full_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

    total = sum(1 for _ in full_ds)
    train_size = int(total * TRAIN_SPLIT)

    train_ds = full_ds.take(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
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

    def normalize(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    full_ds = full_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

    total = sum(1 for _ in full_ds)
    train_size = int(total * TRAIN_SPLIT)

    train_ds = full_ds.take(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = full_ds.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print(f"  Cassava — Total: {total} | Train: {train_size} | Val: {total - train_size}")
    return train_ds, val_ds


# =============================================================================
# QUICK SANITY CHECK
# =============================================================================

def sanity_check(dataset, name, num_classes):
    """Checks a single batch for expected shapes and label range."""
    for images, labels in dataset.take(1):
        print(f"\n[{name}] Batch check:")
        print(f"  Image batch shape: {images.shape}")
        print(f"  Label batch shape: {labels.shape}")
        print(f"  Pixel range: [{images.numpy().min():.2f}, {images.numpy().max():.2f}]")
        print(f"  Label range: [{labels.numpy().min()}, {labels.numpy().max()}]")
        assert images.shape[1:] == (IMG_SIZE, IMG_SIZE, 3), "Unexpected image shape!"
        assert labels.numpy().max() < num_classes, "Label index out of range!"
        print(f"  ✅ {name} sanity check passed.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TensorFlow version:", tf.__version__)
    print("GPU available:", tf.config.list_physical_devices("GPU"))
    print("=" * 60)

    # --- PlantVillage ---
    pv_train, pv_val = load_plantvillage()
    sanity_check(pv_train, "PlantVillage", NUM_CLASSES["plantvillage"])

    # --- Rice ---
    rice_train, rice_val = load_rice()
    sanity_check(rice_train, "Rice", NUM_CLASSES["rice"])

    # --- Cassava ---
    cassava_train, cassava_val = load_cassava()
    sanity_check(cassava_train, "Cassava", NUM_CLASSES["cassava"])

    print("\n✅ Data loading step complete. Proceed to 02_model.py")


# =============================================================================
# ⚠️ OUTSTANDING GAPS
# =============================================================================
#
# 1. BATCH SIZE — Not stated in the paper. Using 32 as default.
#
# 2. LEARNING RATE / OPTIMIZER SCHEDULE — Paper states Adam but gives
#    no learning rate, decay, or schedule. Starting with 1e-3 in training step.
#
# 3. DROPOUT RATE — Mentioned in the paper but never quantified.
#    Will use 0.5 in model definition; tune if validation accuracy is poor.
#
# 4. DATA AUGMENTATION — Not described in the paper at all.
#    No augmentation applied to stay faithful to the paper.
#
# 5. PLANTVILLAGE TFRECORD SCHEMA — The feature keys "image" and "label"
#    are inferred from the standard TFDS plant_village schema. If parsing
#    fails on a fresh download, inspect the tfrecord with:
#       import tensorflow as tf
#       raw = next(iter(tf.data.TFRecordDataset([<path_to_shard>])))
#       print(tf.train.Example.FromString(raw.numpy()))
#    and update feature_description in _load_plantvillage_from_tfrecords().
#
# =============================================================================