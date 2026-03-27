"""
Replication - Data Loading
"""

import os
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
os.environ["TFDS_DATA_DIR"] = "/mnt/HDD T4T/tensorflow_datasets"

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


# Load PlantVillage Dataset
# Source: tensorflow_datasets — 'plant_village'
# Only corn, potato, and tomato subsets are used

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

# Load PlantVillage via TFDS, filter to include only corn/potato/tomato
def load_plantvillage():
    print("Loading PlantVillage dataset via TFDS...")
    # Load data with shuffle
    ds, info = tfds.load(
        "plant_village",
        split="train",
        with_info=True,
        as_supervised=True,
        shuffle_files=True,
    )

    # Get labels and indicies of kept features (corn, tomato, potato subset)
    label_names = info.features["label"].names
    keep_indices = [
        i for i, name in enumerate(label_names)
        if name in PLANTVILLAGE_KEEP_LABELS
    ]

    # Fix index fragmentation by remapping to 0-indexed labels for 17-class subset
    old_to_new = {old: new for new, old in enumerate(keep_indices)}

    # Filter function to which returns true if a label's index is within the kept classes
    def filter_fn(image, label):
        return tf.reduce_any(tf.equal(label, keep_indices))

    # Mapping function to update each label's index to the 0-17 range
    def remap_label(image, label):
        new_label = tf.py_function(
            lambda l: old_to_new[int(l)], [label], tf.int64
        )
        return image, new_label

    # Preprocess each image to resize and normalize 
    def preprocess(image, label):
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    # Apply helper functions to filter the dataset, update filtered sample indicies, and preprocess filtered samples
    ds = ds.filter(filter_fn).map(remap_label).map(
        preprocess, num_parallel_calls=tf.data.AUTOTUNE
    )

    # Count filtered dataset size for train/test/val splitting
    total = info.splits["train"].num_examples
    # Get actual split size based on filtered sample count
    train_size = int(total * TRAIN_SPLIT)

    # Shuffle samples
    ds = ds.shuffle(buffer_size=total, seed=SEED)
    # Split training data
    train_ds = ds.take(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    # Split validation data
    val_ds = ds.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print(f"  PlantVillage — Total: {total} | Train: {train_size} | Val: {total - train_size}")
    return train_ds, val_ds


# Load Rice disease dataset manually
# Source: https://data.mendeley.com/datasets/fwcj7stb8r/1
# Expected folder structure:
#   rice_data/
#     bacterial_blight/   (1584 images)
#     blast/              (1440 images)
#     brown_spot/         (1600 images)
#     tungro/             (1308 images)

RICE_DATA_DIR = "./data/rice_data"

def load_rice():
    # Ensure dataset file exists locally
    if not os.path.exists(RICE_DATA_DIR):
        raise FileNotFoundError(
            f"Rice dataset not found at '{RICE_DATA_DIR}'.\n"
        )

    print("Loading Rice disease dataset from disk...")
    # Load dataset with no batching for manual splitting later
    full_ds = tf.keras.utils.image_dataset_from_directory(
        RICE_DATA_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=None,
        shuffle=True,
        seed=SEED,
        label_mode="int",
    )

    # Helper function to normalize samples
    def normalize(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    # Apply preprocessing to dataset
    full_ds = full_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

    # Get total number of samples
    total = sum(1 for _ in full_ds)
    # Get actual train/test split from sample size
    train_size = int(total * TRAIN_SPLIT)

    # Split dataset into training and validation sets with batching applied
    train_ds = full_ds.take(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = full_ds.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print(f"  Rice — Total: {total} | Train: {train_size} | Val: {total - train_size}")
    return train_ds, val_ds


# Load Cassava disease dataset manually
# Source: https://www.kaggle.com/competitions/cassava-disease/data
# Expected folder structure:
#   cassava_data/
#     Healthy/                        (316 images)
#     Cassava_Bacterial_Blight/       (466 images)
#     Cassava_Brown_Streak_Disease/   (1443 images)
#     Cassava_Green_Mite/             (773 images)
#     Cassava_Mosaic_Disease/         (2658 images)

CASSAVA_DATA_DIR = "./data/cassava_data"

def load_cassava():
    # Ensure dataset file exists locally
    if not os.path.exists(CASSAVA_DATA_DIR):
        raise FileNotFoundError(
            f"Cassava dataset not found at '{CASSAVA_DATA_DIR}'.\n"
        )

    print("Loading Cassava disease dataset from disk...")
    # Load with no batching for manual split downstream
    full_ds = tf.keras.utils.image_dataset_from_directory(
        CASSAVA_DATA_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=None,
        shuffle=True,
        seed=SEED,
        label_mode="int",
    )

    # Normalize samples to 0-255 range
    def normalize(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    # Preprocess samples
    full_ds = full_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

    # Get total sample count
    total = sum(1 for _ in full_ds)
    # Get actual train/validate split size
    train_size = int(total * TRAIN_SPLIT)

    # Split into train and validate datasets with batching applied
    train_ds = full_ds.take(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = full_ds.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

# =============================================================================
# QUICK SANITY CHECK — Run to verify dataset loading before proceeding
# =============================================================================

def sanity_check(dataset, name, num_classes):
    """Checks a single batch for expected shapes and label range."""
    for images, labels in dataset.take(1):
        print(f"\n[{name}] Batch check:")
        print(f"  Image batch shape: {images.shape}")   # (32, 256, 256, 3)
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

    # --- PlantVillage (auto-downloads via TFDS) ---
    pv_train, pv_val = load_plantvillage()
    sanity_check(pv_train, "PlantVillage", NUM_CLASSES["plantvillage"])
    # Statistics require an unbatched, label-only pass — reload unbatched
    pv_unbatched = pv_train.unbatch()
    pv_unbatched = pv_unbatched.unbatch()

    # --- Rice (manual download required) ---
    rice_train, rice_val = load_rice()
    sanity_check(rice_train, "Rice", NUM_CLASSES["rice"])
    rice_unbatched = tf.keras.utils.image_dataset_from_directory(
        RICE_DATA_DIR, image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=None, label_mode="int"
    )

    # --- Cassava (manual download required) ---
    cassava_train, cassava_val = load_cassava()
    sanity_check(cassava_train, "Cassava", NUM_CLASSES["cassava"])
    cassava_unbatched = tf.keras.utils.image_dataset_from_directory(
        CASSAVA_DATA_DIR, image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=None, label_mode="int"
    )

    print("\n✅ Data loading step complete. Proceed to 02_model.py")


# =============================================================================
# ⚠️ OUTSTANDING GAPS — To be resolved before/during training
# =============================================================================
#
# 1. BATCH SIZE — Not stated in the paper. Using 32 as default.
#    Adjust in the GLOBAL CONFIG section if needed.
#
# 2. LEARNING RATE / OPTIMIZER SCHEDULE — Paper states Adam but gives
#    no learning rate, decay, or schedule. Starting with 1e-3 in training step.
#
# 3. DROPOUT RATE — Mentioned in the paper but never quantified.
#    Will use 0.5 in model definition; tune if validation accuracy is poor.
#
# 4. DATA AUGMENTATION — Not described in the paper at all.
#    No augmentation is applied here to stay faithful to the paper.
#    If Cassava accuracy falls significantly short of 76.59%, consider
#    adding flips/rotations as a tuning step.
#
# 5. RICE & CASSAVA DOWNLOAD — Must be done manually (no TFDS support).
#    Rice:    https://data.mendeley.com/datasets/fwcj7stb8r/1
#    Cassava: https://www.kaggle.com/competitions/cassava-leaf-disease-classification
#    Update RICE_DATA_DIR and CASSAVA_DATA_DIR paths after downloading.
#
# 6. PLANTVILLAGE LABEL MAPPING — TFDS label string names must map exactly
#    to the 17 classes in Table 4. Verify label_names from TFDS info object
#    match PLANTVILLAGE_KEEP_LABELS list above before training.
#
# =============================================================================