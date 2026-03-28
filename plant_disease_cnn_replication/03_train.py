"""
Replication - Model Training
Hassan & Maji (2022): "Plant Disease Identification Using a Novel Convolutional Neural Network"
IEEE Access, DOI: 10.1109/ACCESS.2022.3141371

Training protocol:
  - Optimizer: Adam (paper states Adam, no LR given — see GAPS)
  - Epochs: 50
  - Split: 80% train / 20% validation
  - K-fold: 5-fold cross-validation for robustness evaluation
  - Metrics: accuracy, precision, recall, F1-score, confusion matrix

Run order: 01_data.py → 02_model.py → this script
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf

# GPU memory growth — must be set before any other TF operations
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Import project modules
from data_loading import (          # 01_data.py
    load_plantvillage, load_rice, load_cassava,
    NUM_CLASSES, EPOCHS, BATCH_SIZE, SEED, N_FOLDS,
    RICE_DATA_DIR, CASSAVA_DATA_DIR,
)
from model import build_model       # 02_model.py

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

tf.random.set_seed(SEED)
np.random.seed(SEED)

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

# GAP: Paper states Adam but gives no learning rate, schedule, or decay.
# 1e-3 is Adam's default and the most common starting point in the literature.
LEARNING_RATE = 1e-3

DROPOUT_RATE = 0.5      # GAP: not stated in paper — standard default

OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# HELPERS
# =============================================================================

def make_callbacks(dataset_name: str, fold: int = None) -> list:
    """
    Returns a standard callback set for one training run.

    ModelCheckpoint saves the best validation accuracy weights.
    EarlyStopping is NOT used — the paper trains for exactly 50 epochs.
    """
    tag = dataset_name if fold is None else f"{dataset_name}_fold{fold}"
    ckpt_path = os.path.join(OUTPUT_DIR, f"best_{tag}.keras")

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(OUTPUT_DIR, f"log_{tag}.csv")
        ),
    ]


def compile_model(model: tf.keras.Model) -> tf.keras.Model:
    """
    Compiles model with Adam and sparse categorical crossentropy.
    Labels from the data pipeline are integer-encoded (not one-hot),
    so sparse_categorical_crossentropy is appropriate.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, dataset_name: str, fold: int = None):
    """Saves accuracy and loss curves matching Figures 7–9 in the paper."""
    tag = dataset_name if fold is None else f"{dataset_name}_fold{fold}"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title(f"{dataset_name} — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title(f"{dataset_name} — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"curves_{tag}.png"), dpi=150)
    plt.close()


def evaluate_and_report(model, val_ds, dataset_name: str,
                         num_classes: int, fold: int = None):
    """
    Runs inference on the validation set and saves:
      - Classification report (precision, recall, F1 per class)
      - Confusion matrix heatmap
      - Summary JSON with scalar metrics
    """
    tag = dataset_name if fold is None else f"{dataset_name}_fold{fold}"

    # Collect ground-truth labels and predictions
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Classification report (Table 8 equivalent)
    report = classification_report(y_true, y_pred, output_dict=True)
    report_text = classification_report(y_true, y_pred)
    print(f"\n[{tag}] Classification Report:\n{report_text}")

    with open(os.path.join(OUTPUT_DIR, f"report_{tag}.txt"), "w") as f:
        f.write(report_text)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, num_classes), max(5, num_classes - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{dataset_name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"cm_{tag}.png"), dpi=150)
    plt.close()

    # Scalar summary
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    summary = {
        "dataset": dataset_name,
        "fold": fold,
        "accuracy": float(accuracy),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
    }
    with open(os.path.join(OUTPUT_DIR, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {summary['macro_precision']:.4f}")
    print(f"  Recall:    {summary['macro_recall']:.4f}")
    print(f"  F1-score:  {summary['macro_f1']:.4f}")

    return summary


# =============================================================================
# MAIN TRAINING RUN (Table 7 equivalent — single 80/20 split)
# =============================================================================

def train_dataset(dataset_name: str, train_ds, val_ds, num_classes: int):
    """Full 50-epoch training run on a single dataset."""
    print(f"\n{'='*60}")
    print(f"Training on {dataset_name} ({num_classes} classes)")
    print(f"{'='*60}")

    model = build_model(num_classes=num_classes, dropout_rate=DROPOUT_RATE)
    model = compile_model(model)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=make_callbacks(dataset_name),
        verbose=1,
    )

    plot_history(history, dataset_name)

    # Load best checkpoint for evaluation
    best_path = os.path.join(OUTPUT_DIR, f"best_{dataset_name}.keras")
    if os.path.exists(best_path):
        model.load_weights(best_path)

    summary = evaluate_and_report(model, val_ds, dataset_name, num_classes)
    return summary


# =============================================================================
# K-FOLD CROSS-VALIDATION (Table 9 equivalent)
#
# GAP: The paper uses 5-fold CV but loads the full dataset each fold.
# The data loaders in 01_data.py return pre-split tf.data pipelines,
# so for k-fold we need access to the raw (unbatched, unshuffled) dataset.
# The fold loop below reconstructs splits from the raw directory loaders.
#
# PlantVillage k-fold is omitted here because the tfrecord loading path
# in 01_data.py does not expose a resplittable dataset object easily.
# It is included for Rice and Cassava where image_dataset_from_directory
# is used and resplitting is straightforward.
# =============================================================================

def kfold_dataset(dataset_name: str, data_dir: str, num_classes: int):
    """
    Runs 5-fold cross-validation on a directory-based dataset.
    Reproduces Table 9 in the paper.
    """
    print(f"\n{'='*60}")
    print(f"5-Fold CV — {dataset_name}")
    print(f"{'='*60}")

    # Load full dataset (unbatched) for manual splitting
    full_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(256, 256),
        batch_size=None,
        shuffle=True,
        seed=SEED,
        label_mode="int",
    )

    def normalize(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    full_ds = full_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

    total = sum(1 for _ in full_ds)
    fold_size = total // N_FOLDS

    fold_summaries = []

    for fold in range(N_FOLDS):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

        val_start = fold * fold_size
        val_end = val_start + fold_size

        # Validation: the fold's slice; Training: everything else
        val_ds = full_ds.skip(val_start).take(fold_size)
        train_ds = full_ds.take(val_start).concatenate(
            full_ds.skip(val_end)
        )

        train_ds = train_ds.shuffle(
            buffer_size=total - fold_size, seed=SEED
        ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        model = build_model(num_classes=num_classes, dropout_rate=DROPOUT_RATE)
        model = compile_model(model)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=make_callbacks(dataset_name, fold=fold + 1),
            verbose=1,
        )

        plot_history(history, dataset_name, fold=fold + 1)

        best_path = os.path.join(OUTPUT_DIR, f"best_{dataset_name}_fold{fold+1}.keras")
        if os.path.exists(best_path):
            model.load_weights(best_path)

        summary = evaluate_and_report(
            model, val_ds, dataset_name, num_classes, fold=fold + 1
        )
        fold_summaries.append(summary)

    # Aggregate across folds (Table 9 equivalent)
    accuracies = [s["accuracy"] for s in fold_summaries]
    print(f"\n[{dataset_name}] K-Fold Results:")
    print(f"  Per-fold accuracies: {[f'{a:.4f}' for a in accuracies]}")
    print(f"  Mean: {np.mean(accuracies):.4f} | Std: {np.std(accuracies):.4f}")
    print(f"  Range: {np.min(accuracies):.4f} – {np.max(accuracies):.4f}")

    with open(os.path.join(OUTPUT_DIR, f"kfold_{dataset_name}.json"), "w") as f:
        json.dump({
            "dataset": dataset_name,
            "fold_summaries": fold_summaries,
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "min_accuracy": float(np.min(accuracies)),
            "max_accuracy": float(np.max(accuracies)),
        }, f, indent=2)

    return fold_summaries


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TensorFlow:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))
    print("=" * 60)

    all_summaries = {}

    # --- Main training runs (Table 7) ---
    pv_train, pv_val = load_plantvillage()
    all_summaries["plantvillage"] = train_dataset(
        "plantvillage", pv_train, pv_val, NUM_CLASSES["plantvillage"]
    )

    rice_train, rice_val = load_rice()
    all_summaries["rice"] = train_dataset(
        "rice", rice_train, rice_val, NUM_CLASSES["rice"]
    )

    cassava_train, cassava_val = load_cassava()
    all_summaries["cassava"] = train_dataset(
        "cassava", cassava_train, cassava_val, NUM_CLASSES["cassava"]
    )

    # --- K-Fold cross-validation (Table 9) ---
    kfold_dataset("rice", RICE_DATA_DIR, NUM_CLASSES["rice"])
    kfold_dataset("cassava", CASSAVA_DATA_DIR, NUM_CLASSES["cassava"])

    # --- Final summary across all datasets ---
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    paper_targets = {
        "plantvillage": 0.9939,
        "rice": 0.9966,
        "cassava": 0.7659,
    }
    for name, summary in all_summaries.items():
        target = paper_targets[name]
        delta = summary["accuracy"] - target
        print(f"  {name:12s}  Achieved: {summary['accuracy']:.4f}  "
              f"Target: {target:.4f}  Delta: {delta:+.4f}")

    with open(os.path.join(OUTPUT_DIR, "final_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n✅ Training complete. Results saved to ./results/")


# =============================================================================
# ⚠️  OUTSTANDING GAPS
# =============================================================================
#
# 1. LEARNING RATE — Paper states Adam but gives no learning rate, decay,
#    or schedule. Using 1e-3 (Adam default). If validation accuracy plateaus
#    early, try 3e-4 or add ReduceLROnPlateau.
#
# 2. BATCH SIZE — Not stated. Using 32 (standard default from 01_data.py).
#
# 3. DROPOUT RATE — Not stated. Using 0.5.
#
# 4. DATA AUGMENTATION — Not described in the paper. None applied here.
#    If replicated accuracy is below target (especially on Cassava), note
#    that augmentation may have been used implicitly.
#
# 5. PLANTVILLAGE K-FOLD — Omitted because the tfrecord-based loader in
#    01_data.py does not expose a cleanly resplittable dataset. The paper
#    reports k-fold results for all three datasets (Table 9). To add this,
#    refactor load_plantvillage() to return an unbatched ds and total count,
#    then apply the same fold-slicing logic used for Rice and Cassava.
#
# 6. CLASS WEIGHTS — The Cassava dataset is imbalanced (the paper notes
#    this explicitly). No class weighting is applied here, matching the
#    paper's apparent approach. If accuracy is poor, add class_weight to
#    model.fit() as a post-hoc experiment.
#
# =============================================================================