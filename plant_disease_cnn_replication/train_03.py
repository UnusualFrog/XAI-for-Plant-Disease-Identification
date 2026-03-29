"""
Replication - Model Training


Training protocol:
  - Optimizer: Adam (paper states Adam, no LR given — see GAPS)
  - Epochs: 50
  - Split: 80% train / 20% validation
  - K-fold: 5-fold cross-validation for robustness evaluation
  - Metrics: accuracy, precision, recall, F1-score, confusion matrix

"""

import tempfile, os
# Disable cuDNN algorithm autotuning. The autotuner needs GPU scratch memory
# to benchmark kernel configs — this fails when the BFC pool is near its
# ceiling. Disabling it picks a safe default algorithm instead.
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
import json
import numpy as np
import matplotlib
# No output version
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf

import gc

# Handle gpu
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4000)]
        # 5.5GB of your 8GB: enough for training + k-fold XLA scratch,
        # leaving ~2.5GB for the OS, display driver, and CPU-side pr2efetch.
    )

# Import project modules
from data_loading_01 import (
    load_plantvillage_full, load_plantvillage, load_rice, load_cassava,
    NUM_CLASSES, EPOCHS, BATCH_SIZE, SEED, N_FOLDS,
    RICE_DATA_DIR, CASSAVA_DATA_DIR,
)
from model_02 import build_model

# Set global random seed
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Model Training Hyperparameters
# NOTE: Paper states Adam optimizer but gives no learning rate, schedule, or decay, 1e-3 used as default
LEARNING_RATE = 1e-3
# NOTE: dropout rate not stated, 0.5 used as default
DROPOUT_RATE = 0.5
# Directory to output results of training
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# FIX: Capped shuffle buffer to avoid OOM when filling buffer with full dataset;
# same cap applied consistently across all pipelines
SHUFFLE_BUFFER = 2000

# Helper functions

# FIX: Added monitor parameter so Cassava can use val_loss instead of val_accuracy
# due to class imbalance making val_accuracy an unreliable checkpoint signal
def make_callbacks(dataset_name, fold=None, monitor="val_accuracy"):
    # Set filename to dataset with current fold if available
    tag = dataset_name if fold is None else f"{dataset_name}_fold{fold}"
    # Set output file path for current dataset
    ckpt_path = os.path.join(OUTPUT_DIR, f"best_{tag}.keras")

    return [
        # Save model with best validation accuracy
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor=monitor,
            save_best_only=True,
            verbose=0,
        ),
        # Save training metrics to CSV
        tf.keras.callbacks.CSVLogger(
            os.path.join(OUTPUT_DIR, f"log_{tag}.csv")
        ),
    ]

# Complile model with Adam optimizer and sparse categorical crossentropy loss to optimize accuracy
def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# Plot acuracy and loss curves
def plot_history(history, dataset_name, fold = None):
    # Set filename to dataset name with current fold if available
    tag = dataset_name if fold is None else f"{dataset_name}_fold{fold}"
    # 1 row of 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot training vs. validation accuracy over epochs to first subplot
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title(f"{dataset_name} — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Plot training vs. validation loss over epochs to second subplot
    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title(f"{dataset_name} — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    # Save plots to file
    plt.savefig(os.path.join(OUTPUT_DIR, f"curves_{tag}.png"), dpi=150)
    plt.close()

# Evalute model on validation dataset to produce classification report, confusion matrix and json summary
def evaluate_and_report(model, val_ds, dataset_name, num_classes, fold = None):
    # Set filename to datasetnet with current fold if available
    tag = dataset_name if fold is None else f"{dataset_name}_fold{fold}"

    # Collect ground-truth labels and predictions
    # FIX: Collect predictions and labels in a single pass to avoid ordering
    # mismatch between two separate iterations over val_ds
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Classification report for precision, recall, F1 and accuracy
    report = classification_report(y_true, y_pred, output_dict=True)
    report_text = classification_report(y_true, y_pred)
    # Display classification report
    print(f"\n[{tag}] Classification Report:\n{report_text}")

    # Save classification report to file
    with open(os.path.join(OUTPUT_DIR, f"report_{tag}.txt"), "w") as f:
        f.write(report_text)

    # Save confusion matrix heatmap to file
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, num_classes), max(5, num_classes - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{dataset_name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"cm_{tag}.png"), dpi=150)
    plt.close()

    # Save scalar summary to json
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

# Train model on training dataset
def train_dataset(dataset_name, train_ds, val_ds, num_classes, train_size):
    print(f"\n{'='*60}")
    print(f"Training on {dataset_name} ({num_classes} classes)")
    print(f"{'='*60}")

    # Create blank model
    model = build_model(num_classes=num_classes, dropout_rate=DROPOUT_RATE)
    model = compile_model(model)

    # FIX: Use val_loss monitor for Cassava due to class imbalance
    monitor = "val_loss" if dataset_name == "cassava" else "val_accuracy"

    # Train model on training data
    history = model.fit(
        train_ds,
        steps_per_epoch=train_size,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=make_callbacks(dataset_name, monitor=monitor),
        verbose=1,
    )

    # Plot loss and accuracy
    plot_history(history, dataset_name)

    # Load best checkpoint for evaluation
    best_path = os.path.join(OUTPUT_DIR, f"best_{dataset_name}.keras")
    if os.path.exists(best_path):
        model = tf.keras.models.load_model(best_path)

    # Evaluate model and compare training metrics to validation metrics
    summary = evaluate_and_report(model, val_ds, dataset_name, num_classes)
    return summary

# Crossvalidate the cassavana or rice dataset with 5 fold K-fold crossvalidation
def kfold_local_dataset(dataset_name: str, data_dir: str, num_classes: int):
    KFOLD_BATCH_SIZE = 8
    print(f"\n{'='*60}")
    print(f"5-Fold CV — {dataset_name}")
    print(f"{'='*60}")

    # Load full dataset (unbatched) for manual splitting
    full_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(256, 256),
        batch_size=None,
        # FIX: shuffle=True required to distribute classes across folds;
        # shuffle=False loads in filesystem/alphabetical order causing each
        # fold to contain predominantly one class
        shuffle=True,
        seed=SEED,
        label_mode="int",
    )

    # Normalize feature values to range of 0-1
    def normalize(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    full_ds = full_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

    # Get count of samples for k-fold splititng, cache value to improve performance
    cache_path = os.path.join(tempfile.gettempdir(), f"{dataset_name}_kfold_cache")
    full_ds = full_ds.cache(cache_path)
    total = sum(1 for _ in full_ds)
    fold_size = total // N_FOLDS
    train_fold_size = (total - fold_size) // KFOLD_BATCH_SIZE

    full_ds = full_ds.shuffle(buffer_size=SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=False)

    fold_summaries = []

    for fold in range(N_FOLDS):
        if fold > 0:
            del model
            del train_ds, val_ds  # release pipeline references from previous fold
            gc.collect()

        val_start = fold * fold_size
        val_end = val_start + fold_size

        val_ds = full_ds.skip(val_start).take(fold_size)
        train_ds = full_ds.take(val_start).concatenate(full_ds.skip(val_end))

        # FIX: Use KFOLD_BATCH_SIZE in .batch() — was incorrectly using BATCH_SIZE
        train_ds = train_ds.shuffle(
            buffer_size=SHUFFLE_BUFFER, seed=SEED
        ).repeat().batch(KFOLD_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(KFOLD_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        model = build_model(num_classes=num_classes, dropout_rate=DROPOUT_RATE)
        model = compile_model(model)

        # FIX: Use val_loss monitor for Cassava due to class imbalance
        monitor = "val_loss" if dataset_name == "cassava" else "val_accuracy"

        # train model on training data
        history = model.fit(
            train_ds,
            steps_per_epoch=train_fold_size,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=make_callbacks(dataset_name, fold=fold + 1, monitor=monitor),
            verbose=1,
        )

        # plot training validation and accuracy
        plot_history(history, dataset_name, fold=fold + 1)

        # Load model from best performing fold
        best_path = os.path.join(OUTPUT_DIR, f"best_{dataset_name}_fold{fold+1}.keras")
        # Check if best performing model exists
        if os.path.exists(best_path):
            model = tf.keras.models.load_model(best_path)

        # Evalute model using best performing fold
        summary = evaluate_and_report(
            model, val_ds, dataset_name, num_classes, fold=fold + 1
        )
        fold_summaries.append(summary)

    # Aggregate across folds
    accuracies = [s["accuracy"] for s in fold_summaries]
    print(f"\n[{dataset_name}] K-Fold Results:")
    print(f"  Per-fold accuracies: {[f'{a:.4f}' for a in accuracies]}")
    print(f"  Mean: {np.mean(accuracies):.4f} | Std: {np.std(accuracies):.4f}")
    print(f"  Range: {np.min(accuracies):.4f} – {np.max(accuracies):.4f}")

    # Sve results to JSON
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

# Perform 5 fold k-fold cross validation on plantvillage
def kfold_plantvillage(num_classes):
    KFOLD_BATCH_SIZE = 8    
    print(f"\n{'='*60}")
    print("5-Fold CV — PlantVillage")
    print(f"{'='*60}")

    # Load full dataset
    full_ds, total = load_plantvillage_full()

    cache_path = os.path.join(tempfile.gettempdir(), "plantvillage_kfold_cache")
    full_ds = full_ds.cache(cache_path)

    # Get fold split size
    fold_size = total // N_FOLDS
    train_fold_size = (total - fold_size) // KFOLD_BATCH_SIZE
    fold_summaries = []

    full_ds = full_ds.shuffle(buffer_size=SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=False)

    # Loop through folds
    for fold in range(N_FOLDS):
        if fold > 0:
            del model
            del train_ds, val_ds  # release pipeline references from previous fold
            gc.collect()

        val_start = fold * fold_size
        val_end = val_start + fold_size

        val_ds = full_ds.skip(val_start).take(fold_size)
        train_ds = full_ds.take(val_start).concatenate(full_ds.skip(val_end))

        # FIX: Use KFOLD_BATCH_SIZE in .batch() — was incorrectly using BATCH_SIZE
        train_ds = train_ds.shuffle(
            buffer_size=SHUFFLE_BUFFER, seed=SEED
        ).repeat().batch(KFOLD_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(KFOLD_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        model = build_model(num_classes=num_classes, dropout_rate=DROPOUT_RATE)
        model = compile_model(model)

        # Train model on training fold data
        history = model.fit(
            train_ds,
            steps_per_epoch=train_fold_size,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=make_callbacks("plantvillage", fold=fold + 1),
            verbose=1,
        )

        # Plot accuracy and loss for current fold
        plot_history(history, "plantvillage", fold=fold + 1)

        # Load model from best performing fold
        best_path = os.path.join(
            OUTPUT_DIR, f"best_plantvillage_fold{fold+1}.keras"
        )
        if os.path.exists(best_path):
            model = tf.keras.models.load_model(best_path)

        # Evalute best performing model
        summary = evaluate_and_report(
            model, val_ds, "plantvillage",
            num_classes, fold=fold + 1
        )

        fold_summaries.append(summary)

    # FIX: Save aggregate k-fold results to JSON matching kfold_local_dataset output
    accuracies = [s["accuracy"] for s in fold_summaries]
    print(f"\n[plantvillage] K-Fold Results:")
    print(f"  Per-fold accuracies: {[f'{a:.4f}' for a in accuracies]}")
    print(f"  Mean: {np.mean(accuracies):.4f} | Std: {np.std(accuracies):.4f}")
    print(f"  Range: {np.min(accuracies):.4f} – {np.max(accuracies):.4f}")

    with open(os.path.join(OUTPUT_DIR, "kfold_plantvillage.json"), "w") as f:
        json.dump({
            "dataset": "plantvillage",
            "fold_summaries": fold_summaries,
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "min_accuracy": float(np.min(accuracies)),
            "max_accuracy": float(np.max(accuracies)),
        }, f, indent=2)

    return fold_summaries


# =============================================================================
# TRAINING ENTRY POINTS
# =============================================================================

def run_training():
    """Run main 80/20 training on all three datasets."""
    all_summaries = {}

    #Main training runs
    pv_train, pv_val = load_plantvillage()
    pv_train_size = int(15403 * 0.8) // BATCH_SIZE
    all_summaries["plantvillage"] = train_dataset(
        "plantvillage", pv_train, pv_val, NUM_CLASSES["plantvillage"],
        train_size=pv_train_size
    )

    rice_train, rice_val = load_rice()
    all_summaries["rice"] = train_dataset(
        "rice", rice_train, rice_val, NUM_CLASSES["rice"],
        train_size=int(5932 * 0.8) // BATCH_SIZE
    )

    cassava_train, cassava_val = load_cassava()
    all_summaries["cassava"] = train_dataset(
        "cassava", cassava_train, cassava_val, NUM_CLASSES["cassava"],
        train_size=int(5656 * 0.8) // BATCH_SIZE
    )

    # Final summary across all datasets
    print("\n" + "=" * 60)
    print("TRAINING RESULTS SUMMARY")
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

    # Save all summaries to json
    with open(os.path.join(OUTPUT_DIR, "final_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n✅ Training complete. Results saved to ./outputs/")


def run_kfold():
    """Run 5-fold cross-validation on all three datasets."""
    
    pv_fold_summaries = kfold_plantvillage(NUM_CLASSES["plantvillage"])
    rice_fold_summaries = kfold_local_dataset("rice", RICE_DATA_DIR, NUM_CLASSES["rice"])
    cassava_fold_summaries = kfold_local_dataset("cassava", CASSAVA_DATA_DIR, NUM_CLASSES["cassava"])
    print("\n✅ Cross-validation complete. Results saved to ./outputs/")


# =============================================================================
# MAIN BLOCK — Text UI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TensorFlow:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))
    print("=" * 60)
    print()
    print("Plant Disease CNN Replication — Hassan & Maji (2022)")
    print()
    print("  1. Run training (all datasets, 80/20 split)")
    print("  2. Run cross-validation (all datasets, 5-fold)")
    print("  3. Run both (training then cross-validation)")
    print()

    choice = input("Select option [1/2/3]: ").strip()

    if choice == "1":
        run_training()
    elif choice == "2":
        run_kfold()
    elif choice == "3":
        run_training()
        run_kfold()
    else:
        print(f"Invalid option '{choice}'. Please run again and enter 1, 2, or 3.")


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
# 5. CLASS WEIGHTS — The Cassava dataset is imbalanced (the paper notes
#    this explicitly). No class weighting is applied here, matching the
#    paper's apparent approach. If accuracy is poor, add class_weight to
#    model.fit() as a post-hoc experiment.
#
# =============================================================================