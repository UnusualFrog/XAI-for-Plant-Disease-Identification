"""
Replication - Model Training
"""

import os
# Disable cuDNN algorithm autotuning to save GPU memory
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf

# Handle GPU memory cap at 4GB to leave headroom for OS and other processes
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4000)]
    )

# Import project modules
from data_loading_01 import (
    load_plantvillage, load_rice, load_cassava,
    load_plantvillage_fold2, load_rice_fold2, load_cassava_fold2,
    NUM_CLASSES, EPOCHS, BATCH_SIZE, SEED
)
from model_02 import build_model

# Set global random seed
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Model Training Hyperparameters
# NOTE: Paper states Adam optimizer but gives no learning rate, schedule, or
# decay, 1e-3 (Adam default) used here
LEARNING_RATE = 1e-3
# NOTE: Dropout rate not stated in paper, 0.5 used as default
DROPOUT_RATE = 0.5
# Directory to output results of training
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Helper functions

# Callback function for saving best model version, saving training logs to csv, and learning rate reduction in training
def make_callbacks(dataset_name, monitor="val_accuracy", tag=None):
    # tag allows fold2 outputs to use a distinct filename
    filename = f"{dataset_name}_{tag}" if tag else dataset_name
    ckpt_path = os.path.join(OUTPUT_DIR, f"best_{filename}.keras")
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path, monitor=monitor,
            save_best_only=True, verbose=0,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(OUTPUT_DIR, f"log_{filename}.csv")
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=0.5, patience=5, min_lr=1e-6,
            mode="min" if "loss" in monitor else "max", verbose=1,
        ),
    ]

# Compile model with Adam optimizer and sparse categorical crossentropy loss
def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# Plot accuracy and loss curves and save to outputs directory
def plot_history(history, dataset_name, tag=None):
    # Include validation fold in name if exists
    display_name = f"{dataset_name} ({tag})" if tag else dataset_name
    file_tag = f"{dataset_name}_{tag}" if tag else dataset_name

    # 1 row of 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Training vs. validation accuracy over epochs
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title(f"{display_name} — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Training vs. validation loss over epochs
    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title(f"{display_name} — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"curves_{file_tag}.png"), dpi=150)
    plt.close()

# Evaluate model on validation dataset
# Produces classification report, confusion matrix heatmap, and JSON summary
def evaluate_and_report(model, val_ds, dataset_name, num_classes, tag=None):
    # Include validation fold in name if exists
    display_name = f"{dataset_name} ({tag})" if tag else dataset_name
    file_tag = f"{dataset_name}_{tag}" if tag else dataset_name

    # Collect predictions and labels in a single pass to avoid ordering
    # mismatch between two separate iterations over val_ds
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Classification report produces precision, recall, F1, accuracy
    report = classification_report(y_true, y_pred, output_dict=True)
    report_text = classification_report(y_true, y_pred)
    print(f"\n[{display_name}] Classification Report:\n{report_text}")

    # Save classification report to file
    with open(os.path.join(OUTPUT_DIR, f"report_{file_tag}.txt"), "w") as f:
        f.write(report_text)

    # Save confusion matrix heatmap to file
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, num_classes), max(5, num_classes - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{display_name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"cm_{file_tag}.png"), dpi=150)
    plt.close()

    # Save scalar summary to JSON
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    summary = {
        "dataset": file_tag,
        "accuracy": float(accuracy),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
    }
    with open(os.path.join(OUTPUT_DIR, f"summary_{file_tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {summary['macro_precision']:.4f}")
    print(f"  Recall:    {summary['macro_recall']:.4f}")
    print(f"  F1-score:  {summary['macro_f1']:.4f}")

    return summary

# Build, compile, train and evaluate model on a single dataset
def train_dataset(dataset_name, train_ds, val_ds, num_classes, steps_per_epoch, tag=None):
    print(f"\n{'='*60}")
    label = f"{dataset_name} ({tag})" if tag else dataset_name
    print(f"Training on {label} ({num_classes} classes)")
    print(f"{'='*60}")

    # Build and complile blank model
    model = build_model(num_classes=num_classes, dropout_rate=DROPOUT_RATE)
    model = compile_model(model)
    # Use val_loss for cassava to address class imbalance
    monitor = "val_loss" if dataset_name == "cassava" else "val_accuracy"

    # Train model on training data
    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=make_callbacks(dataset_name, monitor=monitor, tag=tag),
        verbose=1,
    )

    plot_history(history, dataset_name, tag=tag)

    # Load best model for evaluation
    filename = f"{dataset_name}_{tag}" if tag else dataset_name
    best_path = os.path.join(OUTPUT_DIR, f"best_{filename}.keras")
    if os.path.exists(best_path):
        model = tf.keras.models.load_model(best_path)

    summary = evaluate_and_report(model, val_ds, dataset_name, num_classes, tag=tag)
    return summary

# Train model on each dataset
def run_training():
    all_summaries = {}

    # PlantVillage
    pv_train, pv_val = load_plantvillage()
    # 80/20 split in batches of 16
    pv_train_size = int(15403 * 0.8) // BATCH_SIZE
    all_summaries["plantvillage"] = train_dataset(
        "plantvillage", pv_train, pv_val,
        NUM_CLASSES["plantvillage"],
        steps_per_epoch=pv_train_size,
    )

    # Rice
    rice_train, rice_val = load_rice()
    # 80/20 split in batches of 16
    rice_train_size = int(5932 * 0.8) // BATCH_SIZE
    all_summaries["rice"] = train_dataset(
        "rice", rice_train, rice_val,
        NUM_CLASSES["rice"],
        steps_per_epoch=rice_train_size,
    )

    # Cassava
    cassava_train, cassava_val = load_cassava()
    # 80/20 split in batches of 16
    cassava_train_size = int(5656 * 0.8) // BATCH_SIZE
    all_summaries["cassava"] = train_dataset(
        "cassava", cassava_train, cassava_val,
        NUM_CLASSES["cassava"],
        steps_per_epoch=cassava_train_size,
    )

    # Final summary across all datasets compared against paper targets
    print("\n" + "=" * 60)
    print("TRAINING RESULTS SUMMARY")
    print("=" * 60)
    paper_targets = {
        "plantvillage": 0.9939,
        "rice": 0.9966,
        "cassava": 0.7659,
    }
    
    # Validate summary results
    for name, summary in all_summaries.items():
        target = paper_targets[name]
        delta = summary["accuracy"] - target
        print(f"  {name:12s}  Achieved: {summary['accuracy']:.4f}  "
              f"Target: {target:.4f}  Delta: {delta:+.4f}")

    with open(os.path.join(OUTPUT_DIR, "final_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n Training complete. Results saved to ./outputs/")
    return all_summaries

# Run independent 80/20 split for lightweight validation
def run_fold2():
    fold2_summaries = {}

    # Load dataset with second-fold shuffle for validation 
    pv_train, pv_val = load_plantvillage_fold2()
    fold2_summaries["plantvillage"] = train_dataset(
        "plantvillage", pv_train, pv_val,
        NUM_CLASSES["plantvillage"],
        steps_per_epoch=int(15403 * 0.8) // BATCH_SIZE,
        tag="fold2",
    )

    # Load dataset with second-fold shuffle for validation 
    rice_train, rice_val = load_rice_fold2()
    fold2_summaries["rice"] = train_dataset(
        "rice", rice_train, rice_val,
        NUM_CLASSES["rice"],
        steps_per_epoch=int(5932 * 0.8) // BATCH_SIZE,
        tag="fold2",
    )

    # Load dataset with second-fold shuffle for validation 
    cassava_train, cassava_val = load_cassava_fold2()
    fold2_summaries["cassava"] = train_dataset(
        "cassava", cassava_train, cassava_val,
        NUM_CLASSES["cassava"],
        steps_per_epoch=int(5656 * 0.8) // BATCH_SIZE,
        tag="fold2",
    )

    # Save second fold validation results to file
    with open(os.path.join(OUTPUT_DIR, "fold2_summary.json"), "w") as f:
        json.dump(fold2_summaries, f, indent=2)

    print("\n Fold 2 complete. Results saved to ./outputs/")
    return fold2_summaries

# Compare results of two different validation folds
def compare_folds(fold1_summaries, fold2_summaries):
    print("\n" + "=" * 60)
    print("FOLD STABILITY COMPARISON (±5% threshold)")
    print("=" * 60)
    comparison = {}

    for name in fold1_summaries:
        a1 = fold1_summaries[name]["accuracy"]
        a2 = fold2_summaries[name]["accuracy"]
        # Compute difference in accuracy between fold1 and fold2
        delta = abs(a1 - a2)
        # Stable if difference between folds is within 5%
        status = "STABLE" if delta <= 0.05 else "UNSTABLE"
        print(f"  {name:12s}  Fold1: {a1:.4f}  Fold2: {a2:.4f}  "
              f"Δ: {delta:.4f}  {status}")
        comparison[name] = {
            "fold1_accuracy": a1,
            "fold2_accuracy": a2,
            "delta": float(delta),
            "status": status,
        }
    with open(os.path.join(OUTPUT_DIR, "fold_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

# Main Block
if __name__ == "__main__":
    print("=" * 60)
    print("TensorFlow:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))
    print("=" * 60)
    print()
    print("Plant Disease CNN Replication")
    print()
    print("  1. Run baseline training only")
    print("  2. Run baseline + fold 2 stability check")
    print()
    choice = input("Select option [1/2]: ").strip()

    if choice == "1":
        run_training()
    elif choice == "2":
        fold1 = run_training()
        fold2 = run_fold2()
        compare_folds(fold1, fold2)
    else:
        print(f"Invalid option '{choice}'.")


# =============================================================================
# REPLICATION GAPS
# =============================================================================
#
# 1. Paper states Adam but gives no learning rate, decay,
#    or schedule, using 1e-3 (Adam default)
#
# 2. Batch size not stated in paper, using 16
#
# 3. Dropout rate not stated, using 0.5.
#
# =============================================================================