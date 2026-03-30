"""
Replication - Model Training

Training protocol:
  - Optimizer: Adam (paper states Adam, no LR given — see GAPS)
  - Epochs: 50
  - Split: 80% train / 20% validation
  - Metrics: accuracy, precision, recall, F1-score, confusion matrix
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
    load_plantvillage_full, load_plantvillage, load_rice, load_cassava,
    NUM_CLASSES, EPOCHS, BATCH_SIZE, SEED,
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

# Added monitor parameter so Cassava can use val_loss instead of val_accuracy
# due to class imbalance making val_accuracy an unreliable checkpoint signal
def make_callbacks(dataset_name, monitor="val_accuracy"):
    ckpt_path = os.path.join(OUTPUT_DIR, f"best_{dataset_name}.keras")

    return [
        # Save model checkpoint at best monitored metric
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor=monitor,
            save_best_only=True,
            verbose=0,
        ),
        # Save per-epoch training metrics to CSV
        tf.keras.callbacks.CSVLogger(
            os.path.join(OUTPUT_DIR, f"log_{dataset_name}.csv")
        ),
        # Reduce learning rate by 0.5 when monitored metric plateaus
        # mode inferred from monitor name, min for loss, max for accuracy
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode="min" if "loss" in monitor else "max",
            verbose=1,
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
def plot_history(history, dataset_name):
    # 1 row of 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Training vs. validation accuracy over epochs
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title(f"{dataset_name} — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Training vs. validation loss over epochs
    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title(f"{dataset_name} — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"curves_{dataset_name}.png"), dpi=150)
    plt.close()

# Evaluate model on validation dataset
# Produces classification report, confusion matrix heatmap, and JSON summary
def evaluate_and_report(model, val_ds, dataset_name, num_classes):
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
    print(f"\n[{dataset_name}] Classification Report:\n{report_text}")

    # Save classification report to file
    with open(os.path.join(OUTPUT_DIR, f"report_{dataset_name}.txt"), "w") as f:
        f.write(report_text)

    # Save confusion matrix heatmap to file
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, num_classes), max(5, num_classes - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{dataset_name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"cm_{dataset_name}.png"), dpi=150)
    plt.close()

    # Save scalar summary to JSON
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    summary = {
        "dataset": dataset_name,
        "accuracy": float(accuracy),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
    }
    with open(os.path.join(OUTPUT_DIR, f"summary_{dataset_name}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {summary['macro_precision']:.4f}")
    print(f"  Recall:    {summary['macro_recall']:.4f}")
    print(f"  F1-score:  {summary['macro_f1']:.4f}")

    return summary

# Build, compile, train and evaluate model on a single dataset
def train_dataset(dataset_name, train_ds, val_ds, num_classes, steps_per_epoch):
    print(f"\n{'='*60}")
    print(f"Training on {dataset_name} ({num_classes} classes)")
    print(f"{'='*60}")

    model = build_model(num_classes=num_classes, dropout_rate=DROPOUT_RATE)
    model = compile_model(model)

    # Use val_loss monitor for Cassava due to class imbalance making
    # val_accuracy an unreliable checkpoint signal
    monitor = "val_loss" if dataset_name == "cassava" else "val_accuracy"

    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=make_callbacks(dataset_name, monitor=monitor),
        verbose=1,
    )

    plot_history(history, dataset_name)

    # Load best checkpoint for evaluation
    best_path = os.path.join(OUTPUT_DIR, f"best_{dataset_name}.keras")
    if os.path.exists(best_path):
        model = tf.keras.models.load_model(best_path)

    summary = evaluate_and_report(model, val_ds, dataset_name, num_classes)
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
    for name, summary in all_summaries.items():
        target = paper_targets[name]
        delta = summary["accuracy"] - target
        print(f"  {name:12s}  Achieved: {summary['accuracy']:.4f}  "
              f"Target: {target:.4f}  Delta: {delta:+.4f}")

    with open(os.path.join(OUTPUT_DIR, "final_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n Training complete. Results saved to ./outputs/")

# Main Block
if __name__ == "__main__":
    print("=" * 60)
    print("TensorFlow:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))
    print("=" * 60)
    print()
    print("Plant Disease CNN Replication — Hassan & Maji (2022)")
    print()
    run_training()


# =============================================================================
# REPLICATION GAPS
# =============================================================================
#
# 1. LEARNING RATE — Paper states Adam but gives no learning rate, decay,
#    or schedule. Using 1e-3 (Adam default).
#
# 2. BATCH SIZE — Not stated in paper. Using 16.
#
# 3. DROPOUT RATE — Not stated. Using 0.5.
#
# =============================================================================