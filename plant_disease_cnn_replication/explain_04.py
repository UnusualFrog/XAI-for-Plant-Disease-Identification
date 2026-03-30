import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import cv2
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Cap GPU memory to leave headroom for model + SHAP graph.
# Mirrors the cap in train_03.py. Without this, DeepExplainer.__init__
# runs the full background batch through the model in one pass and
# exhausts VRAM on the 2070 Super.
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4000)]
    )

from data_loading_01 import (
    load_plantvillage, load_rice, load_cassava,
    NUM_CLASSES, SEED, BATCH_SIZE
)

OUTPUT_DIR = "./outputs/explain"
MODELS_DIR = "./outputs"
os.makedirs(f"{OUTPUT_DIR}/gradcam", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/lime",    exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/shap",    exist_ok=True)

# Number of validation images to explain per dataset
N_SAMPLES = 5


def load_model(dataset_name):
    path = os.path.join(MODELS_DIR, f"best_{dataset_name}.keras")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model at {path}. Run training first.")
    print(f"  Loading {dataset_name} model from {path}")
    return tf.keras.models.load_model(path)


def get_val_samples(val_ds, n=N_SAMPLES):
    """Extract n images and their true labels from the validation set."""
    images, labels = [], []
    for batch_imgs, batch_lbls in val_ds:
        for img, lbl in zip(batch_imgs, batch_lbls):
            images.append(img.numpy())
            labels.append(int(lbl.numpy()))
            if len(images) >= n:
                return np.array(images), np.array(labels)
    return np.array(images), np.array(labels)


# =============================================================================
# GRAD-CAM
# =============================================================================

def find_gradcam_target_layer(model):
    """
    Find the last Concatenate layer before GlobalAveragePooling2D.

    In this architecture, the final Concatenate layer (end of
    modified_reduction_b) merges all branch outputs into the last full
    spatial feature map, making it the correct Grad-CAM target.

    Avoids layer.output_shape entirely — that property is not reliably
    available on models loaded from .keras files in TF2/Keras 3, where
    output shapes are not reconstructed from the saved format and raise
    AttributeError on every layer regardless of type. Type-based search
    via isinstance is sufficient and always works on loaded models.

    Search priority:
      1. Last Concatenate before GlobalAveragePooling2D (preferred)
      2. Last Conv2D before GlobalAveragePooling2D (fallback)
    """
    pre_gap_layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            break
        pre_gap_layers.append(layer)

    if not pre_gap_layers:
        raise ValueError(
            "No layers found before GlobalAveragePooling2D. "
            "Check that the model loaded correctly."
        )

    # Priority 1: last Concatenate layer — captures merged multi-branch output
    for layer in reversed(pre_gap_layers):
        if isinstance(layer, tf.keras.layers.Concatenate):
            print(f"  Grad-CAM target layer: {layer.name}  (Concatenate)")
            return layer.name

    # Priority 2: last Conv2D layer — fallback if no Concatenate found
    for layer in reversed(pre_gap_layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            print(f"  Grad-CAM target layer: {layer.name}  (Conv2D fallback)")
            return layer.name

    raise ValueError(
        "Could not find a Concatenate or Conv2D layer before "
        "GlobalAveragePooling2D. Model structure may be unexpected."
    )


def make_gradcam_heatmap(model, image, target_layer_name, class_idx=None):
    """
    Compute Grad-CAM heatmap for a single image.
    If class_idx is None, uses the predicted class.
    Returns the heatmap and the class index used.
    """
    # Sub-model that outputs both the target layer activations and the predictions
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(target_layer_name).output,
            model.output,
        ]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(image[np.newaxis, ...], tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        if class_idx is None:
            # Explicit int conversion prevents EagerTensor indexing issues
            class_idx = int(tf.argmax(predictions[0]))
        loss = predictions[:, class_idx]

    # Gradients of the class score with respect to the target layer output
    grads = tape.gradient(loss, conv_outputs)

    # Defensive check — grads will be None if the target layer is not in the
    # computation graph (e.g. wrong layer name or disconnected branch)
    if grads is None:
        raise ValueError(
            f"Grad-CAM gradient is None for layer '{target_layer_name}'. "
            "The target layer may not be connected to the model output."
        )

    # Global average pool the gradients to get per-filter importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature maps by the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU to keep only positive activations, then normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(class_idx)


def save_gradcam(image, heatmap, dataset_name, sample_idx, true_label, pred_label):
    """Overlay Grad-CAM heatmap on the original image and save."""
    # Resize heatmap to match image spatial dimensions
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    # Convert heatmap to RGB jet colormap
    heatmap_colored = np.uint8(255 * cm.jet(heatmap_resized)[:, :, :3])
    # Convert normalized image to uint8 for overlay
    original = np.uint8(255 * image)
    overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay\nTrue: {true_label} | Pred: {pred_label}")
    axes[2].axis("off")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "gradcam",
                        f"{dataset_name}_sample{sample_idx}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def run_gradcam(model, images, labels, dataset_name):
    print(f"\n[Grad-CAM] {dataset_name}")
    target_layer = find_gradcam_target_layer(model)
    for i, (image, true_label) in enumerate(zip(images, labels)):
        heatmap, pred_label = make_gradcam_heatmap(model, image, target_layer)
        save_gradcam(image, heatmap, dataset_name, i, true_label, pred_label)


# =============================================================================
# LIME
# =============================================================================

def run_lime(model, images, labels, dataset_name):
    print(f"\n[LIME] {dataset_name}")

    # LIME methodology parameters — report both values in the paper.
    # num_samples: perturbed neighbourhood images generated per explanation.
    # num_features: number of superpixels included in the explanation mask.
    LIME_NUM_SAMPLES = 1000
    LIME_NUM_FEATURES = 10

    def predict_fn(imgs):
        return model.predict(tf.cast(imgs, tf.float32), verbose=0)
    explainer = lime_image.LimeImageExplainer(random_state=SEED)

    for i, (image, true_label) in enumerate(zip(images, labels)):
        # num_samples: number of perturbed neighbourhood images generated
        # num_features: number of superpixels shown in the explanation
        # Both are methodology parameters and must be reported in the paper
        explanation = explainer.explain_instance(
            image.astype("double"),
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=LIME_NUM_SAMPLES,
            random_seed=SEED,
        )

        pred_label = explanation.top_labels[0]

        temp, mask = explanation.get_image_and_mask(
            pred_label,
            positive_only=True,
            num_features=LIME_NUM_FEATURES,
            hide_rest=False,
        )

        temp_both, mask_both = explanation.get_image_and_mask(
            pred_label,
            positive_only=False,
            num_features=LIME_NUM_FEATURES,
            hide_rest=False,
        )

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(mark_boundaries(temp, mask))
        axes[1].set_title(
            f"LIME — Supporting Regions\nTrue: {true_label} | Pred: {pred_label}"
        )
        axes[1].axis("off")
        axes[2].imshow(mark_boundaries(temp_both, mask_both))
        axes[2].set_title("LIME — All Regions\n(green=support, red=contradict)")
        axes[2].axis("off")

        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "lime",
                            f"{dataset_name}_sample{i}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")


# =============================================================================
# SHAP
# =============================================================================

def run_shap(model, images, labels, val_ds, dataset_name):
    print(f"\n[SHAP] {dataset_name}")

    # SHAP runs on CPU to avoid VRAM exhaustion.
    # DeepExplainer builds a full gradient computation graph at init time —
    # not just a forward pass — and the combined footprint of the resident GPU
    # model plus SHAP's internal graph exceeds available VRAM on the RTX 2070
    # Super regardless of background size. CPU execution is acceptable here
    # because SHAP only processes N_SAMPLES explanation images against a small
    # background set — runtime is on the order of minutes, not hours.
    # A separate CPU model instance is loaded so the GPU model used by
    # Grad-CAM and LIME remains resident and unaffected.
    print("  Loading CPU model for SHAP...")
    model_path = os.path.join(MODELS_DIR, f"best_{dataset_name}.keras")
    with tf.device("/CPU:0"):
        cpu_model = tf.keras.models.load_model(model_path)

    # Build background dataset from validation samples that do NOT overlap
    # with the explanation images.
    #
    # ORDERING ASSUMPTION: val_ds uses cache() and reshuffle_each_iteration=False,
    # so iteration order is deterministic across calls within the same process.
    # get_val_samples() takes the first N_SAMPLES images from val_ds, so skipping
    # the first N_SAMPLES images here guarantees a non-overlapping background.
    # If the pipeline is ever changed to allow reshuffling, this assumption breaks
    # and images should be passed in directly to exclude them explicitly.
    #
    # 20 background samples — report in paper.
    SHAP_BACKGROUND_SIZE = 20
    background = []
    skipped = 0
    for batch_imgs, _ in val_ds:
        for img in batch_imgs:
            if skipped < N_SAMPLES:
                skipped += 1
                continue
            background.append(img.numpy())
            if len(background) >= SHAP_BACKGROUND_SIZE:
                break
        if len(background) >= SHAP_BACKGROUND_SIZE:
            break
    background = np.array(background, dtype=np.float32)
    print(f"  Background samples collected: {len(background)}")

    with tf.device("/CPU:0"):
        # GradientExplainer is used instead of DeepExplainer because this
        # architecture uses DepthwiseConv2D, which has no entry in DeepExplainer's
        # internal gradient registry (shap_DepthwiseConv2dNative is unregistered).
        # GradientExplainer uses TF's native GradientTape autodiff and has no
        # such registry, making it compatible with any TF-supported op including
        # DepthwiseConv2D. Known TF2 instability with GradientExplainer applies
        # only to GPU eager mode — CPU execution is stable.
        # nsamples: number of background samples drawn per explanation — report in paper.
        explainer = shap.GradientExplainer(cpu_model, background)
        shap_values = explainer.shap_values(
            images.astype(np.float32),
            nsamples=200,
        )

    # GradientExplainer return format varies by SHAP version:
    # - Older versions: list[n_classes] of [n_samples, H, W, C]
    # - Newer versions: single ndarray of shape [n_samples, H, W, C, n_classes]
    #   or [n_samples, H, W, C] for single-output models
    # The isinstance check normalises both cases to list[n_classes].
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 5:
            # Shape [n_samples, H, W, C, n_classes] — split on last axis
            shap_values = [shap_values[..., c] for c in range(shap_values.shape[-1])]
        else:
            # Shape [n_samples, H, W, C] — single output, wrap in list
            shap_values = [shap_values]
    num_classes = len(shap_values)

    for i, (image, true_label) in enumerate(zip(images, labels)):
        # Use GPU model for prediction to keep inference fast
        pred = model.predict(image[np.newaxis, ...], verbose=0)
        pred_label = int(np.argmax(pred))

        # Local explanation — SHAP values for the predicted class only
        # shap_values[pred_label] shape: [n_samples, H, W, C]
        local_shap = shap_values[pred_label][i]        # shape: [H, W, C]

        # Summarise across colour channels for visualisation
        local_shap_summary = np.mean(np.abs(local_shap), axis=-1)  # shape: [H, W]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        im = axes[1].imshow(local_shap_summary, cmap="hot")
        axes[1].set_title(
            f"SHAP — Local Explanation\nTrue: {true_label} | Pred: {pred_label}"
        )
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        shap_norm = local_shap_summary / (local_shap_summary.max() + 1e-8)
        shap_rgb = cm.hot(shap_norm)[:, :, :3]
        overlay = np.clip(image * 0.5 + shap_rgb * 0.5, 0.0, 1.0)
        axes[2].imshow(overlay)
        axes[2].set_title("SHAP Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "shap",
                            f"{dataset_name}_sample{i}_local.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")

    # Global explanation — mean absolute SHAP value across all samples and classes
    mean_shap = np.mean(
        [np.mean(np.abs(shap_values[c]), axis=0) for c in range(num_classes)],
        axis=0
    )  # shape: [H, W, C]
    mean_shap_summary = np.mean(mean_shap, axis=-1)  # shape: [H, W]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mean_shap_summary, cmap="hot")
    ax.set_title(
        f"SHAP — Global Mean |SHAP|\n{dataset_name} ({len(images)} samples)"
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "shap", f"{dataset_name}_global.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # Explicitly delete CPU model to free RAM before next dataset
    del cpu_model


# =============================================================================
# ENTRY POINT
# =============================================================================

def run_explainability(dataset_name, load_fn):
    print(f"\n{'='*60}")
    print(f"Explainability — {dataset_name}")
    print(f"{'='*60}")

    model = load_model(dataset_name)
    # val_ds is a cached tf.data pipeline — safe to iterate multiple times
    _, val_ds = load_fn()

    images, labels = get_val_samples(val_ds, n=N_SAMPLES)
    print(f"  Extracted {len(images)} validation samples")

    run_gradcam(model, images, labels, dataset_name)
    run_lime(model, images, labels, dataset_name)
    # val_ds passed separately so SHAP can build a non-overlapping background
    run_shap(model, images, labels, val_ds, dataset_name)


if __name__ == "__main__":
    print("=" * 60)
    print("Explainability — Grad-CAM, LIME, SHAP")
    print("Hassan & Maji (2022) Replication")
    print("=" * 60)

    run_explainability("plantvillage", load_plantvillage)
    run_explainability("rice",         load_rice)
    run_explainability("cassava",      load_cassava)

    print("\n✅ Explainability complete. Results saved to ./outputs/explain/")