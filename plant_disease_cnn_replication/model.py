"""
Replication - Model Definition
Hassan & Maji (2022): "Plant Disease Identification Using a Novel Convolutional Neural Network"
IEEE Access, DOI: 10.1109/ACCESS.2022.3141371

Architecture: Novel lightweight CNN using modified Inception blocks with depthwise separable
convolutions and residual connections. Target parameter count: 428,100.

GPU memory growth is set at the top of this file. If importing this module into another
script, ensure memory growth is configured before any other TF operations.
"""

import os
import numpy as np
import tensorflow as tf

# =============================================================================
# GPU CONFIGURATION — Must run before any other TF operations
# =============================================================================

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

layers = tf.keras.layers
models = tf.keras.models
Model = tf.keras.Model

# Global Random Seed
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

def conv_bn_relu(x, filters, kernel_size, strides=1, padding="same"):
    """
    Standard convolution + Batch Normalisation + ReLU.
    Applied after every convolution in the paper (Section III-D).
    """
    x = layers.Conv2D(filters, kernel_size, strides=strides,
                      padding=padding, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def depthwise_separable_bn_relu(x, filters, kernel_size, strides=1, padding="same"):
    """
    Depthwise separable convolution + BN + ReLU.
    Factorises standard conv into depthwise + 1×1 pointwise (Section III-C).
    Reduces parameter count significantly vs standard convolution.
    """
    # Depthwise convolution — filters spatial features per channel
    x = layers.DepthwiseConv2D(kernel_size, strides=strides,
                                padding=padding, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # 1×1 pointwise convolution — combines channel information
    x = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


# =============================================================================
# MODIFIED INCEPTION-A BLOCK (Figure 3b)
#
# Original Inception-A branches:
#   1. 1×1 conv
#   2. 1×1 conv → 3×3 conv
#   3. 1×1 conv → 5×5 conv
#   4. 3×3 maxpool → 1×1 conv
#
# Modifications (Section III-D):
#   - Branch 2: 3×3 conv replaced with 3×3 depthwise separable conv
#   - Branch 3: 5×5 conv replaced with two 3×3 depthwise separable convs
#
# Residual connection added around the block (Section III-D).
# If input channels ≠ concatenated output channels, a 1×1 conv aligns them.
# =============================================================================

def modified_inception_a(x, filters_1x1, filters_3x3_reduce, filters_3x3,
                          filters_5x5_reduce, filters_5x5, filters_pool):
    """
    Modified Inception-A block with depthwise separable convolutions and
    residual connection.

    Args:
        filters_1x1:        filters for the direct 1×1 branch
        filters_3x3_reduce: 1×1 bottleneck before 3×3 branch
        filters_3x3:        output filters of the 3×3 depthwise sep branch
        filters_5x5_reduce: 1×1 bottleneck before the double-3×3 branch
        filters_5x5:        output filters of the double-3×3 branch
        filters_pool:       1×1 filters after max-pool branch
    """
    residual = x

    # Branch 1: 1×1 conv
    b1 = conv_bn_relu(x, filters_1x1, 1)

    # Branch 2: 1×1 conv → 3×3 depthwise separable conv
    b2 = conv_bn_relu(x, filters_3x3_reduce, 1)
    b2 = depthwise_separable_bn_relu(b2, filters_3x3, 3)

    # Branch 3: 1×1 conv → two 3×3 depthwise separable convs (replaces 5×5)
    b3 = conv_bn_relu(x, filters_5x5_reduce, 1)
    b3 = depthwise_separable_bn_relu(b3, filters_5x5, 3)
    b3 = depthwise_separable_bn_relu(b3, filters_5x5, 3)

    # Branch 4: 3×3 max-pool → 1×1 conv
    b4 = layers.MaxPooling2D(3, strides=1, padding="same")(x)
    b4 = conv_bn_relu(b4, filters_pool, 1)

    # Concatenate all branches
    out = layers.Concatenate()([b1, b2, b3, b4])

    # Residual connection: align channels with 1×1 conv if necessary
    total_filters = filters_1x1 + filters_3x3 + filters_5x5 + filters_pool
    input_channels = residual.shape[-1]
    if input_channels != total_filters:
        residual = conv_bn_relu(residual, total_filters, 1)

    out = layers.Add()([out, residual])
    return out


# =============================================================================
# MODIFIED INCEPTION-B BLOCK (Figure 4b)
#
# Original Inception-B branches:
#   1. 1×1 conv
#   2. 1×1 conv → 7×7 conv
#   3. 1×1 conv → two 7×7 convs
#   4. 3×3 maxpool → 1×1 conv
#
# Modifications (Section III-D):
#   - Branch 2: 7×7 conv replaced with 7×7 depthwise separable conv
#   - Branch 3: two 7×7 convs replaced with 7×7 depthwise separable conv
#
# Residual connection added around the block.
# =============================================================================

def modified_inception_b(x, filters_1x1, filters_7x7_reduce, filters_7x7,
                          filters_7x7_dbl_reduce, filters_7x7_dbl, filters_pool):
    """
    Modified Inception-B block with depthwise separable convolutions and
    residual connection.

    Args:
        filters_1x1:            filters for the direct 1×1 branch
        filters_7x7_reduce:     1×1 bottleneck before single 7×7 branch
        filters_7x7:            output filters of the single 7×7 branch
        filters_7x7_dbl_reduce: 1×1 bottleneck before double 7×7 branch
        filters_7x7_dbl:        output filters of the double 7×7 branch
        filters_pool:           1×1 filters after max-pool branch
    """
    residual = x

    # Branch 1: 1×1 conv
    b1 = conv_bn_relu(x, filters_1x1, 1)

    # Branch 2: 1×1 conv → 7×7 depthwise separable conv
    b2 = conv_bn_relu(x, filters_7x7_reduce, 1)
    b2 = depthwise_separable_bn_relu(b2, filters_7x7, 7)

    # Branch 3: 1×1 conv → 7×7 depthwise separable conv (replaces two 7×7)
    b3 = conv_bn_relu(x, filters_7x7_dbl_reduce, 1)
    b3 = depthwise_separable_bn_relu(b3, filters_7x7_dbl, 7)

    # Branch 4: 3×3 max-pool → 1×1 conv
    b4 = layers.MaxPooling2D(3, strides=1, padding="same")(x)
    b4 = conv_bn_relu(b4, filters_pool, 1)

    # Concatenate all branches
    out = layers.Concatenate()([b1, b2, b3, b4])

    # Residual connection: align channels if necessary
    total_filters = filters_1x1 + filters_7x7 + filters_7x7_dbl + filters_pool
    input_channels = residual.shape[-1]
    if input_channels != total_filters:
        residual = conv_bn_relu(residual, total_filters, 1)

    out = layers.Add()([out, residual])
    return out


# =============================================================================
# MODIFIED REDUCTION-A BLOCK (Section III-D)
#
# Original Reduction-A: 3×3 maxpool | 3×3 conv | 1×1 conv → 3×3 conv
# Modification: 3×3 conv replaced with 1×1 conv + 3×3 depthwise separable conv
# =============================================================================

def modified_reduction_a(x, filters_3x3, filters_dbl_reduce, filters_dbl):
    """
    Modified Reduction-A block. Reduces spatial dimensions.

    Args:
        filters_3x3:      output filters of the single depthwise sep branch
        filters_dbl_reduce: 1×1 bottleneck for the double-conv branch
        filters_dbl:        output filters of the double-conv branch
    """
    # Branch 1: 3×3 max-pool (stride 2 to reduce spatial dims)
    b1 = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Branch 2: 1×1 conv → 3×3 depthwise separable conv (stride 2)
    b2 = conv_bn_relu(x, filters_3x3, 1)
    b2 = depthwise_separable_bn_relu(b2, filters_3x3, 3, strides=2)

    # Branch 3: 1×1 conv → 3×3 depthwise separable conv (stride 2)
    b3 = conv_bn_relu(x, filters_dbl_reduce, 1)
    b3 = depthwise_separable_bn_relu(b3, filters_dbl, 3, strides=2)

    out = layers.Concatenate()([b1, b2, b3])
    return out


# =============================================================================
# MODIFIED REDUCTION-B BLOCK (Section III-D)
#
# Original Reduction-B: 3×3 maxpool | 3×3 conv → 1×1 | 7×7 → 3×3 conv
# Modification: 3×3 and 7×7 replaced with 1×1 + 3×3 depthwise separable conv
# =============================================================================

def modified_reduction_b(x, filters_3x3_reduce, filters_3x3,
                          filters_7x7_reduce, filters_7x7):
    """
    Modified Reduction-B block. Further reduces spatial dimensions.

    Args:
        filters_3x3_reduce: 1×1 bottleneck for the first depthwise sep branch
        filters_3x3:        output filters of the first depthwise sep branch
        filters_7x7_reduce: 1×1 bottleneck for the second depthwise sep branch
        filters_7x7:        output filters of the second depthwise sep branch
    """
    # Branch 1: 3×3 max-pool (stride 2)
    b1 = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Branch 2: 1×1 conv → 3×3 depthwise separable conv (stride 2)
    b2 = conv_bn_relu(x, filters_3x3_reduce, 1)
    b2 = depthwise_separable_bn_relu(b2, filters_3x3, 3, strides=2)

    # Branch 3: 1×1 conv → 3×3 depthwise separable conv (stride 2)
    b3 = conv_bn_relu(x, filters_7x7_reduce, 1)
    b3 = depthwise_separable_bn_relu(b3, filters_7x7, 3, strides=2)

    out = layers.Concatenate()([b1, b2, b3])
    return out


# =============================================================================
# FULL MODEL (Figure 5)
#
# Entry flow:
#   Conv(32, 3×3) → BN → ReLU
#   DepthwiseSep(64, 3×3) → MaxPool
#   DepthwiseSep(64, 3×3)
#   DepthwiseSep(128, 3×3) → MaxPool
#
# Middle flow:
#   3× Modified Inception-A (with residual) → Modified Reduction-A
#   3× Modified Inception-B (with residual) → Modified Reduction-B
#
# Exit flow:
#   Global Average Pooling → Dropout → Dense(num_classes, softmax)
#
# Filter counts are chosen to hit the paper's 428,100 parameter target.
# =============================================================================

def build_model(num_classes, input_shape=(256, 256, 3), dropout_rate=0.5):
    """
    Builds the novel CNN from Hassan & Maji (2022).

    Args:
        num_classes:   number of output disease classes
                       (17 for PlantVillage, 4 for Rice, 5 for Cassava)
        input_shape:   HxWxC — paper uses 256×256×3
        dropout_rate:  dropout before final Dense layer.
                       Paper mentions dropout but does not state the rate.
                       0.5 is used as a standard default (see GAPS below).

    Returns:
        A compiled tf.keras.Model instance.
    """
    inputs = layers.Input(shape=input_shape)

    # --- Entry flow ---
    # One standard convolution (paper: "one standard convolution")
    x = conv_bn_relu(inputs, 32, 3)

    # Three depthwise separable convolutions interspersed with two max-pools
    # (paper: "three depthwise separable convolutions, two max-pooling")
    x = depthwise_separable_bn_relu(x, 64, 3)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    x = depthwise_separable_bn_relu(x, 64, 3)

    x = depthwise_separable_bn_relu(x, 128, 3)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # --- 3× Modified Inception-A + Reduction-A ---
    for _ in range(3):
        x = modified_inception_a(
            x,
            filters_1x1=32,
            filters_3x3_reduce=16, filters_3x3=32,
            filters_5x5_reduce=8,  filters_5x5=16,
            filters_pool=16,
        )
    x = modified_reduction_a(x, filters_3x3=64,
                              filters_dbl_reduce=32, filters_dbl=64)

    # --- 3× Modified Inception-B + Reduction-B ---
    for _ in range(3):
        x = modified_inception_b(
            x,
            filters_1x1=64,
            filters_7x7_reduce=32,  filters_7x7=64,
            filters_7x7_dbl_reduce=32, filters_7x7_dbl=64,
            filters_pool=32,
        )
    x = modified_reduction_b(x,
                              filters_3x3_reduce=64, filters_3x3=128,
                              filters_7x7_reduce=64, filters_7x7=128)

    # --- Exit flow ---
    # Global average pooling (paper: "one global average pooling")
    x = layers.GlobalAveragePooling2D()(x)

    # Dropout (paper mentions dropout to reduce overfitting, rate not stated)
    x = layers.Dropout(dropout_rate)(x)

    # Final classification layer
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model


# =============================================================================
# ENTRY POINT — Sanity check model structure and parameter count
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Building model for PlantVillage (17 classes)...")
    model = build_model(num_classes=17)
    model.summary()

    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Paper target:     428,100")
    print(f"Difference:       {total_params - 428_100:+,}")

    if abs(total_params - 428_100) / 428_100 < 0.15:
        print("✅ Parameter count within 15% of paper target.")
    else:
        print("⚠️  Parameter count differs from paper by more than 15%.")
        print("   Filter counts in build_model() may need tuning.")

    print("\nBuilding model for Rice (4 classes)...")
    model_rice = build_model(num_classes=4)
    print(f"Rice model params: {model_rice.count_params():,}")

    print("\nBuilding model for Cassava (5 classes)...")
    model_cassava = build_model(num_classes=5)
    print(f"Cassava model params: {model_cassava.count_params():,}")

    print("\n✅ Model definition step complete. Proceed to 03_train.py")


# =============================================================================
# ⚠️  OUTSTANDING GAPS
# =============================================================================
#
# 1. FILTER COUNTS — The paper does not publish exact filter counts per block.
#    The values in build_model() are chosen to approximate 428,100 parameters.
#    If model.summary() shows a large discrepancy, tune the filter counts in
#    modified_inception_a, modified_inception_b, and the entry flow.
#
# 2. DROPOUT RATE — Stated in paper as used but never quantified.
#    Default here is 0.5. Tune if validation accuracy is poor.
#
# 3. ENTRY FLOW EXACT ORDERING — Figure 5 caption is high-level.
#    The ordering (conv → dsconv → pool → dsconv → dsconv → pool) is a
#    reasonable interpretation. Adjust if parameter count is far off target.
#
# 4. REDUCTION BLOCK STRIDES — Paper does not state stride values explicitly.
#    Stride=2 is used in both reduction blocks to halve spatial dimensions,
#    which is standard practice for reduction blocks in Inception architectures.
#
# =============================================================================