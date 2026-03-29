"""
Replication - Model Definition
"""

import os
import numpy as np
import tensorflow as tf

layers = tf.keras.layers
Model = tf.keras.Model

# Global Random Seed
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Standard convolution with batch normalization and ReLU acitvation
def conv_bn_relu(x, filters, kernel_size, strides=1, padding="same"):
    # Apply standard convolution
    x = layers.Conv2D(filters, kernel_size, strides=strides,
                  padding=padding, use_bias=False,
                  kernel_initializer="he_normal")(x)
    # Normalize
    x = layers.BatchNormalization()(x)
    #ReLU activation
    x = layers.Activation("relu")(x)
    return x

# Depthwise seperable convolution with batch normalization and ReLU activation
# followed by pointwise 1x1 convolution with batch norm and ReLU
# reduces parameter count compared to standard convolusion
def depthwise_separable_bn_relu(x, filters, kernel_size, strides=1, padding="same"):

    # Depthwise convolution, convolved channels independently
    x = layers.DepthwiseConv2D(kernel_size, strides=strides,
                            padding=padding, use_bias=False,
                            depthwise_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # 1×1 pointwise convolution, computes weighted sum across channels for each pixel for each filter
    # different filters extract increasingly abstract patterns
    x = layers.Conv2D(filters, 1, padding="same", use_bias=False,
                  kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


# Inception Block A
# Convolves with four branches independently and concatenate results of each
# Extracts features at varying levels of abstraction
# Uses ResNet to improve gradient flow
# If input channels not equal to concatenated output channels, a 1×1 conv aligns them.
def modified_inception_a(x, filters_1x1, filters_3x3_reduce, filters_3x3,
                          filters_5x5_reduce, filters_5x5, filters_pool):
    # ResNet residual (distance from identity)
    residual = x

    # Branch 1: 1×1 conv to capture simple patterns
    b1 = conv_bn_relu(x, filters_1x1, 1)

    # Branch 2: 1×1 conv to 3×3 depthwise separable conv, more complexity 
    b2 = conv_bn_relu(x, filters_3x3_reduce, 1)
    b2 = depthwise_separable_bn_relu(b2, filters_3x3, 3)

    # Branch 3: 1×1 conv to two 3×3 depthwise separable convs, complex patterns
    b3 = conv_bn_relu(x, filters_5x5_reduce, 1)
    b3 = depthwise_separable_bn_relu(b3, filters_5x5, 3)
    b3 = depthwise_separable_bn_relu(b3, filters_5x5, 3)

    # Branch 4: 3×3 avgerage pool to 1×1 conv, extract simple patterns from shrinked feature map
    b4 = layers.AveragePooling2D(3, strides=1, padding="same")(x)
    b4 = conv_bn_relu(b4, filters_pool, 1)

    # Concatenate all output of branches
    out = layers.Concatenate()([b1, b2, b3, b4])

    # ResNet connection aligns channels with 1×1 conv if necessary
    total_filters = filters_1x1 + filters_3x3 + filters_5x5 + filters_pool
    input_channels = residual.shape[-1]
    if input_channels != total_filters:
        # Project output channels into correct dimensions for ResNet integrity
        residual = layers.Conv2D(total_filters, 1, padding="same", use_bias=False,
                         kernel_initializer="he_normal")(residual)
        residual = layers.BatchNormalization()(residual)

    # Combine output with input (ResNet)
    out = layers.Add()([out, residual])
    # ReLU activation
    out = layers.Activation("relu")(out)
    return out


# Inception Block B
# Convolve with four branches independently and concatenate results of each
# Extracts features at varying levels of abstraction
# Inception Block B extracts significantly more abstract and global patterns
# Uses ResNet to improve gradient flow
# If input channels not equal to concatenated output channels, a 1×1 conv aligns them.
def modified_inception_b(x, filters_1x1, filters_7x7_reduce, filters_7x7,
                          filters_7x7_dbl_reduce, filters_7x7_dbl, filters_pool):
    # ResNet residual
    residual = x

    # Branch 1: 1×1 conv, simple patterns
    b1 = conv_bn_relu(x, filters_1x1, 1)

    # Branch 2: 1×1 conv to 7×7 depthwise separable conv
    b2 = conv_bn_relu(x, filters_7x7_reduce, 1)
    b2 = depthwise_separable_bn_relu(b2, filters_7x7, 7)

    # Branch 3: 1×1 conv to double 7×7 depthwise separable conv
    b3 = conv_bn_relu(x, filters_7x7_dbl_reduce, 1)
    b3 = depthwise_separable_bn_relu(b3, filters_7x7_dbl, 7)

    # Branch 4: 3×3 average-pool to 1×1 conv
    b4 = layers.AveragePooling2D(3, strides=1, padding="same")(x)
    b4 = conv_bn_relu(b4, filters_pool, 1)

    # Concatenate all branches
    out = layers.Concatenate()([b1, b2, b3, b4])

    # Residual connection: align channels if necessary
    total_filters = filters_1x1 + filters_7x7 + filters_7x7_dbl + filters_pool
    input_channels = residual.shape[-1]
    if input_channels != total_filters:
        residual = layers.Conv2D(total_filters, 1, padding="same", use_bias=False)(residual)
        residual = layers.BatchNormalization()(residual)

    # Add input to output (ResNet)
    out = layers.Add()([out, residual])
    # Relu Activation
    out = layers.Activation("relu")(out)
    return out


# Reduction block A
# Reduces spatial dimensions of feature map with downsampling

def modified_reduction_a(x, filters_3x3_reduce, filters_3x3, filters_dbl_reduce, filters_dbl, filters_pool):
    # Branch 1: 3×3 max-pool (stride 2 to reduce spatial dims) to 1x1 conv, gets max value in 2x2 window to shrink feature map
    b1 = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    b1 = conv_bn_relu(b1, filters_pool, 1)

    # Branch 2: 1×1 conv to 3×3 depthwise separable conv (stride 2)
    b2 = conv_bn_relu(x, filters_3x3_reduce, 1)
    b2 = depthwise_separable_bn_relu(b2, filters_3x3, 3, strides=2)

    # Branch 3: 1×1 conv to 3×3 depthwise separable conv (stride 2)
    b3 = conv_bn_relu(x, filters_dbl_reduce, 1)
    b3 = depthwise_separable_bn_relu(b3, filters_dbl, 3, strides=2)

    # Concatenate reduced dimension outputs
    out = layers.Concatenate()([b1, b2, b3])
    return out


# Reduction Block B
# Further reduces spatial dimensions of feature map with downsampling
def modified_reduction_b(x, filters_3x3_reduce, filters_3x3,
                          filters_7x7_reduce, filters_7x7, filters_pool):
    # Branch 1: 3×3 max-pool (stride 2)
    b1 = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    b1 = conv_bn_relu(b1, filters_pool, 1)

    # Branch 2: 1×1 conv to 3×3 depthwise separable conv (stride 2)
    b2 = conv_bn_relu(x, filters_3x3_reduce, 1)
    b2 = depthwise_separable_bn_relu(b2, filters_3x3, 3, strides=2)

    # Branch 3: 1×1 conv → 3×3 depthwise separable conv (stride 2)
    b3 = conv_bn_relu(x, filters_7x7_reduce, 1)
    b3 = depthwise_separable_bn_relu(b3, filters_7x7, 3, strides=2)

    out = layers.Concatenate()([b1, b2, b3])
    return out

# Build Full keras Model
# 1.  Input (256×256×3)
# 2.  Conv + BN + ReLU
# 3.  Depthwise Separable Conv Blocks + Average Pool
# 4.  [Inception A + Residual] × 3
# 5.  Reduction A
# 6.  [Inception B + Residual] × 3
# 7.  Reduction B
# 8.  Global Average Pooling
# 9.  Dropout
# 10. Fully Connected Layer
# 11. Softmax
def build_model(num_classes, input_shape=(256, 256, 3), dropout_rate=0.5):
    # 1. Load input
    inputs = layers.Input(shape=input_shape)

    x = depthwise_separable_bn_relu(inputs, 32, 3)
    x = depthwise_separable_bn_relu(x, 64, 3)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    x = depthwise_separable_bn_relu(x, 64, 3)
    x = depthwise_separable_bn_relu(x, 128, 3)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    #4. 3× Inception A with ResNet
    for _ in range(3):
        x = modified_inception_a(
            x,
            filters_1x1=40,
            filters_3x3_reduce=24, filters_3x3=48,
            filters_5x5_reduce=12, filters_5x5=24,
            filters_pool=24,
        )
    # 5. Reduction A
    x = modified_reduction_a(x, filters_3x3_reduce=48, filters_3x3=96,
                          filters_dbl_reduce=48, filters_dbl=96, filters_pool=48)

    #6. 3x Inception B with ResNet
    for _ in range(3):
        x = modified_inception_b(
            x,
            filters_1x1=72,
            filters_7x7_reduce=40, filters_7x7=72,
            filters_7x7_dbl_reduce=40, filters_7x7_dbl=72,
            filters_pool=40,
        )
    #7. Reduction B with ResNet
    x = modified_reduction_b(x,
                          filters_3x3_reduce=64, filters_3x3=128,
                          filters_7x7_reduce=64, filters_7x7=128, filters_pool=64)

    # print(f"  Pre-GAP feature map channels: {x.shape[-1]}")
    # 8. Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dropout regularization
    x = layers.Dropout(dropout_rate)(x)

    # Output layer with Softmax activtion (multi-class classification)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Construct keras model
    model = Model(inputs, outputs)
    return model

# Main Block
# Ensure total parameter counts are within 5% range of paper's count
if __name__ == "__main__":
    print("=" * 60)
    print("Building model for PlantVillage (17 classes)...")
    model_pv = build_model(num_classes=17)
    model_pv.summary()

    total_params = model_pv.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Paper target:     428,100")
    print(f"Difference:       {total_params - 428_100:+,}")

    if abs(total_params - 428_100) / 428_100 < 0.05:
        print("✅ Parameter count within 5% of paper target.")
    else:
        print("⚠️  Parameter count differs by more than 5%.")

    print("\nBuilding model for Rice (4 classes)...")
    model_rice = build_model(num_classes=4)
    print(f"Rice model params: {model_rice.count_params():,}")

    print("\nBuilding model for Cassava (5 classes)...")
    model_cassava = build_model(num_classes=5)
    print(f"Cassava model params: {model_cassava.count_params():,}")

    test_input = tf.keras.Input(shape=(64, 64, 128))
    test_out = modified_inception_a(
        test_input,
        filters_1x1=40,
        filters_3x3_reduce=24, filters_3x3=48,
        filters_5x5_reduce=12, filters_5x5=24,
        filters_pool=24,
    )
    test_model = tf.keras.Model(test_input, test_out)
    block_params = test_model.count_params()
    print(f"Inception A block params: {block_params}")
    assert abs(block_params - 11_808) / 11_808 < 0.05, \
        f"Inception A params {block_params} too far from paper target 11,808"


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
# 3. REDUCTION BLOCK STRIDES — Paper does not state stride values explicitly.
#    Stride=2 is used in both reduction blocks to halve spatial dimensions,
#    which is standard practice for reduction blocks in Inception architectures.
#
# 4. The authors state the parameter count (428,100) but never publish the filter
#    counts per block that produce it. 
# =============================================================================

# Structure
# 1. Input (256×256×3)
# 2. Conv + BN + ReLU
# 3. Depthwise Separable Conv Blocks
# 4. [Inception-A + Residual] × 3
# 5. Reduction-A
# 6. [Inception-B + Residual] × 3
# 7. Reduction-B
# 8. Global Average Pooling
# 9. Dropout
# 10. Fully Connected Layer
# 11. Softmax