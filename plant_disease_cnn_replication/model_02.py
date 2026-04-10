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

# Standard convolution with batch normalization and ReLU activation
def conv_bn_relu(x, filters, kernel_size, strides=1, padding="same"):
    # Apply standard convolution
    x = layers.Conv2D(filters, kernel_size, strides=strides,
                      padding=padding, use_bias=False,
                      kernel_initializer="he_normal")(x)
    # Normalize
    x = layers.BatchNormalization()(x)
    # ReLU activation
    x = layers.Activation("relu")(x)
    return x

# Depthwise separable convolution with batch normalization and ReLU activation
# followed by pointwise 1x1 convolution with batch norm and ReLU
# Reduces parameter count compared to standard convolution
def depthwise_separable_bn_relu(x, filters, kernel_size, strides=1, padding="same"):
    # Depthwise convolution convolves channels independently
    x = layers.DepthwiseConv2D(kernel_size, strides=strides,
                                padding=padding, use_bias=False,
                                depthwise_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # 1×1 pointwise convolution computes weighted sum across channels for each
    # pixel for each filter, different filters extract increasingly abstract patterns
    x = layers.Conv2D(filters, 1, padding="same", use_bias=False,
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


# Inception Block A
# Convolves with four branches independently and concatenates results of each
# Extracts features at varying levels of abstraction
# Uses residual connection to improve gradient flow
# If input channels not equal to concatenated output channels, a 1×1 conv aligns them
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

    # Branch 4: 3×3 average pool to 1×1 conv, extract simple patterns from shrunken feature map
    b4 = layers.AveragePooling2D(3, strides=1, padding="same")(x)
    b4 = conv_bn_relu(b4, filters_pool, 1)

    # Concatenate all output of branches
    out = layers.Concatenate()([b1, b2, b3, b4])

    # Residual connection aligns channels with 1×1 conv if necessary
    total_filters = filters_1x1 + filters_3x3 + filters_5x5 + filters_pool
    input_channels = residual.shape[-1]
    if input_channels != total_filters:
        # Project input channels into correct dimensions for residual integrity
        residual = layers.Conv2D(total_filters, 1, padding="same", use_bias=False,
                                 kernel_initializer="he_normal")(residual)
        residual = layers.BatchNormalization()(residual)

    # Combine output with input (residual connection)
    out = layers.Add()([out, residual])
    # ReLU activation
    out = layers.Activation("relu")(out)
    return out


# Inception Block B
# Convolves with four branches independently and concatenates results of each
# Extracts significantly more abstract and global patterns than Block A
# Uses residual connection to improve gradient flow
# If input channels not equal to concatenated output channels, a 1×1 conv aligns them
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

    # Branch 4: 3×3 average pool to 1×1 conv
    b4 = layers.AveragePooling2D(3, strides=1, padding="same")(x)
    b4 = conv_bn_relu(b4, filters_pool, 1)

    # Concatenate all branches
    out = layers.Concatenate()([b1, b2, b3, b4])

    # Residual connection align channels if necessary
    total_filters = filters_1x1 + filters_7x7 + filters_7x7_dbl + filters_pool
    input_channels = residual.shape[-1]
    if input_channels != total_filters:
        residual = layers.Conv2D(total_filters, 1, padding="same", use_bias=False,
                                 kernel_initializer="he_normal")(residual)
        residual = layers.BatchNormalization()(residual)

    # Add input to output (residual connection)
    out = layers.Add()([out, residual])
    # ReLU activation
    out = layers.Activation("relu")(out)
    return out


# Reduction Block A
# Reduces spatial dimensions of feature map with downsampling
def modified_reduction_a(x, filters_3x3_reduce, filters_3x3,
                          filters_dbl_reduce, filters_dbl, filters_pool):
    # Branch 1: 3×3 max-pool (stride 2 to reduce spatial dims) to 1×1 conv
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
def modified_reduction_b(x, filters_b2_reduce, filters_b2,
                          filters_b3_reduce, filters_b3, filters_pool):
    # Branch 1: 3×3 max-pool (stride 2)
    b1 = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    b1 = conv_bn_relu(b1, filters_pool, 1)

    # Branch 2: 1×1 conv to 3×3 depthwise separable conv (stride 2)
    b2 = conv_bn_relu(x, filters_b2_reduce, 1)
    b2 = depthwise_separable_bn_relu(b2, filters_b2, 3, strides=2)

    # Branch 3: 1×1 conv to 3×3 depthwise separable conv (stride 2)
    b3 = conv_bn_relu(x, filters_b3_reduce, 1)
    b3 = depthwise_separable_bn_relu(b3, filters_b3, 3, strides=2)

    out = layers.Concatenate()([b1, b2, b3])
    return out


# Build full Keras model
#
# Structure:
#  1.  Input (256×256×3)
#  2.  4× Depthwise Separable Conv + BN + ReLU, 2× MaxPool (entry flow)
#  3.  [Inception A + Residual] × 3
#  4.  Reduction A
#  5.  [Inception B + Residual] × 3
#  6.  Reduction B
#  7.  Global Average Pooling
#  8.  Dropout
#  9.  Fully Connected (Softmax)
def build_model(num_classes, input_shape=(256, 256, 3), dropout_rate=0.5):
    # 1. Input
    inputs = layers.Input(shape=input_shape)

    # 2. Entry flow of four depthwise separable convolutions with two max pools
    x = depthwise_separable_bn_relu(inputs, 32, 3)
    x = depthwise_separable_bn_relu(x, 64, 3)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    x = depthwise_separable_bn_relu(x, 64, 3)
    x = depthwise_separable_bn_relu(x, 128, 3)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # 3. 3× Inception A with residual connection
    for _ in range(3):
        x = modified_inception_a(
            x,
            filters_1x1=40,
            filters_3x3_reduce=24, filters_3x3=48,
            filters_5x5_reduce=12, filters_5x5=24,
            filters_pool=24,
        )

    # 4. Reduction A
    x = modified_reduction_a(
        x,
        filters_3x3_reduce=48, filters_3x3=96,
        filters_dbl_reduce=48, filters_dbl=96,
        filters_pool=48,
    )

    # 5. 3× Inception B with residual connection
    for _ in range(3):
        x = modified_inception_b(
            x,
            filters_1x1=72,
            filters_7x7_reduce=40, filters_7x7=72,
            filters_7x7_dbl_reduce=40, filters_7x7_dbl=72,
            filters_pool=40,
        )

    # 6. Reduction B
    x = modified_reduction_b(
        x,
        filters_b2_reduce=64, filters_b2=128,
        filters_b3_reduce=64, filters_b3=128,
        filters_pool=64,
    )

    # 7. Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # 8. Dropout regularization
    x = layers.Dropout(dropout_rate)(x)

    # 9. Output layer with Softmax activation (multi-class classification)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model

# Main Block ensures parameter count is within 5% of paper's reported 428,100 parameters
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
        print("SUCCESS: Parameter count within 5% of paper target.")
    else:
        print("ERROR:  Parameter count differs by more than 5%.")

    print("\nBuilding model for Rice (4 classes)...")
    model_rice = build_model(num_classes=4)
    print(f"Rice model params: {model_rice.count_params():,}")

    print("\nBuilding model for Cassava (5 classes)...")
    model_cassava = build_model(num_classes=5)
    print(f"Cassava model params: {model_cassava.count_params():,}")