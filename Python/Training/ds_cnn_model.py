"""
DS-CNN Model with SE Attention — Phase 3
==========================================
Depthwise Separable CNN + Squeeze-and-Excitation attention blocks
for wake word detection on STM32H7.

Architecture:
  Input (97, 40, 1)
    → Conv2D + BN + ReLU                   (feature expansion)
    → 4× [DepthwiseConv2D + BN + ReLU
           → SE block
           → Conv2D 1×1 + BN + ReLU]       (DS-Conv blocks)
    → GlobalAveragePooling2D
    → Dropout
    → Dense(2, softmax)
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers

from train_config import (
    INPUT_SHAPE,
    FIRST_CONV_FILTERS, FIRST_CONV_KERNEL,
    DS_CONV_BLOCKS, DS_CONV_KERNEL,
    SE_REDUCTION_RATIO,
    DROPOUT_RATE, NUM_CLASSES,
    SPEC_AUGMENT, MAX_TIME_MASK, MAX_FREQ_MASK,
)


# ── SpecAugment Layer ──────────────────────────────────────────────

class SpecAugment(layers.Layer):
    """Applies random time and frequency masking to MFCC spectrograms.

    Active during training only.  Zeros out a random contiguous band
    of time frames and/or frequency bins, forcing the model to not
    rely on any single time-frequency region.

    Input shape : (batch, time, freq, 1)
    Output shape: same
    """

    def __init__(self, max_time_mask=10, max_freq_mask=5, **kwargs):
        super().__init__(**kwargs)
        self.max_time_mask = max_time_mask
        self.max_freq_mask = max_freq_mask

    def call(self, x, training=False):
        if not training:
            return x

        shape = tf.shape(x)
        batch, time_steps, freq_bins = shape[0], shape[1], shape[2]

        # ── Time mask ──────────────────────────────────────────────
        t_len = tf.random.uniform([], 0, self.max_time_mask + 1, dtype=tf.int32)
        t_start = tf.random.uniform([], 0, time_steps - t_len + 1, dtype=tf.int32)
        t_mask = tf.concat([
            tf.ones([batch, t_start, freq_bins, 1]),
            tf.zeros([batch, t_len, freq_bins, 1]),
            tf.ones([batch, time_steps - t_start - t_len, freq_bins, 1]),
        ], axis=1)

        # ── Frequency mask ─────────────────────────────────────────
        f_len = tf.random.uniform([], 0, self.max_freq_mask + 1, dtype=tf.int32)
        f_start = tf.random.uniform([], 0, freq_bins - f_len + 1, dtype=tf.int32)
        f_mask = tf.concat([
            tf.ones([batch, time_steps, f_start, 1]),
            tf.zeros([batch, time_steps, f_len, 1]),
            tf.ones([batch, time_steps, freq_bins - f_start - f_len, 1]),
        ], axis=2)

        return x * t_mask * f_mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_time_mask": self.max_time_mask,
            "max_freq_mask": self.max_freq_mask,
        })
        return config


# ── SE (Squeeze-and-Excitation) Block ──────────────────────────────

def se_block(x, reduction=SE_REDUCTION_RATIO):
    """Channel attention: learn to weight each feature map by importance.

    GlobalAvgPool → Dense(C/r, ReLU) → Dense(C, Sigmoid) → scale.
    Cost: 2*C*C/r parameters (e.g., 2*64*16 = 2048 params for C=64, r=4).
    """
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)                          # (B, C)
    se = layers.Dense(channels // reduction, activation="relu")(se)  # (B, C/r)
    se = layers.Dense(channels, activation="sigmoid")(se)            # (B, C)
    se = layers.Reshape((1, 1, channels))(se)                        # (B, 1, 1, C)
    return x * se                                                    # scale


# ── DS-Conv Block ──────────────────────────────────────────────────

def ds_conv_block(x, filters, kernel_size, use_se=True):
    """Depthwise Separable Convolution block + optional SE attention.

    DepthwiseConv2D → BN → ReLU → [SE] → Conv2D 1×1 → BN → ReLU
    """
    # Depthwise convolution
    x = layers.DepthwiseConv2D(
        kernel_size, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # SE attention
    if use_se:
        x = se_block(x)

    # Pointwise convolution (1×1)
    x = layers.Conv2D(filters, (1, 1), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x


# ── Full Model ─────────────────────────────────────────────────────

def build_ds_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """Build DS-CNN + SE attention model for keyword spotting.

    Returns a compiled-ready Keras model.
    """
    inputs = keras.Input(shape=input_shape, name="mfcc_input")
    x = inputs

    # ── SpecAugment (training only) ────────────────────────────────
    if SPEC_AUGMENT:
        x = SpecAugment(
            max_time_mask=MAX_TIME_MASK,
            max_freq_mask=MAX_FREQ_MASK,
            name="spec_augment",
        )(x)

    # ── First conv: expand channels ────────────────────────────────
    x = layers.Conv2D(
        FIRST_CONV_FILTERS, FIRST_CONV_KERNEL,
        padding="same", use_bias=False,
        name="conv_expand",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # ── DS-Conv blocks ─────────────────────────────────────────────
    for i in range(DS_CONV_BLOCKS):
        x = ds_conv_block(
            x,
            filters=FIRST_CONV_FILTERS,
            kernel_size=DS_CONV_KERNEL,
            use_se=True,
        )

    # ── Classification head ────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs, outputs, name="DS_CNN_SE")
    return model


# ── Quick test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    model = build_ds_cnn()
    model.summary()

    # Count parameters
    total = model.count_params()
    print(f"\nTotal parameters : {total:,}")
    print(f"Float32 size     : {total * 4 / 1024:.1f} KB")
    print(f"INT8 size (est.) : {total / 1024:.1f} KB")
