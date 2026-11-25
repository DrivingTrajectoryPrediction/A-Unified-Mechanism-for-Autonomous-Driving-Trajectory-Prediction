# ============================================================
# Regional Attention-based Adaptive Conditional CycleGAN (RA-ACCycleGAN)
# Functional Implementation (Kaggle-ready)
# Author: Chebees (Modified for Image_1.npy input)
# ============================================================

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Activation, Concatenate, LeakyReLU, \
    BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1️⃣ Regional Attention Block
# ============================================================
def regional_attention(x, filters):
    avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    dense = Conv2D(filters, kernel_size=1, activation='sigmoid', padding='same')(avg_pool)
    return x * dense


# ============================================================
# 2️⃣ Generator (U-Net with Regional Attention)
# ============================================================
def build_generator(input_shape=(512, 512, 3)):
    inputs = Input(shape=input_shape)

    # Encoder
    e1 = Conv2D(64, 4, strides=2, padding='same')(inputs)
    e1 = LeakyReLU(0.2)(e1)

    e2 = Conv2D(128, 4, strides=2, padding='same')(e1)
    e2 = BatchNormalization()(e2)
    e2 = LeakyReLU(0.2)(e2)

    e3 = Conv2D(256, 4, strides=2, padding='same')(e2)
    e3 = BatchNormalization()(e3)
    e3 = LeakyReLU(0.2)(e3)

    e4 = Conv2D(512, 4, strides=2, padding='same')(e3)
    e4 = BatchNormalization()(e4)
    e4 = LeakyReLU(0.2)(e4)

    # Bottleneck + Regional Attention
    b = Conv2D(512, 4, strides=2, padding='same')(e4)
    b = Activation('relu')(b)
    b = regional_attention(b, 512)

    # Decoder
    d1 = Conv2DTranspose(512, 4, strides=2, padding='same')(b)
    d1 = BatchNormalization()(d1)
    d1 = Activation('relu')(d1)
    d1 = Concatenate()([d1, e4])

    d2 = Conv2DTranspose(256, 4, strides=2, padding='same')(d1)
    d2 = BatchNormalization()(d2)
    d2 = Activation('relu')(d2)
    d2 = Concatenate()([d2, e3])

    d3 = Conv2DTranspose(128, 4, strides=2, padding='same')(d2)
    d3 = BatchNormalization()(d3)
    d3 = Activation('relu')(d3)
    d3 = Concatenate()([d3, e2])

    d4 = Conv2DTranspose(64, 4, strides=2, padding='same')(d3)
    d4 = BatchNormalization()(d4)
    d4 = Activation('relu')(d4)
    d4 = Concatenate()([d4, e1])

    outputs = Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')(d4)
    return Model(inputs, outputs, name="Generator")


# ============================================================
# 3️⃣ Discriminator (PatchGAN)
# ============================================================
def build_discriminator(input_shape=(512, 512, 3)):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, 4, strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(256, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(512, 4, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    outputs = Conv2D(1, 4, strides=1, padding='same')(x)
    return Model(inputs, outputs, name="Discriminator")


# ============================================================
# 4️⃣ Adaptive Loss (simplified hybrid)
# ============================================================
def cycle_loss(real, cycled):
    return tf.reduce_mean(tf.abs(real - cycled))


def identity_loss(real, same):
    return tf.reduce_mean(tf.abs(real - same))


# ============================================================
# 5️⃣ Load Image_1.npy and Generate Synthetic Image
# ============================================================
def generate_synthetic_from_npy(generator, npy_path='Image_1.npy'):
    # Load and normalize
    images = np.load(npy_path)[:10, :]
    images = (images.astype('float32') / 127.5) - 1.0

    print("Loaded image shape:", images.shape)

    # Predict (Generate synthetic image)
    synthetic = generator.predict(images)
    synthetic = (synthetic + 1.0) / 2.0  # Rescale to [0,1]

    # Display the first generated image
    plt.figure(figsize=(4, 4))
    plt.imshow(synthetic[0])
    plt.axis('off')
    plt.title('Generated Synthetic Image')
    plt.show()

    return synthetic


# ============================================================
# 6️⃣ Main Execution
# ============================================================
if __name__ == "__main__":
    print("Building RA-ACCycleGAN Generator and Discriminator...")

    gen_G = build_generator()
    disc_X = build_discriminator()

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    gen_G.compile(optimizer=opt, loss='mse')
    disc_X.compile(optimizer=opt, loss='mse')

    print("Models built successfully!")

    # Generate and display output
    synthetic_images = generate_synthetic_from_npy(gen_G, 'Image_1.npy')
    print("Synthetic Images Generated:", synthetic_images.shape)
