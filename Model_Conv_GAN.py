import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Activation, Flatten, \
    Dense, Reshape, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from Evaluate_Error import Image_Evaluation


def add_fog(image, fog_intensity=0.5):
    h, w, c = image.shape
    fog = np.random.normal(loc=255, scale=30, size=(h, w, 1)).astype(np.uint8)
    fog = cv2.merge([fog] * 3)
    foggy_img = cv2.addWeighted(image, 1 - fog_intensity, fog, fog_intensity, 0)
    return foggy_img


def add_rain(image, drops=1000):
    rain_img = image.copy()
    h, w, _ = image.shape
    for _ in range(drops):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        length = np.random.randint(5, 15)
        cv2.line(rain_img, (x, y), (x, y + length), (200, 200, 200), 1)
    rain_img = cv2.blur(rain_img, (3, 3))
    return rain_img


def darken(image, factor=0.5):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)


def load_and_augment(image_path):
    img = image_path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img

    foggy = add_fog(img, 0.6)
    rainy = add_rain(img, 500)
    night = darken(img, 0.3)

    # Normalize to [-1,1] for GAN
    original = img.astype(np.float32) / 127.5 - 1.0
    foggy = foggy.astype(np.float32) / 127.5 - 1.0
    rainy = rainy.astype(np.float32) / 127.5 - 1.0
    night = night.astype(np.float32) / 127.5 - 1.0

    effect_images = np.array([foggy, rainy, night], dtype=np.float32)
    return original, effect_images


# -----------------------------
# Generator (Convolutional)
# -----------------------------
def build_generator(latent_dim=100, img_shape=(64, 64, 3)):
    inputs = Input(shape=(latent_dim,))
    x = Dense(8 * 8 * 128)(inputs)
    x = Reshape((8, 8, 128))(x)
    x = Conv2DTranspose(128, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(64, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(32, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    outputs = Conv2DTranspose(3, 3, strides=1, padding='same', activation='tanh')(x)
    return Model(inputs, outputs, name="Generator")


def build_discriminator(img_shape=(64, 64, 3)):
    inputs = Input(shape=img_shape)
    x = Conv2D(64, 4, strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(256, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name="Discriminator")


def build_gan(generator, discriminator):
    discriminator.trainable = False
    latent = Input(shape=(100,))
    img = generator(latent)
    validity = discriminator(img)
    return Model(latent, validity, name="GAN")


def get_cgan_models_and_effects(image_path, LearningRate, latent_dim=100):
    original, effects = load_and_augment(image_path)
    generator = build_generator(latent_dim=latent_dim)
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    discriminator.summary()
    gan.summary()
    discriminator.compile(optimizer=Adam(LearningRate, 0.5), loss='binary_crossentropy', metrics=['accuracy'])
    gan.compile(optimizer=Adam(LearningRate, 0.5), loss='binary_crossentropy')
    return generator, discriminator, effects


def Model_Conv_GAN(Image, LearningRate):
    Effect = []
    for n in range(len(Image)):
        generator, discriminator, effect_images = get_cgan_models_and_effects(Image[n], LearningRate)
    arr = np.asarray(Effect)
    ReEffect = np.resize(arr, (arr.shape[0] * arr.shape[1], arr.shape[2], arr.shape[3], arr.shape[4]))
    Eval = Image_Evaluation(Image[0], ReEffect[0])

    return Eval, ReEffect
