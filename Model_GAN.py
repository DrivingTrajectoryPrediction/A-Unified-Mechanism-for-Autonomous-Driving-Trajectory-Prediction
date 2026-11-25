import cv2
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, \
    BatchNormalization
from tensorflow.keras.models import Model

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
    # Load original image
    img = image_path
    img = img

    # Apply effects
    foggy = add_fog(img, 0.6)
    rainy = add_rain(img, 1500)
    night = darken(img, 0.3)

    # Normalize images to 0-1 for GAN
    original = img.astype(np.float32) / 255.0
    foggy = foggy.astype(np.float32) / 255.0
    rainy = rainy.astype(np.float32) / 255.0
    night = night.astype(np.float32) / 255.0

    effect_images = np.array([foggy, rainy, night], dtype=np.float32)

    return original, effect_images


def build_generator(sol, input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)
    act = ['linear', 'relu', 'softmax', 'sigmoid', 'tanh']
    # Encoder
    e1 = Conv2D(64, kernel_size=4, activation=act[sol[2]], strides=2, padding='same')(inputs)
    e1 = LeakyReLU(alpha=0.2)(e1)
    e2 = Conv2D(128, kernel_size=4, strides=2, padding='same')(e1)
    e2 = BatchNormalization()(e2)
    e2 = LeakyReLU(alpha=0.2)(e2)
    e3 = Conv2D(256, kernel_size=4, strides=2, padding='same')(e2)
    e3 = BatchNormalization()(e3)
    e3 = LeakyReLU(alpha=0.2)(e3)

    # Decoder with skip connections (attention)
    d1 = Conv2DTranspose(sol[0], kernel_size=4, strides=2, padding='same')(e3)
    d1 = BatchNormalization()(d1)
    d1 = Activation('relu')(d1)
    d1 = Concatenate()([d1, e2])

    d2 = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(d1)
    d2 = BatchNormalization()(d2)
    d2 = Activation('relu')(d2)
    d2 = Concatenate()([d2, e1])

    outputs = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh')(d2)

    return Model(inputs, outputs, name="Generator")


def build_discriminator(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=4, strides=2, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(1, kernel_size=4, padding='same', activation='sigmoid')(x)
    return Model(inputs, x, name="Discriminator")


def get_model_and_images(image_path, sol):
    original_img, effect_imgs = load_and_augment(image_path)
    generator = build_generator(sol)
    discriminator = build_discriminator()
    return generator, discriminator, original_img, effect_imgs


def Model_GAN(Image, LearningRate, sol=None):
    if sol is None:
        sol = [5, 0.01, 1]
    Effect = []
    for n in range(len(Image)):
        generator, discriminator, original, effects = get_model_and_images(Image[n], sol)
        Effect.append(effects)
        generator.summary()
        discriminator.summary()
        discriminator.compile(optimizer=Adam(learning_rate=sol[1]), loss='categorical_crossentropy', metrics=['accuracy'])
    arr = np.asarray(Effect)
    ReEffect = np.resize(arr, (arr.shape[0] * arr.shape[1], arr.shape[2], arr.shape[3], arr.shape[4]))
    Eval = Image_Evaluation(Image[0], ReEffect[0])

    return Eval, ReEffect
