import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, \
    BatchNormalization, Embedding, Multiply, Flatten, Dense
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
    rainy = add_rain(img, 1500)
    night = darken(img, 0.3)

    # Normalize
    original = img.astype(np.float32) / 255.0
    foggy = foggy.astype(np.float32) / 255.0
    rainy = rainy.astype(np.float32) / 255.0
    night = night.astype(np.float32) / 255.0

    # Effects with labels: 0=fog,1=rain,2=night
    effect_images = np.array([foggy, rainy, night], dtype=np.float32)
    labels = np.array([0, 1, 2], dtype=np.int32)
    return original, effect_images, labels


def build_generator(input_shape=(256, 256, 3), n_conditions=3):
    # Image input
    img_input = Input(shape=input_shape)
    # Condition input
    cond_input = Input(shape=(1,))
    cond_embedding = Embedding(n_conditions, np.prod(input_shape))(cond_input)
    cond_embedding = tf.reshape(cond_embedding, (-1, input_shape[0], input_shape[1], input_shape[2]))
    x = Concatenate()([img_input, cond_embedding])

    # Encoder
    e1 = Conv2D(64, 4, strides=2, padding='same')(x)
    e1 = LeakyReLU(0.2)(e1)
    e2 = Conv2D(128, 4, strides=2, padding='same')(e1)
    e2 = BatchNormalization()(e2)
    e2 = LeakyReLU(0.2)(e2)
    e3 = Conv2D(256, 4, strides=2, padding='same')(e2)
    e3 = BatchNormalization()(e3)
    e3 = LeakyReLU(0.2)(e3)

    # Decoder with skip connections
    d1 = Conv2DTranspose(128, 4, strides=2, padding='same')(e3)
    d1 = BatchNormalization()(d1)
    d1 = Activation('relu')(d1)
    d1 = Concatenate()([d1, e2])
    d2 = Conv2DTranspose(64, 4, strides=2, padding='same')(d1)
    d2 = BatchNormalization()(d2)
    d2 = Activation('relu')(d2)
    d2 = Concatenate()([d2, e1])
    outputs = Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')(d2)

    return Model([img_input, cond_input], outputs, name="ConditionalGenerator")


def build_discriminator(input_shape=(256, 256, 3), n_conditions=3):
    img_input = Input(shape=input_shape)
    cond_input = Input(shape=(1,))
    cond_embedding = Embedding(n_conditions, np.prod(input_shape))(cond_input)
    cond_embedding = tf.reshape(cond_embedding, (-1, input_shape[0], input_shape[1], input_shape[2]))
    x = Concatenate()([img_input, cond_embedding])

    x = Conv2D(64, 4, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(1, 4, padding='same', activation='sigmoid')(x)
    return Model([img_input, cond_input], x, name="ConditionalDiscriminator")


def get_conditional_cgan(image_path, learning_rate=0.0002):
    original, effects, labels = load_and_augment(image_path)

    generator = build_generator()
    discriminator = build_discriminator()

    generator.summary()
    discriminator.summary()

    discriminator.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    return generator, discriminator, original, effects, labels


def Model_Con_GAN(Image, LearningRate):
    Effect = []
    for n in range(len(Image)):
        generator, discriminator, original, effects, labels = get_conditional_cgan(Image)
        discriminator.compile(optimizer=Adam(learning_rate=LearningRate), loss='categorical_crossentropy',
                              metrics=['accuracy'])

    arr = np.asarray(Effect)
    ReEffect = np.resize(arr, (arr.shape[0] * arr.shape[1], arr.shape[2], arr.shape[3], arr.shape[4]))
    Eval = Image_Evaluation(Image[0], ReEffect[0])

    return Eval, ReEffect
