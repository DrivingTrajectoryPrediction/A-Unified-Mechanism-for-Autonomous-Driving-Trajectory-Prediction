import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, GlobalAveragePooling2D, Dense, \
    AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from Evaluate_Error import net_evaluation


# Dense block
def dense_block(x, num_layers, growth_rate):
    concat_features = [x]
    for i in range(num_layers):
        bn = BatchNormalization()(Concatenate()(concat_features))
        relu = ReLU()(bn)
        conv = Conv2D(growth_rate, (3, 3), padding='same')(relu)
        concat_features.append(conv)
    x = Concatenate()(concat_features)
    return x


# Transition layer
def transition_layer(x, reduction=0.5):
    filters = int(x.shape[-1] * reduction)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, (1, 1), padding='same')(x)
    x = AveragePooling2D((2, 2), strides=2)(x)
    return x


# DenseNet Model
def build_densenet(Activation, input_shape=(224, 224, 3), num_classes=1000, growth_rate=32, block_layers=None):
    if block_layers is None:
        block_layers = [6, 12, 24, 16]
    inputs = Input(shape=input_shape)

    # Initial convolution
    x = Conv2D(64, (7, 7), strides=2, padding='same', activation=Activation)(inputs)
    x = AveragePooling2D((3, 3), strides=2, padding='same')(x)

    # Dense blocks with transition layers
    for i, layers in enumerate(block_layers):
        x = dense_block(x, layers, growth_rate)
        if i != len(block_layers) - 1:  # No transition after last dense block
            x = transition_layer(x)

    # Classification head
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def Model_DenseNet(Image, GT, Activation):
    densenet_model = build_densenet(Activation, input_shape=(224, 224, 3), num_classes=10)
    densenet_model.summary()
    Pred = densenet_model.predict(Image, GT)
    eval = net_evaluation(Pred, GT)

    return eval, Pred
