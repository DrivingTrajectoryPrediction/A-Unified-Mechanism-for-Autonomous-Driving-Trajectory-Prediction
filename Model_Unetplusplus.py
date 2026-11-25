import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from Evaluate_Error import net_evaluation


def conv_block(x, filters, kernel_size=(3, 3), padding='same', activation='relu'):
    x = Conv2D(filters, kernel_size, padding=padding, activation=activation)(x)
    x = Conv2D(filters, kernel_size, padding=padding, activation=activation)(x)
    return x


def build_unet_plus_plus(Activation, input_shape=(256, 256, 3), num_classes=1, deep_supervision=False):
    inputs = Input(input_shape)

    # Encoder
    x0_0 = conv_block(inputs, 64)
    p0 = MaxPooling2D((2, 2))(x0_0)

    x1_0 = conv_block(p0, 128)
    p1 = MaxPooling2D((2, 2))(x1_0)

    x2_0 = conv_block(p1, 256)
    p2 = MaxPooling2D((2, 2))(x2_0)

    x3_0 = conv_block(p2, 512)
    p3 = MaxPooling2D((2, 2))(x3_0)

    x4_0 = conv_block(p3, 1024)

    # Decoder / Nested skip connections
    x3_1 = conv_block(Concatenate()([x3_0, UpSampling2D((2, 2))(x4_0)]), 512)
    x2_2 = conv_block(Concatenate()([x2_0, UpSampling2D((2, 2))(x3_1)]), 256)
    x1_3 = conv_block(Concatenate()([x1_0, UpSampling2D((2, 2))(x2_2)]), 128)
    x0_4 = conv_block(Concatenate()([x0_0, UpSampling2D((2, 2))(x1_3)]), 64)

    # Optional intermediate outputs for deep supervision
    if deep_supervision:
        output1 = Conv2D(num_classes, (1, 1), activation=Activation)(x0_1)
        output2 = Conv2D(num_classes, (1, 1), activation='sigmoid')(x0_2)
        output3 = Conv2D(num_classes, (1, 1), activation='sigmoid')(x0_3)
        output4 = Conv2D(num_classes, (1, 1), activation='sigmoid')(x0_4)
        outputs = [output1, output2, output3, output4]
    else:
        outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(x0_4)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model


def Model_Unetplusplus(Image, GT, Activation):
    unetpp_model = build_unet_plus_plus(Activation)
    unetpp_model.summary()
    Pred = unetpp_model.predict(Image, GT)
    eval = net_evaluation(Pred, GT)

    return eval, Pred
