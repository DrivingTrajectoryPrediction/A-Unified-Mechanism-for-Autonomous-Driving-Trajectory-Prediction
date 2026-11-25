import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from Evaluate_Error import net_evaluation


def build_fcn(Activation, input_shape=(256, 256, 3), num_classes=1):
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = Conv2D(64, (3,3), activation=Activation, padding='same')(inputs)
    conv1 = Conv2D(64, (3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2,2))(conv1)

    conv2 = Conv2D(128, (3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2,2))(conv2)

    conv3 = Conv2D(256, (3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2,2))(conv3)

    conv4 = Conv2D(512, (3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3,3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D((2,2))(conv4)

    # Bottleneck
    conv5 = Conv2D(1024, (3,3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3,3), activation='relu', padding='same')(conv5)

    # Decoder / Upsampling
    up6 = Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(conv5)
    add6 = Add()([up6, conv4])
    conv6 = Conv2D(512, (3,3), activation='relu', padding='same')(add6)
    conv6 = Conv2D(512, (3,3), activation='relu', padding='same')(conv6)

    up7 = Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(conv6)
    add7 = Add()([up7, conv3])
    conv7 = Conv2D(256, (3,3), activation='relu', padding='same')(add7)
    conv7 = Conv2D(256, (3,3), activation='relu', padding='same')(conv7)

    up8 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(conv7)
    add8 = Add()([up8, conv2])
    conv8 = Conv2D(128, (3,3), activation='relu', padding='same')(add8)
    conv8 = Conv2D(128, (3,3), activation='relu', padding='same')(conv8)

    up9 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(conv8)
    add9 = Add()([up9, conv1])
    conv9 = Conv2D(64, (3,3), activation='relu', padding='same')(add9)
    conv9 = Conv2D(64, (3,3), activation='relu', padding='same')(conv9)

    # Final classifier layer
    outputs = Conv2D(num_classes, (1,1), activation='sigmoid')(conv9)  # Binary segmentation

    # Build model
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def Model_FCN(Image, GT, Activation):
    fcn_model = build_fcn(Activation)
    fcn_model.summary()
    Pred = fcn_model.predict(Image, GT)
    eval = net_evaluation(Pred, GT)

    return eval, Pred
