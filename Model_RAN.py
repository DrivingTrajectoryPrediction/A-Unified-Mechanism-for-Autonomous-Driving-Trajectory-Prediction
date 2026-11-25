import numpy as np
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense
from keras.models import Model

from Evaluate_Error import evaluat_error


def residual_block(input_tensor, filters, dilation_rate=1):
    """
    Residual block with dilated convolutions.
    """
    x = Conv2D(filters, (3, 3), padding='same', dilation_rate=dilation_rate)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding='same', dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)

    # Residual connection
    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x


def dilated_residual_attention_network(Epoch, input_shape, num_classes=1):
    """
    Builds a Dilated Residual Attention Network for 3D image classification.
    """
    inputs = Input(shape=input_shape)

    # Initial Conv Layer
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = residual_block(x, 32, dilation_rate=1)

    # Further Convolutions for feature extraction
    x = Conv2D(64, (3, 3), padding='same', dilation_rate=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', dilation_rate=4)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Global Pooling and Output
    x = GlobalAveragePooling2D()(x)

    # Classification Layer
    if num_classes == 1:
        outputs = Dense(num_classes, activation='sigmoid')(x)  # Binary classification
    else:
        outputs = Dense(num_classes, activation='softmax')(x)  # Multi-class classification

    model = Model(inputs, outputs)
    return model


def Model_RAN(Train_Data, Train_Target, Test_data, Test_Target, Epoch):
    IMG_SIZE = 16
    Feat1 = np.zeros((Train_Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Train_Data.shape[0]):
        temp = np.resize(Train_Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Feat1[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Feat2 = np.zeros((Test_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Test_data.shape[0]):
        temp = np.resize(Test_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Feat2[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    input_shape = (16, 16, 3)  # Add channel dimension for grayscale
    # Instantiate the model
    model = dilated_residual_attention_network(Epoch, input_shape, num_classes=Train_Target.shape[1])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Print the model summary
    model.summary()
    model.fit(Feat1, Train_Target, epochs=Epoch)
    pred = model.predict(Feat2)
    Eval = evaluat_error(pred, Test_Target)
    return np.asarray(Eval), pred

