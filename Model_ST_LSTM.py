import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from Evaluate_Error import evaluat_error


def swin_feature_extractor(input_shape=(224, 224, 3)):
    """
    Returns a Keras Model that outputs feature vector per frame
    """
    # Option 1: Using keras_cv or a small Swin model
    base_model = tf.keras.applications.InceptionV3(  # placeholder for Swin
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    return Model(inputs=base_model.input, outputs=x)


def build_swin_lstm_model(sol, input_shape=(224, 224, 3), lstm_units=128, num_classes=5):
    """
    seq_len: number of frames in the sequence
    input_shape: shape of each frame
    """
    act = ['linear', 'relu', 'tanh', 'softmax', 'sigmoid']
    seq_input = Input(shape=input_shape)

    # Use TimeDistributed to apply Swin feature extractor to each frame
    swin_model = swin_feature_extractor(input_shape)
    x = TimeDistributed(swin_model)(seq_input)  # (batch, seq_len, feature_dim)

    # LSTM over sequence
    x = LSTM(lstm_units)(x)

    # Classification
    output = Dense(num_classes, activation=act[sol[2]])(x)

    model = Model(seq_input, output)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def Model_ST_LSTM(Train_Data, Train_Target, Test_Data, Test_Target, Epochs, sol=None):
    IMG_SIZE = 256
    Train_X = np.zeros((Train_Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Train_Data.shape[0]):
        temp = np.resize(Train_Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((Test_Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Test_Data.shape[0]):
        temp = np.resize(Test_Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))
    if sol is None:
        sol = [5, 0.01, 1]
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    model = build_swin_lstm_model(sol, input_shape, lstm_units=128, num_classes=Train_Target.shape[1])
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=sol[1]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(Train_X, Train_Target, epochs=Epochs)

    model.summary()
    pred = model.predict(Test_X)
    Eval = evaluat_error(pred, Test_Target)

    return Eval, pred

