import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model

from Evaluate_Error import evaluat_error


def Model_InceptionV3(Train_Data, Train_Target, Test_Data, Test_Target, Epochs):
    IMG_SIZE = (299, 299, 3)
    NUM_CLASSES = Train_Target.shape[1]

    # Input layer
    inputs = Input(shape=IMG_SIZE)

    # Base model (pretrained on ImageNet)
    base_model = InceptionV3(
        include_top=False,  # remove the top layer
        weights='imagenet',
        input_tensor=inputs
    )
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.fit(Train_Data, Train_Target, epochs=Epochs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    pred = model.predict(Test_Data)
    Eval = evaluat_error(pred, Test_Target)

    return Eval, pred
