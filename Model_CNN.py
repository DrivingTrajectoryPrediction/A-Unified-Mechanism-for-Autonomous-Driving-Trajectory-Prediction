import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

from Evaluate_Error import evaluat_error


def Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target, Epoch):
    IMG_SIZE = 256
    Train_X = np.zeros((Train_Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Train_Data.shape[0]):
        temp = np.resize(Train_Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((Test_Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Test_Data.shape[0]):
        temp = np.resize(Test_Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    # Define the CNN model
    model = Sequential([
        Conv2D(64, kernel_size=2, activation='relu',
               input_shape=(Train_X.shape[1], Train_X.shape[2], Train_X.shape[3])),
        MaxPooling2D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.2),
        Dense(Train_Target.shape[1], activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(Train_X, Train_Target, epochs=Epoch, batch_size=4)
    pred = model.predict(Test_X)
    Eval = evaluat_error(pred, Test_Target)

    return Eval, pred



