import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Concatenate, \
    GlobalAveragePooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from Evaluate_Error import net_evaluation


# Improved Loss Function (MSE + MAE)
def improved_loss(y_true, y_pred):
    n = tf.cast(tf.size(y_true), tf.float32)
    mse = tf.reduce_sum(tf.square(y_true - y_pred)) / (2 * n)
    mae = tf.reduce_sum(tf.abs(y_true - y_pred)) / n
    return mse + mae


# Residual Dense Block (functional)
def residual_dense_block(x, filters, growth_rate=32, layers=4):
    concat_features = [x]
    for i in range(layers):
        out = Conv2D(growth_rate, (3, 3), padding='same', activation='relu')(Concatenate()(concat_features))
        concat_features.append(out)
    out = Conv2D(filters, (1, 1), padding='same')(Concatenate()(concat_features))
    out = Add()([x, out])
    return out


# Pyramid Pooling Module (simplified)
def pyramid_pooling_module(x, pool_sizes=[1, 2, 3, 6]):
    h, w = x.shape[1], x.shape[2]
    pooled_outputs = [x]
    for size in pool_sizes:
        pool = tf.keras.layers.AveragePooling2D(pool_size=(h // size, w // size), strides=(h // size, w // size))(x)
        up = UpSampling2D(size=(h // size, w // size), interpolation='bilinear')(pool)
        pooled_outputs.append(up)
    return Concatenate()(pooled_outputs)


# Build a simple functional segmentation network
def build_segmentation_model(Activation, input_shape=(256, 256, 3), num_classes=1):
    inputs = Input(shape=input_shape)

    # Encoder
    x = Conv2D(64, (3, 3), padding='same', activation=Activation)(inputs)
    x = residual_dense_block(x, 64)

    # Pyramid pooling
    x = pyramid_pooling_module(x)

    # Decoder
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(num_classes, (1, 1), activation='sigmoid')(x)  # For binary segmentation

    model = Model(inputs, x)
    model.compile(optimizer=Adam(1e-4), loss=improved_loss, metrics=['accuracy'])

    return model


def Model_RDNet(Image, GT, Activation):
    model = build_segmentation_model(Image, GT, Activation)
    model.summary()
    Pred = model.predict(Image, GT)
    eval = net_evaluation(Pred, GT)

    return eval, Pred

