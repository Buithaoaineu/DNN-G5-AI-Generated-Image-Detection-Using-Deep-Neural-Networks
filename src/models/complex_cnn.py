#Import thư viện
import tensorflow as tf
from tensorflow.keras import layers, models


#Xây dựng residual block
def residual_block(x, filters, stride=1):
    shortcut = x

    x = layers.Conv2D(filters, 3, strides=stride, padding="same",
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same",
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride,
                                 padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    return layers.ReLU()(layers.Add()([x, shortcut]))


#CNN 4 khối
def build_model():
    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for filters, stride in [(32, 1), (64, 2), (128, 2), (256, 2)]:
        x = residual_block(x, filters, stride)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="Complex_CNN_4_Blocks")

    #Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model