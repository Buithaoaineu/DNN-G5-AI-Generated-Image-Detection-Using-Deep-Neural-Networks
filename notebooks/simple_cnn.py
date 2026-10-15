from pyexpat import model

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout)


def build_simple_cnn(input_shape=(160, 160, 3)):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(GlobalAveragePooling2D())

# 7. Fully Connected Layer
# Combine extracted features for classification
    model.add(Dense(128, activation='relu'))

# Reduce overfitting by randomly dropping 50% of units during training
    model.add(Dropout(0.5))

# 8. Output Layer
# Predict the probability that the image is AI-generated
    model.add(Dense(1, activation='sigmoid'))

# 9. Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy',
    metrics=['accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    return model