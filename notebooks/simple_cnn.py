import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout)

model = Sequential()

model.add(Input(shape=(224, 224, 3)))

# 3. First Convolutional Block
# Extract features using 32 filters of size 3 x 3
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))

# Reduce spatial dimensions by half
model.add(MaxPooling2D(pool_size=(2, 2)))

# 4. Second Convolutional Block
# Extract more complex features using 64 filters
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))

# Reduce spatial dimensions by half
model.add(MaxPooling2D(pool_size=(2, 2)))

# 5. Third Convolutional Block
# Extract higher-level features using 128 filters
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))

# Reduce spatial dimensions by half
model.add(MaxPooling2D(pool_size=(2, 2)))

# 6. Global Average Pooling
# Convert feature maps into a 128-value vector
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
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

# 10. Display the model architecture
model.summary()