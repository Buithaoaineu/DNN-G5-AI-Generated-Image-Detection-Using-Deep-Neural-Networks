import tensorflow as tf
from tensorflow.keras import layers, models


def build_model():
    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(inputs)

    base_network = tf.keras.applications.MobileNet(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        name="mobilenet_backbone",
    )
    base_network.trainable = False

    x = base_network(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="Fine_Tuning_MobileNet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def fine_tune_model(model, unfreeze_from="conv_pw_13_relu", learning_rate=1e-5):
    """Unfreeze one block of the pretrained MobileNet backbone for fine-tuning."""
    if not isinstance(model, tf.keras.Model):
        raise TypeError("model must be a tf.keras.Model instance.")

    backbone = model.get_layer("mobilenet_backbone")
    if backbone is None:
        raise ValueError("The model must contain a layer named 'mobilenet_backbone'.")

    for layer in backbone.layers:
        layer.trainable = False

    should_unfreeze = False
    for layer in backbone.layers:
        if layer.name == unfreeze_from:
            should_unfreeze = True
        if should_unfreeze:
            layer.trainable = True

    if not any(layer.trainable for layer in backbone.layers):
        for layer in backbone.layers[-20:]:
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model
