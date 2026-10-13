import tensorflow as tf
from tensorflow.keras import layers, models



def build_model():
    inputs = layers.Input(shape=(160, 160, 3))

    # Input [0, 1] -> MobileNet nhận [-1, 1]
    x = layers.Rescaling(scale=2.0, offset=-1.0)(inputs)

    base_network = tf.keras.applications.MobileNet(
        input_shape=(160, 160, 3),
        include_top=False,
        weights="imagenet",
        name="mobilenet_backbone",
    )

    base_network.trainable = False

    x = base_network(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(
        inputs, outputs, name="Fine_Tuning_MobileNet"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    return model


def fine_tune_model(model, unfreeze_from="conv_pw_13_relu", learning_rate=1e-5): #lr của đoạn này nhỏ hơn đoạn trên 10 lần vì cập nhật mạnh quá thì có thể bị phá vỡ kiến thức mô hình đã học
    """Mở khóa 1 block hoặc mở một phần backbone của MobileNet để nó cập nhật trọng số cho phù hợp hơn"""
    backbone = model.get_layer("mobilenet_backbone")

    for layer in backbone.layers: #Đoạn này là khóa toàn bộ trước đã
        layer.trainable = False

    should_unfreeze = False #Mở từ layer được chọn tính về sau ấy
    for layer in backbone.layers:
        if layer.name == unfreeze_from:
            should_unfreeze = True
        if should_unfreeze:
            layer.trainable = True

    if not should_unfreeze : #Trường hợp không tìm thấy thì mở 20 layer cuối
        for layer in backbone.layers[-20:]:
            layer.trainable = True
            
#Biên dịch lại model với learning rate nhỏ
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), #cách cập nhật trọng số, sử dụng Adam
        loss="binary_crossentropy", #mức độ dự đoán sai
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")], #chỉ số để đánh giá mô hình
    )
    return model
