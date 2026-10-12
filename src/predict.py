import pandas as pd
import tensorflow as tf
from data.preprocess import preprocess_image

#model complex_cnn
model = tf.keras.models.load_model("src/models/complex_cnn.keras")
df = pd.read_csv("../data/test_split.csv")

image = preprocess_image("../data/" + df.iloc[0]["file_name"])[None, ...]
prob = model.predict(image, verbose=0)[0, 0]

print("Xác suất ảnh AI gen:", round(float(prob), 4))
print("Kết quả:", "AI-gen" if prob >= 0.5 else "Real")