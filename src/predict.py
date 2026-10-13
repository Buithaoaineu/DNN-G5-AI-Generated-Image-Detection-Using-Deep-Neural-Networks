import pandas as pd
import tensorflow as tf
import sys
sys.path.append("..")
from src.data.preprocess import preprocess_image
# Tải model đã huấn luyện
models = {
    "Simple CNN": "models/simple_cnn.keras",
    "Complex CNN": "models/complex_cnn.keras",
    "Fine Tuning": "models/fine_tuning.keras"
}

# Đọc dữ liệu test
df = pd.read_csv("data/test_split.csv")

# Tiền xử lý ảnh và dự đoán
image = preprocess_image("data/" + df.iloc[0]["file_name"])[None, ...]

for name, path in models.items():
    model = tf.keras.models.load_model(path)
    prob = model.predict(image, verbose=0)[0, 0]
    print(f"{name}: AI={prob:.4f} | {'AI-gen' if prob >= 0.5 else 'Real'}")
