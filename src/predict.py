import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import sys

sys.path.append("..")
from src.data.preprocess import preprocess_image

# Tải các model đã huấn luyện
models = {
    "Simple CNN": "models/simple_cnn.keras",
    "Complex CNN": "models/complex_cnn.keras",
    "Fine Tuning": "models/fine_tuned_mobilenet.keras"
}

# Đọc dữ liệu test
df = pd.read_csv("data/test_split.csv")

# Chọn ảnh để predict trên 3 models
index = 0
file_name = df.iloc[index]["file_name"]
image_path = os.path.join("data", file_name)

# Hiển thị ảnh gốc
original_image = Image.open(image_path).convert("RGB")

# Tiền xử lý ảnh cho các model CNN thông thường
image = preprocess_image(image_path)[None, ...]

# Hiển thị ảnh
plt.figure(figsize=(6, 6))
plt.imshow(original_image)
plt.title(f"Ảnh được dự đoán")
plt.axis("off")
plt.show()

# Dự đoán bằng từng model
for name, path in models.items():
    model = tf.keras.models.load_model(path)

    # MobileNetV2 cần đầu vào [-1, 1] nếu preprocess_image() trả ảnh RGB trong [0, 1]
    model_image = image.copy()

    if name == "Fine Tuning":
        model_image = model_image * 2.0 - 1.0

    prob = float(model.predict(model_image, verbose=0)[0, 0])
    result = "AI-gen" if prob >= 0.5 else "Real"

    print(f"{name}: AI={prob:.4f} | Kết quả: {result}")
