import pandas as pd
import tensorflow as tf
from src.data.preprocess import preprocess_image

# Tải model đã huấn luyện
model = tf.keras.models.load_model("src/models/complex_cnn.keras")

# Đọc dữ liệu test
df = pd.read_csv("data/test_split.csv")

# Tiền xử lý ảnh và dự đoán
image = preprocess_image("data/" + df.iloc[0]["file_name"])[None, ...]
prob = model.predict(image, verbose=0)[0, 0]

print("Xác suất ảnh AI gen:", round(float(prob), 4))
print("Kết quả:", "AI-gen" if prob >= 0.5 else "Real")