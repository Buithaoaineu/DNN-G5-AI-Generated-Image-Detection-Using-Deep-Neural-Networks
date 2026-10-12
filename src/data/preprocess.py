# Khai báo thư viện cần thiết
from PIL import Image
import numpy as np

# Đọc ảnh, chuyển định dạng ảnh sang RGB, resize ảnh về 160x160 và chuẩn hóa pixel về [0, 1]
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((160, 160))
    return np.array(image, dtype=np.float32) / 255.0
