#Khai báo các thư viện/hàm cần dùng
from PIL import Image
import numpy as np

#Đọc và tiền xử lý ảnh
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    return image
