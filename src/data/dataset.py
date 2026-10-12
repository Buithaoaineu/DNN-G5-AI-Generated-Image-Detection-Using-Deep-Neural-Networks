# Khai báo thư viện cần thiết
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Đọc dữ liệu gốc từ file train.csv
DATA_DIR = "data"
data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
print("Dữ liệu ban đầu:", data.shape)

# Chọn ngẫu nhiên 5,000 ảnh mỗi lớp để cân bằng dữ liệu
data = pd.concat([g.sample(n=5000, random_state=42) for _, g in data.groupby("label")], ignore_index=True)
print("Tổng ảnh được chọn:", len(data))

# Chia dữ liệu thành 70% train và 30% tạm, giữ tỷ lệ nhãn giữa các tập
train, temp = train_test_split(data, test_size=0.30, random_state=42, stratify=data["label"])

# Chia phần dữ liệu tạm thành 15% validation và 15% test
val, test = train_test_split(temp, test_size=0.50, random_state=42, stratify=temp["label"])

# Lưu ba file CSV vào thư mục data
train.to_csv(os.path.join(DATA_DIR, "train_split.csv"), index=False)
val.to_csv(os.path.join(DATA_DIR, "validation_split.csv"), index=False)
test.to_csv(os.path.join(DATA_DIR, "test_split.csv"), index=False)