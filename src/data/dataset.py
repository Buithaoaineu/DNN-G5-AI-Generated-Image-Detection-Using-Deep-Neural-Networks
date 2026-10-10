#Khai báo các thư viện/hàm cần dùng
import pandas as pd
from sklearn.model_selection import train_test_split

#Đọc dữ liệu từ file csv
data = pd.read_csv('./data/train.csv')

#Tách dữ liệu thành tập train và test
dt_Train, dt_Test = train_test_split(
    data, test_size=0.15, random_state=42,
    stratify=data['label']
)

#Tách dt_Train thành tập train và validation
dt_Train, dt_Val = train_test_split(
    dt_Train, test_size=0.15/0.85, random_state=42,
    stratify=dt_Train['label']
)

#Lưu các tập dữ liệu
dt_Train.to_csv('./data/train_split.csv', index=False)
dt_Val.to_csv('./data/validation_split.csv', index=False)
dt_Test.to_csv('./data/test_split.csv', index=False)
