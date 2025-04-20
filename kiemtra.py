import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Bước 1: Tải mô hình đã huấn luyện và scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')  # Đảm bảo rằng scaler đã được lưu khi huấn luyện

# Bước 2: Kiểm tra các đặc trưng mà mô hình đã học được
# Tải các tên đặc trưng (features) mà mô hình đã học được
model_features = model.feature_names_in_  # Đảm bảo rằng mô hình đã lưu thông tin này

# In ra các đặc trưng
print("Các đặc trưng mà mô hình đã học được:")
print(model_features)

# Bước 3: Kiểm tra các cột trong scaler
# Scaler đã được huấn luyện với dữ liệu có các cột số liệu nào
# Chúng ta có thể kiểm tra các cột cần chuẩn hóa
scaler_columns = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else None

# In ra các cột mà scaler đã chuẩn hóa
print("\nCác cột mà scaler đã được huấn luyện:")
print(scaler_columns)

