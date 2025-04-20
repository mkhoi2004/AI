import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Bước 1: Tải mô hình đã huấn luyện và scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Bước 2: Hàm dự đoán
def predict_user_type(data):
    # Dữ liệu đầu vào (data) phải là một DataFrame với các đặc trưng giống như trong mô hình đã huấn luyện
    # Tiền xử lý dữ liệu: chuẩn hóa với scaler đã huấn luyện
    
    # Lấy danh sách các đặc trưng từ model đã huấn luyện
    model_features = joblib.load('model_features.pkl')  # Đây là các đặc trưng đã được huấn luyện

    # Đảm bảo rằng tất cả các đặc trưng có mặt trong data
    for feature in model_features:
        if feature not in data.columns:
            data[feature] = 0  # Nếu thiếu, thêm cột mới với giá trị bằng 0

    # Sắp xếp lại các cột của data để đúng với thứ tự khi huấn luyện
    data = data[model_features]

    # Chuẩn hóa dữ liệu với scaler đã huấn luyện
    numerical_columns = ['Battery Capacity (kWh)', 'Energy Consumed (kWh)', 'Charging Duration (hours)', 
                         'Charging Rate (kW)', 'Charging Cost (USD)', 'State of Charge (Start %)', 
                         'State of Charge (End %)', 'Distance Driven (since last charge) (km)', 
                         'Temperature (°C)', 'Vehicle Age (years)']  # Các cột cần chuẩn hóa

    data[numerical_columns] = scaler.transform(data[numerical_columns])

    # Bước 3: Dự đoán sử dụng mô hình
    prediction = model.predict(data)
    
    return prediction

# Ví dụ sử dụng hàm dự đoán
user_id_1_data = pd.DataFrame({
    'Vehicle Model_Chevy Bolt': [0],
    'Vehicle Model_Hyundai Kona': [0],
    'Vehicle Model_Nissan Leaf': [0],
    'Vehicle Model_Tesla Model 3': [0],
    'Charging Station Location_Houston': [1],
    'Charging Station Location_Los Angeles': [0],
    'Charging Station Location_New York': [0],
    'Charging Station Location_San Francisco': [0],
    'Time of Day_Evening': [1],
    'Time of Day_Morning': [0],
    'Time of Day_Night': [0],
    'Day of Week_Monday': [0],
    'Day of Week_Saturday': [0],
    'Day of Week_Sunday': [0],
    'Day of Week_Thursday': [0],
    'Day of Week_Tuesday': [1],
    'Day of Week_Wednesday': [0],
    'Charger Type_Level 1': [0],
    'Charger Type_Level 2': [1],
    'Battery Capacity (kWh)': [108.46],
    'Energy Consumed (kWh)': [60.71],
    'Charging Duration (hours)': [0.59],
    'Charging Rate (kW)': [36.39],
    'Charging Cost (USD)': [13.09],
    'State of Charge (Start %)': [29.37],
    'State of Charge (End %)': [86.12],
    'Distance Driven (since last charge) (km)': [293.60],
    'Temperature (°C)': [27.95],
    'Vehicle Age (years)': [2.0],
})

# Dự đoán kết quả
predicted_class = predict_user_type(user_id_1_data)
print(f'Dự đoán kết quả: {predicted_class}')
