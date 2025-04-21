import pandas as pd

# Đọc dữ liệu
df = pd.read_csv('cleaned_data_fixed.csv')

# Kiểm tra phân bố của User Type
user_type_distribution = df['User Type'].value_counts()

# In kết quả phân bố
print("Phân bố User Type:")
print(user_type_distribution)

import pandas as pd
from catboost import CatBoostClassifier

# Tải mô hình đã lưu
model = CatBoostClassifier()
model.load_model('catboost_model.cbm')

# Lấy tên các đặc trưng (features) đã được sử dụng trong mô hình
# (Giả sử bạn đã có DataFrame X ban đầu, hoặc bạn có thể sử dụng tên cột của DataFrame)
feature_names = ['Vehicle Model', 'Charging Station Location', 'Time of Day', 'Day of Week', 'Charger Type', 
                 'Battery Capacity (kWh)', 'Energy Consumed (kWh)', 'Charging Duration (hours)', 'Charging Rate (kW)', 
                 'Charging Cost (USD)', 'State of Charge (Start %)', 'State of Charge (End %)', 'Distance Driven (since last charge) (km)', 
                 'Temperature (°C)', 'Vehicle Age (years)']

# Kiểm tra tên các đặc trưng đã được lưu trong mô hình
print("Features used in the model:")
for feature in feature_names:
    print(feature)
    
# Lấy tầm quan trọng của các đặc trưng
feature_importance = model.get_feature_importance()

# In tầm quan trọng của các đặc trưng
print("\nFeature Importance:")
for feature, importance in zip(feature_names, feature_importance):
    print(f'{feature}: {importance}')
