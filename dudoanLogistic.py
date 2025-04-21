import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Tải mô hình đã huấn luyện
model_pipeline = joblib.load('logistic_regression_model.pkl')

# Dữ liệu đầu vào mới (dữ liệu từ 10 ID đầu tiên trong file)
input_data = [
    {
        'User ID': 'User_1',
        'Charging Station ID': 'Station_391',
        'Charging Start Time': '2024-01-01 00:00:00',
        'Charging End Time': '2024-01-01 00:39:00',
        'Vehicle Model': 'BMW i3',
        'Charging Station Location': 'Houston',
        'Time of Day': 'Evening',
        'Day of Week': 'Tuesday',
        'Charger Type': 'DC Fast Charger',
        'Battery Capacity (kWh)': 108.46,
        'Energy Consumed (kWh)': 60.71,
        'Charging Duration (hours)': 0.59,
        'Charging Rate (kW)': 36.39,
        'Charging Cost (USD)': 13.09,
        'State of Charge (Start %)': 29.37,
        'State of Charge (End %)': 86.12,
        'Distance Driven (since last charge) (km)': 293.60,
        'Temperature (°C)': 27.95,
        'Vehicle Age (years)': 2
    },
    {
        'User ID': 'User_2',
        'Charging Station ID': 'Station_428',
        'Charging Start Time': '2024-01-01 01:00:00',
        'Charging End Time': '2024-01-01 03:01:00',
        'Vehicle Model': 'Hyundai Kona',
        'Charging Station Location': 'San Francisco',
        'Time of Day': 'Morning',
        'Day of Week': 'Monday',
        'Charger Type': 'Level 1',
        'Battery Capacity (kWh)': 100.0,
        'Energy Consumed (kWh)': 12.34,
        'Charging Duration (hours)': 3.13,
        'Charging Rate (kW)': 30.68,
        'Charging Cost (USD)': 21.13,
        'State of Charge (Start %)': 10.12,
        'State of Charge (End %)': 84.66,
        'Distance Driven (since last charge) (km)': 112.11,
        'Temperature (°C)': 14.31,
        'Vehicle Age (years)': 3
    },
    {
        'User ID': 'User_3',
        'Charging Station ID': 'Station_181',
        'Charging Start Time': '2024-01-01 02:00:00',
        'Charging End Time': '2024-01-01 04:48:00',
        'Vehicle Model': 'Chevy Bolt',
        'Charging Station Location': 'San Francisco',
        'Time of Day': 'Morning',
        'Day of Week': 'Thursday',
        'Charger Type': 'Level 2',
        'Battery Capacity (kWh)': 75.0,
        'Energy Consumed (kWh)': 19.13,
        'Charging Duration (hours)': 2.45,
        'Charging Rate (kW)': 27.51,
        'Charging Cost (USD)': 35.67,
        'State of Charge (Start %)': 6.85,
        'State of Charge (End %)': 69.92,
        'Distance Driven (since last charge) (km)': 71.80,
        'Temperature (°C)': 21.00,
        'Vehicle Age (years)': 2
    },
    {
        'User ID': 'User_4',
        'Charging Station ID': 'Station_327',
        'Charging Start Time': '2024-01-01 03:00:00',
        'Charging End Time': '2024-01-01 06:42:00',
        'Vehicle Model': 'Hyundai Kona',
        'Charging Station Location': 'Houston',
        'Time of Day': 'Evening',
        'Day of Week': 'Saturday',
        'Charger Type': 'Level 1',
        'Battery Capacity (kWh)': 50.0,
        'Energy Consumed (kWh)': 79.46,
        'Charging Duration (hours)': 1.27,
        'Charging Rate (kW)': 32.88,
        'Charging Cost (USD)': 13.04,
        'State of Charge (Start %)': 83.12,
        'State of Charge (End %)': 99.62,
        'Distance Driven (since last charge) (km)': 199.58,
        'Temperature (°C)': 38.32,
        'Vehicle Age (years)': 1
    },
    {
        'User ID': 'User_5',
        'Charging Station ID': 'Station_108',
        'Charging Start Time': '2024-01-01 04:00:00',
        'Charging End Time': '2024-01-01 05:46:00',
        'Vehicle Model': 'Hyundai Kona',
        'Charging Station Location': 'Los Angeles',
        'Time of Day': 'Morning',
        'Day of Week': 'Saturday',
        'Charger Type': 'Level 1',
        'Battery Capacity (kWh)': 50.0,
        'Energy Consumed (kWh)': 19.63,
        'Charging Duration (hours)': 2.02,
        'Charging Rate (kW)': 10.22,
        'Charging Cost (USD)': 10.16,
        'State of Charge (Start %)': 54.26,
        'State of Charge (End %)': 63.74,
        'Distance Driven (since last charge) (km)': 203.66,
        'Temperature (°C)': -7.83,
        'Vehicle Age (years)': 1
    },
    {
        'User ID': 'User_6',
        'Charging Station ID': 'Station_367',
        'Charging Start Time': '2024-01-01 05:00:00',
        'Charging End Time': '2024-01-01 07:21:00',
        'Vehicle Model': 'Nissan Leaf',
        'Charging Station Location': 'Portland',
        'Time of Day': 'Morning',
        'Day of Week': 'Friday',
        'Charger Type': 'Level 2',
        'Battery Capacity (kWh)': 60.0,
        'Energy Consumed (kWh)': 35.10,
        'Charging Duration (hours)': 2.35,
        'Charging Rate (kW)': 14.91,
        'Charging Cost (USD)': 22.09,
        'State of Charge (Start %)': 42.71,
        'State of Charge (End %)': 90.12,
        'Distance Driven (since last charge) (km)': 130.57,
        'Temperature (°C)': 18.73,
        'Vehicle Age (years)': 2
    },
    {
        'User ID': 'User_7',
        'Charging Station ID': 'Station_522',
        'Charging Start Time': '2024-01-01 06:00:00',
        'Charging End Time': '2024-01-01 07:52:00',
        'Vehicle Model': 'Tesla Model 3',
        'Charging Station Location': 'New York',
        'Time of Day': 'Morning',
        'Day of Week': 'Tuesday',
        'Charger Type': 'DC Fast Charger',
        'Battery Capacity (kWh)': 75.0,
        'Energy Consumed (kWh)': 33.74,
        'Charging Duration (hours)': 1.87,
        'Charging Rate (kW)': 18.05,
        'Charging Cost (USD)': 26.48,
        'State of Charge (Start %)': 38.64,
        'State of Charge (End %)': 84.45,
        'Distance Driven (since last charge) (km)': 95.56,
        'Temperature (°C)': 20.11,
        'Vehicle Age (years)': 1
    },
    {
        'User ID': 'User_8',
        'Charging Station ID': 'Station_543',
        'Charging Start Time': '2024-01-01 07:00:00',
        'Charging End Time': '2024-01-01 08:23:00',
        'Vehicle Model': 'Chevy Bolt',
        'Charging Station Location': 'Los Angeles',
        'Time of Day': 'Morning',
        'Day of Week': 'Thursday',
        'Charger Type': 'Level 2',
        'Battery Capacity (kWh)': 66.0,
        'Energy Consumed (kWh)': 28.64,
        'Charging Duration (hours)': 1.38,
        'Charging Rate (kW)': 20.76,
        'Charging Cost (USD)': 19.92,
        'State of Charge (Start %)': 43.52,
        'State of Charge (End %)': 86.34,
        'Distance Driven (since last charge) (km)': 145.24,
        'Temperature (°C)': 22.09,
        'Vehicle Age (years)': 3
    }
]

# Chuyển đổi thành DataFrame
input_df_10 = pd.DataFrame(input_data)

# Dự đoán với mô hình đã huấn luyện
predictions = model_pipeline.predict(input_df_10)

# Hiển thị kết quả dự đoán
for i, prediction in enumerate(predictions):
    print(f"Predicted User Type for User_{i+1}: {prediction}")
