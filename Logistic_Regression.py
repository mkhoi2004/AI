import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv('ev_charging_patterns.csv')

# Kiểm tra các cột và thông tin của dữ liệu
print(df.columns)

# Cột phân loại và cột số
categorical_columns = ['User ID','Charging Station ID','Charging Start Time','Charging End Time','Vehicle Model', 'Charging Station Location', 'Time of Day', 
                       'Day of Week', 'Charger Type']
numeric_columns = ['Battery Capacity (kWh)', 'Energy Consumed (kWh)', 'Charging Duration (hours)', 
                   'Charging Rate (kW)', 'Charging Cost (USD)', 'State of Charge (Start %)', 
                   'State of Charge (End %)', 'Distance Driven (since last charge) (km)', 
                   'Temperature (°C)', 'Vehicle Age (years)']

# Phân loại người dùng vào các nhóm
def classify_user(row):
    # Commuter: Di chuyển quãng đường ngắn, sử dụng ít năng lượng, sạc nhanh
    if row['Distance Driven (since last charge) (km)'] < 120 and row['Energy Consumed (kWh)'] < 60:
        return 'Commuter'
    # Casual-Driver: Di chuyển quãng đường trung bình, năng lượng tiêu thụ vừa phải
    elif row['Distance Driven (since last charge) (km)'] < 180 and row['Energy Consumed (kWh)'] < 80:
        return 'Casual-Driver'
    # Long-Distance Traveler: Di chuyển quãng đường dài, tiêu thụ nhiều năng lượng
    else:
        return 'Long-Distance Traveler'

# Áp dụng phân loại
df['User Type'] = df.apply(classify_user, axis=1)

# Chia dữ liệu thành X (các đặc trưng) và y (biến mục tiêu)
X = df[categorical_columns + numeric_columns]
y = df['User Type']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo các bước tiền xử lý cho cột phân loại và số
preprocessor = ColumnTransformer(
    transformers=[ 
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Điền giá trị thiếu bằng giá trị trung bình
        ]), numeric_columns),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),  # Điền NaN bằng 'Unknown'
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Chuyển đổi cột phân loại thành one-hot encoding
        ]), categorical_columns)
    ]
)

# Xây dựng Pipeline cho tiền xử lý và mô hình hóa
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs'))  # Sử dụng 'multinomial' cho phân loại đa lớp
])

# Huấn luyện mô hình
model_pipeline.fit(X_train, y_train)

# Đánh giá mô hình
accuracy = model_pipeline.score(X_test, y_test)
print(f'Accuracy: {accuracy}')

# Dự đoán trên tập kiểm tra
y_pred = model_pipeline.predict(X_test)

# Đánh giá mô hình bằng các chỉ số như accuracy, precision, recall, f1-score
print(classification_report(y_test, y_pred))

# Thêm ma trận nhầm lẫn (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Vẽ ma trận nhầm lẫn với các chú thích
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model_pipeline.classes_, yticklabels=model_pipeline.classes_)

# Thêm tiêu đề và nhãn cho các trục
plt.title('Confusion Matrix with Annotations')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Thêm các chú thích trực quan vào ma trận
for i in range(len(cm)):
    for j in range(len(cm[i])):
        plt.text(j, i, f'{cm[i][j]}', ha='center', va='center', color='black')

# Hiển thị ma trận nhầm lẫn
plt.show()

# Bước 9: Lưu mô hình đã huấn luyện
joblib.dump(model_pipeline, 'logistic_regression_model.pkl')

# Lưu tên các đặc trưng mà mô hình đã học được
model_features = X_train.columns.tolist()  # Tên các đặc trưng đã học
joblib.dump(model_features, 'model_features.pkl')  # Lưu tên các đặc trưng

print("Mô hình đã được lưu thành công.")
