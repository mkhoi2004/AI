import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib  

# Bước 1: Đọc và xử lý dữ liệu
df_uploaded_check = pd.read_csv('final_cleaned_ev_charging_patterns_no_station.csv')

# Xử lý ngoại lai (Outliers) bằng phương pháp IQR
numerical_columns = df_uploaded_check.select_dtypes(include=['float64', 'int64']).columns
Q1 = df_uploaded_check[numerical_columns].quantile(0.25)
Q3 = df_uploaded_check[numerical_columns].quantile(0.75)
IQR = Q3 - Q1

# Xác định ngoại lai
outliers = ((df_uploaded_check[numerical_columns] < (Q1 - 1.5 * IQR)) | (df_uploaded_check[numerical_columns] > (Q3 + 1.5 * IQR)))

# Loại bỏ các ngoại lai
df_no_outliers = df_uploaded_check[~outliers.any(axis=1)]

# Bước 2: Chuẩn hóa dữ liệu (Scaling)
scaler = StandardScaler()

# Chọn các cột cần chuẩn hóa
numerical_columns_for_scaling = df_no_outliers.select_dtypes(include=['float64', 'int64']).columns
df_no_outliers[numerical_columns_for_scaling] = scaler.fit_transform(df_no_outliers[numerical_columns_for_scaling])

# Bước 3: Mã hóa các cột phân loại (One-Hot Encoding)
df_encoded = pd.get_dummies(df_no_outliers, drop_first=True)

# Bước 4: Xác định X (đặc trưng) và y (mục tiêu)
user_type_columns = [col for col in df_encoded.columns if 'User Type' in col]  # Cột 'User Type' đã được mã hóa thành các cột mới

# Bỏ các cột One-Hot của 'User Type' để tạo đặc trưng
X_final = df_encoded.drop(columns=user_type_columns)

# Mục tiêu là 'User Type' (bao gồm 3 loại: Commuter, Casual Driver, Long-Distance Traveler)
y_final = df_encoded[user_type_columns[0]]  # Chọn cột mục tiêu là 'User Type' (có 3 lớp)

# Bước 5: Chia dữ liệu thành tập huấn luyện và kiểm tra (80% huấn luyện, 20% kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

# Bước 6: Huấn luyện mô hình Logistic Regression (multi-class)
model = LogisticRegression(max_iter=1000, multi_class='ovr')
model.fit(X_train, y_train)

# Bước 7: Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Bước 8: Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy*100:.2f}%')
print(report)

# Bước 9: Lưu mô hình đã huấn luyện
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Lưu scaler để sử dụng cho dự đoán sau này

# Lưu tên các đặc trưng mà mô hình đã học được
model_features = X_train.columns.tolist()  # Tên các đặc trưng đã học
joblib.dump(model_features, 'model_features.pkl')  # Lưu tên các đặc trưng

print("Mô hình đã được lưu thành công.")
