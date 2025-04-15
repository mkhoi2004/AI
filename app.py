from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load mô hình
logistic_model = joblib.load("best_model_logistic_regression.pkl")
catboost_model = joblib.load("best_model_catboost.pkl")

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu từ form
        battery_capacity = float(request.form['battery_capacity'])
        charging_duration = float(request.form['charging_duration'])
        charging_rate = float(request.form['charging_rate'])
        energy_consumed = float(request.form['energy_consumed'])
        model_choice = request.form['model_choice']
        
        # Chuẩn bị dữ liệu đầu vào
        data = np.array([[battery_capacity, charging_duration, charging_rate, energy_consumed]])
        
        # Dự đoán
        if model_choice == 'logistic':
            prediction = logistic_model.predict(data)[0]
        elif model_choice == 'catboost':
            prediction = catboost_model.predict(data)[0]
        else:
            prediction = "Không xác định"
        
        return render_template('index.html', prediction=prediction)
    except Exception as e:
        return render_template('index.html', prediction=f"Lỗi: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
