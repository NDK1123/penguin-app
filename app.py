# app.py

from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Tải mô hình và các cột đã được huấn luyện
model = joblib.load('penguin_model.pkl')
model_columns = joblib.load('penguin_model_columns.pkl')

# Tạo một dictionary để ánh xạ các mã hóa trở lại tên loài
species_mapping = {
    0: 'Adelie',
    1: 'Chinstrap',
    2: 'Gentoo'
}

# Biến để lưu trữ lịch sử dự đoán
# Đây là một giải pháp đơn giản, sẽ bị reset khi server khởi động lại.
# Trong thực tế, bạn nên sử dụng database hoặc session.
prediction_history = []

@app.route('/')
def home():
    # Truyền lịch sử dự đoán khi tải trang
    return render_template('index.html', prediction_history=prediction_history)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Lấy dữ liệu từ form
            bill_length = float(request.form['bill_length'])
            bill_depth = float(request.form['bill_depth'])
            flipper_length = float(request.form['flipper_length'])
            body_mass = float(request.form['body_mass'])
            island = request.form['island']
            sex = request.form['sex']
            year = request.form['year']

            # Tạo DataFrame từ input
            # Cần tạo đúng cấu trúc cột như lúc huấn luyện (One-Hot Encoding)
            query_data = {
                'bill_length_mm': [bill_length],
                'bill_depth_mm': [bill_depth],
                'flipper_length_mm': [flipper_length],
                'body_mass_g': [body_mass],
                'island_Dream': [1 if island == 'Dream' else 0],
                'island_Torgersen': [1 if island == 'Torgersen' else 0],
                'sex_male': [1 if sex == 'male' else 0],
                'year_2008': [1 if year == '2008' else 0],
                'year_2009': [1 if year == '2009' else 0]
            }
            
            query_df = pd.DataFrame(query_data)
            
            # Sắp xếp lại các cột để khớp với mô hình đã huấn luyện
            query_df = query_df.reindex(columns=model_columns, fill_value=0)
            
            # Thực hiện dự đoán
            prediction_encoded = model.predict(query_df)[0]
            
            # Chuyển đổi mã hóa thành tên loài
            prediction_species = species_mapping.get(prediction_encoded, 'Không xác định')
            
            # Tạo chuỗi kết quả và thêm vào lịch sử
            result_string = f"Loài chim cánh cụt được dự đoán là: {prediction_species}"
            prediction_history.insert(0, result_string)
            
            # Trả về template với kết quả và lịch sử
            return render_template('index.html', prediction_text=result_string, prediction_history=prediction_history)
        
        except ValueError:
            # Trả về template với lỗi và lịch sử
            return render_template('index.html', error='Dữ liệu nhập vào không hợp lệ. Vui lòng nhập số.', prediction_history=prediction_history)
        except Exception as e:
            # Trả về template với lỗi và lịch sử
            return render_template('index.html', error=f'Đã xảy ra lỗi: {e}', prediction_history=prediction_history)

if __name__ == '__main__':
    app.run(port=5001)