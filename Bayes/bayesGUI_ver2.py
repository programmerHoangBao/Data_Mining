import pandas as pd
import numpy as np
from collections import defaultdict
import tkinter as tk
from tkinter import ttk

# Đọc dữ liệu
try:
    df_buy_computer = pd.read_csv("D:\IT_HCMUTE\Hoc_ki_6\KHAI_PHA_DU_LIEU\Data_Mining\data\Buys_Computer.csv")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'Buys_Computer.csv'.")
    exit(1)

# Hàm tính xác suất tiên nghiệm và xác suất điều kiện
def calculate_probabilities(df, target_col='buys_computer', alpha=1):
    classes = df[target_col].unique()
    class_counts = df[target_col].value_counts()
    total_samples = len(df)
    
    prior_probs = {
        c: (class_counts[c] + alpha) / (total_samples + len(classes) * alpha)
        for c in classes
    }
    
    features = [col for col in df.columns if col != target_col]
    conditional_probs = defaultdict(lambda: defaultdict(dict))
    for feature in features:
        feature_values = df[feature].unique()
        for c in classes:
            class_df = df[df[target_col] == c]
            for val in feature_values:
                count = len(class_df[class_df[feature] == val])
                conditional_probs[feature][val][c] = (
                    (count + alpha) / (len(class_df) + len(feature_values) * alpha)
                )
    
    return prior_probs, conditional_probs, classes, features

# Hàm dự đoán
def predict(instance, prior_probs, conditional_probs, classes, features):
    max_prob = -1
    predicted_class = None
    
    for c in classes:
        prob = prior_probs[c]
        for feature in features:
            value = instance[feature]
            prob *= conditional_probs[feature][value][c]
        
        if prob > max_prob:
            max_prob = prob
            predicted_class = c
    
    return predicted_class

# Tạo giao diện GUI với tkinter
class NaiveBayesGUI:
    def __init__(self, root, df):
        self.root = root
        self.df = df
        self.root.title("Naive Bayes Prediction")
        self.root.geometry("500x600")  # Tăng kích thước cửa sổ để có không gian
        self.root.configure(bg="#e8ecef")  # Màu nền nhẹ
        
        # Tính xác suất
        self.prior_probs, self.conditional_probs, self.classes, self.features = calculate_probabilities(self.df)
        
        # Dictionary để lưu các giá trị đã chọn
        self.selected_values = {}
        self.dropdowns = {}
        
        # Tiêu đề
        tk.Label(
            self.root, 
            text="DỰ ĐOÁN MUA MÁY TÍNH", 
            font=("Helvetica", 18, "bold"), 
            bg="#e8ecef", 
            fg="#2c3e50"
        ).pack(pady=20)
        
        # Tạo frame để chứa các dropdown
        self.input_frame = tk.Frame(self.root, bg="#e8ecef")
        self.input_frame.pack(pady=10, padx=20, fill="x")
        
        # Tạo dropdown cho từng thuộc tính
        for feature in self.features:
            # Label cho thuộc tính
            tk.Label(
                self.input_frame, 
                text=f"{feature.capitalize()}:", 
                font=("Arial", 12, "bold"), 
                bg="#e8ecef", 
                fg="#34495e"
            ).pack(anchor="w", padx=10, pady=2)
            
            # Dropdown
            values = list(self.df[feature].unique())
            self.selected_values[feature] = tk.StringVar(self.root)
            self.selected_values[feature].set(values[0])  # Giá trị mặc định
            dropdown = ttk.Combobox(
                self.input_frame, 
                textvariable=self.selected_values[feature], 
                values=values, 
                state="readonly", 
                width=25, 
                font=("Arial", 11)
            )
            dropdown.pack(pady=5, padx=10, fill="x")
            self.dropdowns[feature] = dropdown
        
        # Frame chứa nút
        button_frame = tk.Frame(self.root, bg="#e8ecef")
        button_frame.pack(pady=15)
        
        # Nút dự đoán
        tk.Button(
            button_frame, 
            text="Dự đoán", 
            command=self.predict, 
            font=("Arial", 12, "bold"), 
            bg="#3498db", 
            fg="white", 
            activebackground="#2980b9", 
            relief="flat", 
            padx=20, 
            pady=5
        ).pack(side="left", padx=10)
        
        # Nút thoát
        tk.Button(
            button_frame, 
            text="Thoát", 
            command=self.root.quit, 
            font=("Arial", 12, "bold"), 
            bg="#e74c3c", 
            fg="white", 
            activebackground="#c0392b", 
            relief="flat", 
            padx=20, 
            pady=5
        ).pack(side="left", padx=10)
        
        # Frame để hiển thị kết quả
        self.result_frame = tk.Frame(self.root, bg="#ffffff", bd=2, relief="groove")
        self.result_frame.pack(pady=20, padx=20, fill="both")
        
        # Label tiêu đề kết quả
        tk.Label(
            self.result_frame, 
            text="KẾT QUẢ DỰ ĐOÁN", 
            font=("Arial", 12, "bold"), 
            bg="#ffffff", 
            fg="#2c3e50"
        ).pack(pady=5)
        
        # Khu vực hiển thị kết quả
        self.result_label = tk.Label(
            self.result_frame, 
            text="Chọn giá trị và nhấn 'Dự đoán' để xem kết quả", 
            font=("Arial", 11), 
            fg="#2ecc71", 
            bg="#ffffff", 
            wraplength=400,
            justify="left"
        )
        self.result_label.pack(pady=10, padx=10)
    
    def predict(self):
        # Lấy giá trị từ dropdown
        instance = {feature: self.selected_values[feature].get() for feature in self.features}
        
        # Dự đoán
        prediction = predict(instance, self.prior_probs, self.conditional_probs, self.classes, self.features)
        
        # Hiển thị kết quả
        result_text = f"Dữ liệu:\n{self.format_instance(instance)}\n\nKết quả: {prediction}"
        self.result_label.config(text=result_text)
    
    def format_instance(self, instance):
        # Định dạng dữ liệu dự đoán cho đẹp
        formatted = ""
        for feature, value in instance.items():
            formatted += f"{feature.capitalize()}: {value}\n"
        return formatted.strip()

# Hàm chính
def main():
    root = tk.Tk()
    app = NaiveBayesGUI(root, df_buy_computer)
    root.mainloop()

if __name__ == "__main__":
    main()