import pandas as pd
import numpy as np
from collections import defaultdict
import sys

sys.stdout.reconfigure(encoding='utf-8')


def calculate_probabilities(df, target_col='buys_computer', alpha=1):
    # Lớp mục tiêu
    classes = df[target_col].unique()
    class_counts = df[target_col].value_counts()
    total_samples = len(df)
    
    # Lưu trữ xác suất
    prior_probs = {}
    conditional_probs = defaultdict(lambda: defaultdict(dict))
    
    # Tính xác suất tiên nghiệm P(buys_computer)
    for c in classes:
        prior_probs[c] = (class_counts[c] + alpha) / (total_samples + len(classes) * alpha)
    
    # Tính xác suất điều kiện P(feature|buys_computer)
    features = [col for col in df.columns if col != target_col]
    for feature in features:
        feature_values = df[feature].unique()
        for c in classes:
            class_df = df[df[target_col] == c]
            for val in feature_values:
                count = len(class_df[class_df[feature] == val])
                # Laplace Correction: (count + alpha) / (total + alpha * number of values)
                conditional_probs[feature][val][c] = (count + alpha) / (len(class_df) + len(feature_values) * alpha)
    
    return prior_probs, conditional_probs, classes

def predict(instance, prior_probs, conditional_probs, classes, features):
    max_prob = -1
    predicted_class = None
    
    for c in classes:
        prob = prior_probs[c]
        for feature, value in instance.items():
            prob *= conditional_probs[feature][value][c]
        
        if prob > max_prob:
            max_prob = prob
            predicted_class = c
    
    return predicted_class

def main():
    
    df_buy_computer = pd.read_csv(r"D:\Data_Mining\project\data\Buys_Computer.csv")
    
    prior_probs, conditional_probs, classes = calculate_probabilities(df_buy_computer)

    print(f"Xác suất mua máy tính (buy_computer): {prior_probs}")

    # Chuyển defaultdict thành dict
    conditional_probs_dict = dict(conditional_probs)

    # In từng thuộc tính
    for feature in conditional_probs_dict:
        print(f"\nXác suất điều kiện cho thuộc tính '{feature}':")
        for value in conditional_probs_dict[feature]:
            print(f"  Giá trị '{value}':")
            for cls in classes:
                print(f"    P({value} | {cls}) = {conditional_probs_dict[feature][value][cls]:.4f}")
            
        print(f"Tên classes: {classes}")
    
    new_instance = {
        'age': '<=30',
        'in come': 'medium',
        'student': 'yes',
        'credit_rating': 'excellent'
    }  

    print(new_instance)
    
    features = ['age', 'in come', 'student', 'credit_rating']
    prediction = predict(new_instance, prior_probs, conditional_probs, classes, features)
    
    print(f"Dữ liệu dự đoán: {new_instance}")
    print(f"Kết quả dự đoán buys_computer: {prediction}")
    
if __name__ == "__main__":
    main()