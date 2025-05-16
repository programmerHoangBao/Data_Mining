import pandas as pd
import numpy as np
from collections import defaultdict
import sys
import tkinter as tk
from tkinter import ttk, messagebox

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

def create_gui(df, prior_probs, conditional_probs, classes):
    root = tk.Tk()
    root.title("Buys Computer Predictor")
    root.geometry("400x400")
    root.resizable(False, False)

    # Define valid options for each feature (based on typical dataset values)
    options = {
        'age': ['<=30', '31...40', '>40'],
        'income': ['high', 'medium', 'low'],
        'student': ['yes', 'no'],
        'credit_rating': ['excellent', 'fair']
    }

    # Store selected values
    selections = {
        'age': tk.StringVar(value=options['age'][0]),
        'income': tk.StringVar(value=options['income'][0]),
        'student': tk.StringVar(value=options['student'][0]),
        'credit_rating': tk.StringVar(value=options['credit_rating'][0])
    }

    # Create and place widgets
    tk.Label(root, text="Buys Computer Predictor", font=("Arial", 16, "bold")).pack(pady=10)

    # Frame for input fields
    input_frame = tk.Frame(root)
    input_frame.pack(pady=10, padx=10, fill='x')

    for i, (feature, var) in enumerate(selections.items()):
        tk.Label(input_frame, text=feature.capitalize(), font=("Arial", 10)).grid(row=i, column=0, sticky='e', pady=5)
        dropdown = ttk.Combobox(input_frame, textvariable=var, values=options[feature], state='readonly', width=20)
        dropdown.grid(row=i, column=1, sticky='w', pady=5)

    # Result label
    result_label = tk.Label(root, text="", font=("Arial", 10), wraplength=350)
    result_label.pack(pady=10)

    def on_predict():
        # Create instance from selections
        instance = {
            'age': selections['age'].get(),
            'income': selections['income'].get(),
            'student': selections['student'].get(),
            'credit_rating': selections['credit_rating'].get()
        }
        features = ['age', 'income', 'student', 'credit_rating']
        prediction = predict(instance, prior_probs, conditional_probs, classes, features)
        
        # Update result label
        result_text = f"Input: {instance}\nPrediction: Will buy computer = {prediction}"
        result_label.config(text=result_text)

    # Predict button
    tk.Button(root, text="Predict", command=on_predict, bg="#4CAF50", fg="white", font=("Arial", 10, "bold")).pack(pady=10)

    # Run the GUI
    root.mainloop()

def main():
    # Load dataset
    df_buy_computer = pd.read_csv(r"D:\Data_Mining\project\data\Buys_Computer.csv")
    
    # Calculate probabilities
    prior_probs, conditional_probs, classes = calculate_probabilities(df_buy_computer)

    # Print probabilities (optional, kept from your original code)
    print(f"Xác suất mua máy tính (buy_computer): {prior_probs}")
    conditional_probs_dict = dict(conditional_probs)
    for feature in conditional_probs_dict:
        print(f"\nXác suất điều kiện cho thuộc tính '{feature}':")
        for value in conditional_probs_dict[feature]:
            print(f"  Giá trị '{value}':")
            for cls in classes:
                print(f"    P({value} | {cls}) = {conditional_probs_dict[feature][value][cls]:.4f}")
        print(f"Tên classes: {classes}")

    # Launch GUI
    create_gui(df_buy_computer, prior_probs, conditional_probs, classes)

if __name__ == "__main__":
    main()