import math
import csv
import os
from graphviz import Digraph
import sys

sys.stdout.reconfigure(encoding='utf-8')


def read_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} không tồn tại.")
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            headers = reader.fieldnames
            print("Các cột trong CSV:", headers)
            for row in reader:
                cleaned_row = {key.strip(): value.strip() for key, value in row.items()}
                data.append(cleaned_row)
        print("Dữ liệu 5 dòng đầu:")
        for row in data[:5]:
            print(row)
        return data
    except Exception as e:
        raise Exception(f"Lỗi khi đọc file CSV: {str(e)}")
    

def calculate_entropy(data, target_col): 
    total = len(data)
    if total == 0:
        return 0
    target_values = [row[target_col] for row in data]
    value_counts = {}
    for val in target_values:
        value_counts[val] = value_counts.get(val, 0) + 1
    entropy = 0
    for count in value_counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability) if probability > 0 else 0
    return entropy


def calculate_information_gain(data, attribute, target_col):
    total_entropy = calculate_entropy(data, target_col)
    attribute_values = set(row[attribute] for row in data)
    weighted_entropy = 0
    total = len(data)
    
    for value in attribute_values:
        subset = [row for row in data if row[attribute] == value]
        subset_size = len(subset)
        if subset_size > 0:
            subset_entropy = calculate_entropy(subset, target_col)
            weighted_entropy += (subset_size / total) * subset_entropy
    
    return total_entropy - weighted_entropy


def find_best_attribute(data, attributes, target_col):
    max_gain = -float('inf')
    best_attr = None
    print(f"\nTính Information Gain cho các thuộc tính: {attributes}")
    for attr in attributes:
        if attr not in data[0]:
            raise KeyError(f"Thuộc tính {attr} không có trong dữ liệu.")
        gain = calculate_information_gain(data, attr, target_col)
        print(f"Information Gain của {attr}: {gain:.4f}")
        if gain > max_gain:
            max_gain = gain
            best_attr = attr
    print(f"Chọn thuộc tính tốt nhất: {best_attr} (Gain: {max_gain:.4f})")
    return best_attr


def build_decision_tree(data, attributes, target_col, depth=0):
    if not data:
        raise ValueError("Dữ liệu rỗng.")
    
    target_values = [row[target_col] for row in data]
    if len(set(target_values)) == 1:
        return target_values[0]
    if not attributes:
        return max(set(target_values), key=target_values.count)
    
    best_attr = find_best_attribute(data, attributes, target_col)
    if not best_attr:
        return max(set(target_values), key=target_values.count)
    
    tree = {best_attr: {}}
    attr_values = set(row[best_attr] for row in data)
    new_attributes = [attr for attr in attributes if attr != best_attr]
    
    for value in attr_values:
        subset = [row for row in data if row[best_attr] == value]
        tree[best_attr][value] = (
            max(set(target_values), key=target_values.count) if not subset 
            else build_decision_tree(subset, new_attributes, target_col, depth + 1)
        )
    
    return tree


def draw_tree(tree, dot=None, parent=None, edge_label=None, node_counter=None):
    if dot is None:
        dot = Digraph(comment='Decision Tree')
        # dot.attr(rankdir='LR')
        dot.attr('node', shape='box', style='filled', fillcolor='lightblue')
        node_counter = [0]  # Danh sách để lưu bộ đếm, cho phép thay đổi giá trị trong hàm đệ quy
    
    if isinstance(tree, dict):
        for attr, branches in tree.items():
            #Từ điển có cấu trúc {thuộc_tính: {giá_trị: nhánh}}
            node_counter[0] += 1
            node_id = f"node_{node_counter[0]}"  # Tạo node_id duy nhất
            dot.node(node_id, attr)
            if parent:
                dot.edge(parent, node_id, label=edge_label)
            for value, subtree in branches.items():
                draw_tree(subtree, dot, node_id, value, node_counter)
    else:
        node_counter[0] += 1
        node_id = f"node_{node_counter[0]}"
        dot.node(node_id, tree, fillcolor='lightgreen')
        if parent:
            dot.edge(parent, node_id, label=edge_label)
    
    return dot


file_path = r"D:\Data_Mining\project\data\Buys_Computer.csv"
try:
    data = read_csv(file_path)

    print("\nCác cột thuộc tính trong file đầu vào: ", list(data[0].keys()))

    print("\nDanh sách các cột có thể chọn làm cột mục tiêu:")
    columns = list(data[0].keys())
    for i, col in enumerate(columns, 1):
        print(f"{i}. {col}")

    while True:
        try:
            choice = int(input("\nNhập số thứ tự của cột mục tiêu: "))
            if 1 <= choice <= len(columns):
                target_col = columns[choice - 1]
                break
            else:
                print(f"Vui lòng chọn số từ 1 đến {len(columns)}.")
        except ValueError:
            print("Vui lòng nhập một số hợp lệ.")

    attributes = [col for col in data[0].keys() if col != target_col]

    print("\nThuộc tính bạn chọn làm cột mục tiêu là:", target_col)
    print(f"Thuộc tính còn lại là: {attributes}")
    
    print("\nBắt đầu xây dựng cây quyết định...")
    decision_tree = build_decision_tree(data, attributes, target_col)
    
    dot = draw_tree(decision_tree)
    output_path = "decision_tree"
    dot.render(output_path, format='png', cleanup=True)
        
except FileNotFoundError as e:
    print(f"Lỗi: {e}")
except KeyError as e:
    print(f"Lỗi: {e}")
except Exception as e:
    print(f"Lỗi không xác định: {str(e)}")