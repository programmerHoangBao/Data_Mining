import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

def parse_coordinates(coord_str):
    match = re.match(r'\(([\d.]+),([\d.]+)\)', coord_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    raise ValueError(f"Invalid coordinate format: {coord_str}")
    
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def plot_clusters(points, clusters, centroids, names):
    plt.figure(figsize=(10, 8))
    colors = ['r', 'g', 'b']
    
    # Plot points
    for i in range(len(points)):
        cluster_idx = clusters[i]
        plt.scatter(points[i, 0], points[i, 1], c=colors[cluster_idx], label=f'Cluster {cluster_idx + 1}' if i == 0 else "")
        plt.text(points[i, 0], points[i, 1], names[i], fontsize=8)
    
    # Plot centroids
    for i, centroid in enumerate(centroids):
        plt.scatter(centroid[0], centroid[1], c='black', marker='x', s=200, linewidths=3, label=f'Centroid {i + 1}')
    
    plt.title('K-means Clustering Result')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()  # Hiển thị biểu đồ

def kmeans(points, initial_centroids):
    index = 0
    clusters = np.zeros(len(points), dtype=int)  # Khởi tạo danh sách clusters
    initial_centroids = np.array(initial_centroids, dtype=float)
    
    while True:
        print('-------------------------------------------------------------')
        print(f'Lần lập thứ {index}: ')
        print('Tính khoảng cách của các điểm với centroids: ')
        
        # Gán các điểm vào cụm gần nhất
        for i in range(len(points)):
            distances = [euclidean_distance(points[i], centroid) for centroid in initial_centroids]
            print(f'{points[i]}, {distances}')
            clusters[i] = np.argmin(distances)  # Gán điểm vào cụm có khoảng cách nhỏ nhất
        
        print('Cập nhật lại tâm mới: ')
        new_centroids = np.zeros_like(initial_centroids)
        for i in range(len(initial_centroids)):
            cluster_points = points[clusters == i]  # Lấy các điểm thuộc cụm i
            if len(cluster_points) > 0:  # Kiểm tra xem cụm có điểm nào không
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[i] = initial_centroids[i]  # Giữ nguyên centroid nếu cụm rỗng
            print(f'Các điểm thuộc centroid thứ {i}: {cluster_points.tolist()}')
            print(f'Tọa độ điểm centroid mới thứ {i}: {new_centroids[i]}')
        
        # Kiểm tra điều kiện dừng
        if np.all(new_centroids == initial_centroids):
            break
        initial_centroids = new_centroids
        index += 1
    
    return clusters, new_centroids  # Trả về danh sách clusters và centroids

def main():
    df = pd.read_csv(r"D:\Data_Mining\project\data\k-mean.csv")
    names = []
    points = []
    for _, row in df.iterrows():
        x, y = parse_coordinates(row['Coordinates'])
        names.append(row['Name'])
        points.append([x, y])
    
    initial_centroids = [
        [2.0, 5.0],
        [0.0, 5.0]
    ]
    points = np.array(points)
    initial_centroids = np.array(initial_centroids)
    clusters, centroids = kmeans(points, initial_centroids)
    plot_clusters(points, clusters, centroids, names)
    
if __name__ == '__main__':
    main()