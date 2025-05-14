import numpy as np
import pandas as pd
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')


def parse_coordinates(coord_str):
    match = re.match(r'\(([\d.]+),([\d.]+)\)', coord_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    raise ValueError(f"Invalid coordinate format: {coord_str}")
    
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1-point2)**2))

def kmeans(points, initial_centroids):
    index = 0
    while True:
        clusters = []
        print('-------------------------------------------------------------')
        print(f'Lần lập thứ {index}: ')
        print('Tính khoản cách của các điểm với centroids: ')
        for i in range(len(points)):
            distances = []
            for j in range(len(initial_centroids)):
                distance = euclidean_distance(points[i], initial_centroids[j])
                distances.append(distance)
            print(f'{points[i]}, {distances}')
        
            cluster_idx = 0
            distance_min = distances[0]
            for t in range(1, len(distances)):
                if distance_min > distances[t]:
                    distance_min = distances[t]
                    cluster_idx = t
                    
            clusters.append(cluster_idx)
            
        print('Cập nhật lại tâm mới: ')
        new_centroids = np.zeros_like(initial_centroids)
        for i in range(len(new_centroids)):
            cluster_points = []
            for j in range(len(clusters)):
                if (clusters[j] == i):
                    cluster_points.append(points[j])
            new_centroids[i] = np.mean(cluster_points, axis=0)
            cluster_points_array = np.vstack(cluster_points)
            print(f'Các điểm thuộc centroid thứ {i}: {cluster_points_array.tolist()}')
            print(f'Toạ độ điểm centroid mới thứ {i}: {new_centroids[i]}')
        if (np.all(new_centroids == initial_centroids)):
            break
        else:
            initial_centroids = new_centroids
        index += 1
        
    return cluster_points, new_centroids
            


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
    kmeans(points, initial_centroids)
    
if __name__ == '__main__':
    main()