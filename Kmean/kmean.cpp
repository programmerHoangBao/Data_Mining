#include <iostream>
#include<stdio.h>
#include<cmath>
#define SIZE 20

using namespace std;

struct Point {
    float x;
    float y;
};

float euclidean_distance(Point a, Point b);
Point avg_point(Point points[], int lenPoint);
bool isEqualArrayPoint(Point arr_point_1[], Point arr_point_2[], int len);
void kmean(Point points[], int lenPoints, Point initial_centroids[], int lenCentroids);

int main() {
    Point points[] = {
        {2, 5}, {0, 5}, {2, 2}, {2, 4}, {5, 3},
        {1, 5}, {3, 11}, {3, 5}, {2, 12}, {2, 10}
    };
    Point initial_centroids[] = {
        {2, 5}, {0, 5}
    };
    
    int lenPoints = sizeof(points) / sizeof(points[0]);
    int lenCentroids = sizeof(initial_centroids) / sizeof(initial_centroids[0]);
    
    kmean(points, lenPoints, initial_centroids, lenCentroids);
    
    return 0;
}

float euclidean_distance(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

Point avg_point(Point points[], int lenPoint) {
    if (lenPoint == 0){
    	return {0, 0};	
	}
    float sum_x = 0, sum_y = 0;
    for (int i = 0; i < lenPoint; i++) {
        sum_x += points[i].x;
        sum_y += points[i].y;
    }
    return {sum_x / lenPoint, sum_y / lenPoint};
}

bool isEqualArrayPoint(Point arr_point_1[], Point arr_point_2[], int len) {
    for (int i = 0; i < len; i++) {
        if (arr_point_1[i].x != arr_point_2[i].x || arr_point_1[i].y != arr_point_2[i].y) {
            return false;
        }
    }
    return true;
}

void kmean(Point points[], int lenPoints, Point initial_centroids[], int lenCentroids) {
    int clusters[SIZE] = {}; //Luu chi so cum cua moi diem
    Point new_centroids[SIZE] = {}; // Luu centroids moi
    Point cluster_points[SIZE] = {}; // Luu cac diem trong moi cum tam thoi
    int lenClusterPoint = 0;
    int index = 0;

    while (true) {
        cout << "-----------------------------------------------" << endl;
        cout << "Lan lap " << index << ":" << endl;
        cout << "Tinh khoan cach diem voi centroid:" << endl;

        // Gan cac diem vao cum
        for (int i = 0; i < lenPoints; i++) {
            float min_distance = euclidean_distance(points[i], initial_centroids[0]);
            int cluster_idx = 0;
            printf("%d : (%.2f, %.2f)\t", i, points[i].x, points[i].y);
            for (int j = 0; j < lenCentroids; j++) {
                float distance = euclidean_distance(points[i], initial_centroids[j]);
                printf("%.2f \t", distance);
                if (distance < min_distance) {
                    min_distance = distance;
                    cluster_idx = j;
                }
            }
            printf("-> Cum %d \n", cluster_idx);
            clusters[i] = cluster_idx;
        }

        for (int i = 0; i < lenCentroids; i++){
        	// Gom cac diem theo cum
        	printf("Cum %d gom cac diem:\n", i);
        	lenClusterPoint = 0;
        	for (int j = 0; j < lenPoints; j++){
        		if (clusters[j] == i){
        			cluster_points[lenClusterPoint] = points[j];
        			printf("(%.2f, %.2f), ", cluster_points[lenClusterPoint].x, cluster_points[lenClusterPoint].y);
        			lenClusterPoint += 1;
				}
			}
			cout << endl;
			new_centroids[i] = avg_point(cluster_points, lenClusterPoint);
			printf("Centroid moi cua cum %d: (%.2f, %.2f)\n", i, new_centroids[i].x, new_centroids[i].y);
		}

        // Kiem tra co hoi tu hay chua
        if (isEqualArrayPoint(initial_centroids, new_centroids, lenCentroids)) {
            break;
        }

        // Cap nhat Centroid
        for (int i = 0; i < lenCentroids; i++) {
            initial_centroids[i] = new_centroids[i];
        }

        index++;
    }
}
