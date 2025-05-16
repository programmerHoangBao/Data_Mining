#include<stdio.h>
#include<iostream>
#include<string>
#include<cmath>
#include<vector>
#define SIZE 20
#define FEATURES 5 //So luong dat trung
using namespace std;

void print(vector<string> v);
vector<string> getClass(int target_col, vector<vector<string>> data);
vector<float> calculate_prior_prob(int target_col, vector<string> classes, vector<vector<string>> data, int alpha = 1);
float calculate_conditional_prob(string val, string predict,int target_col, vector<vector<string>> data, int alpha = 1);
void Bayes(int target_col, vector<string> &new_instance, vector<vector<string>> data);

int main(){
    vector<vector<string>> data = {
        {"<=30", "high" , "no", "fair", "no"},
        {"<=30", "high" , "no", "excelent", "no"},
        {"31...40", "high", "no", "fair", "yes"},
        {">40", "medium", "no", "fair", "yes"},
        {">40", "low", "yes", "fair", "yes"},
        {">40", "low", "yes", "excellent", "no"},
        {"31...40", "low", "yes", "excellent", "yes"},
        {"<=30", "medium", "no", "fair", "no"},
        {"<=30", "low", "yes", "fair", "yes"},
        {">40", "medium", "yes", "fair", "yes"},
        {"<=30", "medium", "yes", "excellent", "yes"},
        {"31...40", "medium", "no", "excellent", "yes"},
        {"31...40", "high", "yes", "fair", "yes"},
        {">40", "medium", "no", "excellent", "no"}
    };

    vector<string> new_instance = {"<=30", "medium", "yes", "excellent"};

    Bayes(4, new_instance, data);

    print(new_instance);
    return 0;
}

void print(vector<string> v){
    for (string s : v){
        cout << s << " \t";
    }
    cout << endl;
}

vector<string> getClass(int target_col, vector<vector<string>> data){
    vector<string> classes;
    bool isExit;
    int lenData = data.size();
    for (int i = 0; i < lenData; i++){
        isExit = false;
        for (int j = 0; j < classes.size(); j++){
            if (data[i][target_col] == classes[j]){
                isExit = true;
                break;
            }
        }
        if (!isExit){
            classes.push_back(data[i][target_col]);
        }
    }
    return classes;
}

//Tinh xac xuat cua cac class trong lop muc tieu
vector<float> calculate_prior_prob(int target_col, vector<string> classes, vector<vector<string>> data, int alpha){
    vector<float> prior_prob;
    int lenData = data.size();
    int lenClass = classes.size();
    int count;
    for (int i = 0; i < lenClass; i++){
        count = 0;
        for (int j = 0; j < lenData; j++){
            if (classes[i] == data[j][target_col]){
                count += 1;
            }
        }
        float x = (float)(count + alpha)/(lenData + lenClass*alpha);
        prior_prob.push_back(x);
    }
    return prior_prob;
}

float calculate_conditional_prob(string val, string predict,int target_col, vector<vector<string>> data, int alpha){
    vector<string> classTarget = getClass(target_col, data);
    int lenData = data.size();
    int count_val = 0;
    int count_predict = 0;
    for (int i = 0; i < lenData; i++){
        if (data[i][target_col] == val && data[i][4] == predict){
            count_val += 1;
        }
        if (data[i][4] == predict){
            count_predict += 1;
        }
    } 
    return (float)(count_val+alpha)/(count_predict+classTarget.size()*alpha);
}
void Bayes(int target_col, vector<string> &new_instance, vector<vector<string>> data){
    //Tinh xac xuat cua cac class trong cot muc tieu target_col
    vector<string> classByComputer = getClass(4, data);
    vector<float> prior_priob_buycomputer = calculate_prior_prob(4, classByComputer, data);

    //Tinh xac xuat cua cac thuoc tinh trong new_instance khi biet ket qua muc tieu
    float max_prob = -1;
    string predicted_class = "";
    for (int i = 0; i < classByComputer.size(); i++){
        float prob =  prior_priob_buycomputer[i];
        for (int j = 0; j < new_instance.size(); j++){
            prob *= calculate_conditional_prob(new_instance[j], classByComputer[i], j, data);
        }
        if (prob > max_prob){
            max_prob = prob;
            predicted_class = classByComputer[i];
        }
    }

    new_instance.push_back(predicted_class);
}
