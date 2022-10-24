import os
import time
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, precision_score, \
    recall_score, plot_confusion_matrix
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Load ECG data and split into features and labels
def load_ECG_data(folder_name, file_name):
    file_path = os.path.join(".", folder_name, file_name)
    df = pd.read_csv(file_path, header=None)

    df_ECG = df.iloc[:, 0:-1].copy()
    ds_lab = df.iloc[:, -1].copy()

    return df_ECG, ds_lab

def apply_svm():
    # TIMING
    time_start = time.time()

    df_ECG_train, ds_lab_train = load_ECG_data("ECG_data", "train_data.csv")
    df_ECG_test, ds_lab_test = load_ECG_data("ECG_data", "test_data.csv")
    df_ECG_val, ds_lab_val = load_ECG_data("ECG_data", "val_data.csv")

    # TIMING
    time_stop = time.time()
    time_loading = time_stop - time_start
    print("data loading: ", time_loading)

    # Scaling
    scale = StandardScaler()
    scale.fit(df_ECG_train)
    df_ECG_train_scaled = scale.transform(df_ECG_train)
    df_ECG_test_scaled = scale.transform(df_ECG_test)
    df_ECG_val_scaled = scale.transform(df_ECG_val)

    svmlin_mc_ovr = svm.SVC(kernel='linear', C=1.0, coef0=0.0, tol=1e-3).fit(df_ECG_train_scaled, ds_lab_train)

    # TIMING
    time_stop = time.time()
    time_fit = time_stop - time_start
    print("Time fitted: ", time_fit)

    y_test_prediction_svmovr = svmlin_mc_ovr.predict(df_ECG_test_scaled)

    Accuracy_svmovr = accuracy_score(df_ECG_test_scaled, y_test_prediction_svmovr)
    print("Accuracy: " + str(Accuracy_svmovr))

    # Confusion matrix
    labels_num = range(15)
    labels = ("N", "/", "L", "R", "e", "j", "A", "a", "J", "S", "E", "F", "V", "f", "Q")
    plot_confusion_matrix(svmlin_mc_ovr, df_ECG_test_scaled, ds_lab_test,  labels=labels_num, display_labels=labels)