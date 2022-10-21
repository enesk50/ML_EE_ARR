import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
import keras.api._v2.keras as keras
from keras import layers, models

# Load ECG data and split into features and labels
def load_ECG_data(folder_name, file_name):
    file_path = os.path.join(".", folder_name, file_name)
    df = pd.read_csv(file_path, header=None)

    df_ECG = df.iloc[:, 0:-1].copy()
    ds_lab = df.iloc[:, -1].copy()

    return df_ECG, ds_lab


if __name__ == '__main__':
    # TIMING
    time_start = time.time()

    # Data loading
    df_ECG_train, ds_lab_train = load_ECG_data("ECG_data", "train_data.csv")
    df_ECG_test, ds_lab_test = load_ECG_data("ECG_data", "test_data.csv")
    df_ECG_val, ds_lab_val = load_ECG_data("ECG_data", "val_data.csv")

    # TIMING
    time_stop = time.time()
    time_loading = time_stop - time_start

    # Model setup
    n_time_steps = 250
    input_shape = (n_time_steps,1)

    model = models.Sequential()

    model.add(layers.Conv1D(filters=125, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=15, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    # Fit the model
    history = model.fit(df_ECG_train, ds_lab_train, epochs=10, 
                        validation_data=(df_ECG_val, ds_lab_val))

    # TIMING
    time_stop = time.time()
    time_tensor = time_stop - time_start

    # Check accuracy with test data
    print()
    print("Test results")
    test_loss, test_acc = model.evaluate(df_ECG_test, ds_lab_test, verbose=2)

    
    # TIMING
    print("_" * 80)
    print()
    print("Loading time: ", str(time_loading))
    print("Tensor flow time: ", str(time_tensor))

    # Confusion matrix
    labels_num = range(15)
    labels = ("N","/","L","R","e","j","A","a","J","S","E","F","V","f","Q")

    lab_predict = tf.math.argmax(model.predict(df_ECG_test), axis=1)

    cm = confusion_matrix(ds_lab_test, lab_predict, labels=labels_num)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap='Blues')

    # Epoch plot
    ep_fig, ep_ax = plt.subplots()
    ep_ax.plot(history.history['accuracy'], label='accuracy')
    ep_ax.plot(history.history['val_accuracy'], label = 'val_accuracy')
    ep_ax.set_xlabel('Epoch')
    ep_ax.set_ylabel('Accuracy')
    ep_ax.set_ylim([0.5, 1])
    ep_ax.legend(loc='lower right')

    plt.show()
