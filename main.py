import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

    # Epoch plot
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()
