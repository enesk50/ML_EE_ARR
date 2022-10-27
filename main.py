import os
import time
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, \
    f1_score, accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
import keras.api._v2.keras as keras
from keras import layers, models

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

# Print the scores
def generate_scores(expected_data, predicted_data, labels):
    Accuracy_model = accuracy_score(expected_data, predicted_data)
    F1_mode = f1_score(expected_data, predicted_data, average='weighted', zero_division=0, labels=labels)
    Precision_model = precision_score(expected_data, predicted_data, average='weighted', zero_division=0, labels=labels)
    Recall_model = recall_score(expected_data, predicted_data, average='weighted', zero_division=0, labels=labels)
    print("Accuracy: " + str(Accuracy_model))
    print("F1 score: " + str(F1_mode))
    print("Recall score: " + str(Recall_model))
    print("Precision score: " + str(Precision_model))

if __name__ == '__main__':
    # TIMING
    time_start = time.time()

    # Data loading
    df_ECG_train, ds_lab_train = load_ECG_data("ECG_data", "train_data.csv")
    df_ECG_test, ds_lab_test = load_ECG_data("ECG_data", "test_data.csv")
    df_ECG_val, ds_lab_val = load_ECG_data("ECG_data", "val_data.csv")

    # Label definitions
    labels_num = range(15)
    labels = ("N", "/", "L", "R", "e", "j", "A", "a", "J", "S", "E", "F", "V", "f", "Q")

    # TIMING
    time_stop = time.time()
    time_loading = time_stop - time_start

    # Model setup
    n_time_steps = 250
    input_shape = (n_time_steps, 1)

    model = models.Sequential()

    model.add(layers.Conv1D(filters=125, kernel_size=3, activation='relu', input_shape=input_shape))
    # model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Flatten())
    model.add(layers.Dense(units=15, activation='softmax'))

    model.summary()

    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    # Fit the model and weight classes
    # class_weights_arr = compute_class_weight('balanced', classes=labels_num, y=ds_lab_train)
    # class_weights_dict = dict(zip(labels_num, class_weights_arr))
    
    history = model.fit(df_ECG_train, ds_lab_train, epochs=10,
                        validation_data=(df_ECG_val, ds_lab_val),
                        # class_weight=class_weights_dict
                        )

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
    lab_predict = tf.math.argmax(model.predict(df_ECG_test), axis=1)

    cm = confusion_matrix(ds_lab_test, lab_predict, labels=labels_num)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap='Blues')

    # Epoch plot
    ep_fig, ep_ax = plt.subplots()
    ep_ax.plot(history.history['accuracy'], label='accuracy')
    ep_ax.plot(history.history['val_accuracy'], label='val_accuracy')
    ep_ax.set_xlabel('Epoch')
    ep_ax.set_ylabel('Accuracy')
    ep_ax.set_ylim([0.5, 1])
    ep_ax.legend(loc='lower right')

    plt.show()

    # Scores
    accuracy_per_class = []
    for lab in labels_num:
        true_negatives = np.sum(np.delete(np.delete(cm, lab, axis=0), lab, axis=1))
        true_positives = cm[lab, lab]
        accuracy_per_class.append((true_positives + true_negatives) / np.sum(cm))

    print("_" * 80)
    print()
    generate_scores(ds_lab_test, lab_predict, labels=labels_num)

    print("Accuracy per class:")
    print(pd.DataFrame(accuracy_per_class, index=labels, columns=["Accuracy"]))
