from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import csv
from fyp import utility
from keras import regularizers
from keras import optimizers
from fyp.cnn_approach import shuffling_script


# --------------------------- Code to create GrayScale Model --------------- #
def create_model_grayscale():
    df_posneg_combined_shuffled = pd.read_csv(
        "/Users/shoaibanwar/PycharmProjects/neural_test/csvs_pixelfeatures_RGB/trainingDataCombined.csv", header=None)
    print(df_posneg_combined_shuffled.shape)

    values_from_df = df_posneg_combined_shuffled.values

    X = values_from_df[:, 0:484]
    Y = values_from_df[:, 484]

    # create model
    model = Sequential()
    model.add(Dense(40, input_dim=484, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(15, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    # model.add(Dense(35, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))

    opt = optimizers.SGD(lr=0.01)
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X, Y, epochs=30, batch_size=100)

    # evaluate the model
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # serialize model to JSON
    model_json = model.to_json()
    with open("/Users/shoaibanwar/PycharmProjects/neural_test/fyp/cnn_approach/modelStorage/model.json",
              "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(
        "/Users/shoaibanwar/PycharmProjects/neural_test/fyp/cnn_approach/modelStorage/model.h5")

    print("Saved model to disk")


# --------------------------- Code to create Color Model --------------- #
def create_model_color():
    df_path_pos = "/Users/shoaibanwar/PycharmProjects/neural_test/csvs_pixelfeatures_RGB/fyp_color_model_files/rescaled_pos_train_22x22x3.csv"
    df_path_neg = "/Users/shoaibanwar/PycharmProjects/neural_test/csvs_pixelfeatures_RGB/fyp_color_model_files/rescaled_neg_training_22x22x3.csv"

    df_training_data = shuffling_script.join_and_shuffle_dataframes(df_path_pos, df_path_neg)

    print(df_training_data.shape)

    values_from_df = df_training_data.values

    X = values_from_df[:, 0:1452]
    Y = values_from_df[:, 1452]

    # prev reg val = 0.01
    # create model
    model = Sequential()
    model.add(Dense(50, input_dim=1452, activation='relu', kernel_regularizer=regularizers.l2(0.05)))
    model.add(Dense(30, activation='relu', kernel_regularizer=regularizers.l2(0.05)))
    model.add(Dense(30, activation='relu', kernel_regularizer=regularizers.l2(0.05)))
    model.add(Dense(30, activation='relu', kernel_regularizer=regularizers.l2(0.05)))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.051)))

    opt = optimizers.SGD(lr=0.01)
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X, Y, epochs=20, batch_size=100)

    # evaluate the model
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # serialize model to JSON
    model_json = model.to_json()
    with open("/Users/shoaibanwar/PycharmProjects/neural_test/fyp/cnn_approach/modelStorage/rescaled_2_model_color.json",
              "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(
        "/Users/shoaibanwar/PycharmProjects/neural_test/fyp/cnn_approach/modelStorage/rescaled_2_model_color.h5")
    print("Saved model to disk")
    return None


if __name__ == "__main__":
    print("Model Creation Script Runs")
    create_model_color()
