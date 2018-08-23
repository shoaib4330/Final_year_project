from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import csv
from fyp import utility
from keras.models import model_from_json
from fyp.cnn_approach import shuffling_script


def test_color_model():
    df_path_pos_test = "/Users/shoaibanwar/PycharmProjects/neural_test/csvs_pixelfeatures_RGB/fyp_color_model_files/rescaled_pos_test_22x22x3.csv"
    df_path_neg_test = "/Users/shoaibanwar/PycharmProjects/neural_test/csvs_pixelfeatures_RGB/fyp_color_model_files/rescaled_neg_test_22x22x3.csv"

    df_training_data = shuffling_script.join_and_shuffle_dataframes(df_path_pos_test, df_path_neg_test)
    print(df_training_data.shape)

    result = df_training_data.values
    X = result[:, 0:1452]
    Y = result[:, 1452]

    # load json and create model_additonal_layer
    json_file = open('/Users/shoaibanwar/PycharmProjects/neural_test/fyp/cnn_approach/modelStorage/rescaled_2_model_color.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model_additonal_layer
    loaded_model.load_weights("/Users/shoaibanwar/PycharmProjects/neural_test/fyp/cnn_approach/modelStorage/rescaled_2_model_color.h5")
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = loaded_model.evaluate(X, Y)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))


if __name__ == "__main__":
    print("Main Runs")

    test_color_model()
    # df_testData = pd.read_csv("/Users/shoaibanwar/PycharmProjects/neural_test/csvs_pixelfeatures_RGB/testDataCombined.csv")
    #
    # print(df_testData.shape)
    #
    # result = df_testData.values
    # X = result[:, 0:484]
    # Y = result[:, 484]
    #
    # # load json and create model_additonal_layer
    # json_file = open('/Users/shoaibanwar/PycharmProjects/neural_test/fyp/cnn_approach/modelStorage/model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model_additonal_layer
    # loaded_model.load_weights("/Users/shoaibanwar/PycharmProjects/neural_test/fyp/cnn_approach/modelStorage/model.h5")
    # loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # score = loaded_model.evaluate(X, Y)
    # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
