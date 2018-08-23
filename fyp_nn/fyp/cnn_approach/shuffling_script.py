import numpy as np
import pandas as pd
import csv


def join_and_shuffle_dataframes(df1_path, df2_path):
    df_a = pd.read_csv(df1_path, header=None)
    df_b = pd.read_csv(df2_path, header=None)
    # print(df_a.shape)
    # print(df_b.shape)
    combined_dfs = pd.concat([df_a, df_b])
    combined_dfs = combined_dfs.sample(frac=1)
    return combined_dfs


if __name__ == "__main__":
    print("Shuffling_Script Runs")

    csv_pos = "/Users/shoaibanwar/PycharmProjects/neural_test/csvs_pixelfeatures_RGB/post_test_22x22.csv"
    csv_neg = "/Users/shoaibanwar/PycharmProjects/neural_test/csvs_pixelfeatures_RGB/neg_test_22x22x3.csv"

    shuffled_df = join_and_shuffle_dataframes(csv_pos, csv_neg)
    shuffled_df.to_csv("/Users/shoaibanwar/PycharmProjects/neural_test/csvs_pixelfeatures_RGB/testDataCombined.csv", index=False, header=None)

    print("Shuffling_Script--- Done")
