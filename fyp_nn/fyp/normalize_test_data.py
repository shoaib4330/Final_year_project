import numpy as np
import pandas as pd


def normalize_feature_set(features_list):
    data_frame_mean = pd.read_csv("/Users/shoaibanwar/PycharmProjects/neural_test/fyp_csvs/means.csv")
    data_frame_std = pd.read_csv("/Users/shoaibanwar/PycharmProjects/neural_test/fyp_csvs/stds.csv")
    means = data_frame_mean.values
    stds = data_frame_std.values
    means = np.squeeze(means)
    stds = np.squeeze(stds)
    normalized_feat_list = list()
    for index, feat  in enumerate(features_list):
        feat = (feat - means[index])/stds[index]
        normalized_feat_list.append(feat)
    return normalized_feat_list


if __name__ == "__main__":
    print("--Main Called--")

