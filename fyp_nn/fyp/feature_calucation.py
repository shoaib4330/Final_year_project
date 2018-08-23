import cv2
import numpy as np
import pandas as pd
from fyp import utility


# RGB Max 3
def get_r_g_b_max(input_img):
    red_max = np.max(input_img[:, :, 0])
    green_max = np.max(input_img[:, :, 1])
    blue_max = np.max(input_img[:, :, 2])
    return red_max, green_max, blue_max


# RGB Min 6
def get_r_g_b_min(input_img):
    red_min = np.min(input_img[:, :, 0])
    green_min = np.min(input_img[:, :, 1])
    blue_min = np.min(input_img[:, :, 2])
    return red_min, green_min, blue_min


# RGB Mean 9
def get_r_g_b_mean(input_img):
    red = np.mean(input_img[:, :, 0])
    green = np.mean(input_img[:, :, 1])
    blue = np.mean(input_img[:, :, 2])
    return red, green, blue


# RGB Max_Mean 12
def get_r_g_b_maxmean(input_img):

    r_mean = np.mean(input_img[:, :, 0])
    g_mean = np.mean(input_img[:, :, 1])
    b_mean = np.mean(input_img[:, :, 2])

    arr_r_above_mean = np.argwhere(input_img[:, :, 0] > r_mean)
    arr_g_above_mean = np.argwhere(input_img[:, :, 1] > g_mean)
    arr_b_above_mean = np.argwhere(input_img[:, :, 2] > b_mean)

    r_maxmean = np.mean(arr_r_above_mean)
    g_maxmean = np.mean(arr_g_above_mean)
    b_maxmean = np.mean(arr_b_above_mean)

    return r_maxmean, g_maxmean, b_maxmean


# RGB Min_Mean 15
def get_r_g_b_minmean(input_img):

    r_mean = np.mean(input_img[:, :, 0])
    g_mean = np.mean(input_img[:, :, 1])
    b_mean = np.mean(input_img[:, :, 2])

    arr_r_below_mean = np.argwhere(input_img[:, :, 0] < r_mean)
    arr_g_below_mean = np.argwhere(input_img[:, :, 1] < g_mean)
    arr_b_below_mean = np.argwhere(input_img[:, :, 2] < b_mean)

    r_minmean = np.mean(arr_r_below_mean)
    g_minmean = np.mean(arr_g_below_mean)
    b_minmean = np.mean(arr_b_below_mean)

    return r_minmean, g_minmean, b_minmean


# HSV Max 18
def get_hue_sat_intensity_max(input_image_hsv):
    hue_max = np.max(input_image_hsv[:, :, 0])
    sat_max = np.max(input_image_hsv[:, :, 1])
    intensity_max = np.max(input_image_hsv[:, :, 2])
    return hue_max, sat_max, intensity_max


# HSV Min 21
def get_hue_sat_intensity_min(input_image_hsv):
    hue_min = np.min(input_image_hsv[:, :, 0])
    sat_min = np.min(input_image_hsv[:, :, 1])
    intensity_min = np.min(input_image_hsv[:, :, 2])
    return hue_min, sat_min, intensity_min


# HSV Mean 24
def get_hue_sat_intensity_mean(input_img):
    # computing mean_hue
    hue = np.mean(input_img[:, :, 0])
    sat = np.mean(input_img[:, :, 1])
    val = np.mean(input_img[:, :, 2])
    return hue, sat, val


# HSV Max_Mean 27
def get_h_s_v_maxmean(input_img_hsv):

    hue_mean = np.mean(input_img_hsv[:, :, 0])
    sat_mean = np.mean(input_img_hsv[:, :, 1])
    val_mean = np.mean(input_img_hsv[:, :, 2])

    arr_hue_above_mean = np.argwhere(input_img_hsv[:, :, 0] > hue_mean)
    arr_sat_above_mean = np.argwhere(input_img_hsv[:, :, 1] > sat_mean)
    arr_val_above_mean = np.argwhere(input_img_hsv[:, :, 2] > val_mean)

    hue_maxmean = np.mean(arr_hue_above_mean)
    sat_maxmean = np.mean(arr_sat_above_mean)
    val_maxmean = np.mean(arr_val_above_mean)

    return hue_maxmean, sat_maxmean, val_maxmean


# HSV Min_Mean 30
def get_h_s_v_minmean(input_img_hsv):

    hue_mean = np.mean(input_img_hsv[:, :, 0])
    sat_mean = np.mean(input_img_hsv[:, :, 1])
    val_mean = np.mean(input_img_hsv[:, :, 2])

    arr_hue_below_mean = np.argwhere(input_img_hsv[:, :, 0] < hue_mean)
    arr_sat_below_mean = np.argwhere(input_img_hsv[:, :, 1] < sat_mean)
    arr_val_below_mean = np.argwhere(input_img_hsv[:, :, 2] < val_mean)

    hue_minmean = np.mean(arr_hue_below_mean)
    sat_minmean = np.mean(arr_sat_below_mean)
    val_minmean = np.mean(arr_val_below_mean)
    return hue_minmean, sat_minmean, val_minmean


def cal_feature_for_image(in_image):
    img_as_rgb = in_image
    img_as_hsv = cv2.cvtColor(img_as_rgb, cv2.COLOR_BGR2HSV)  # Converted, HSV Version
    max_R, max_G, max_B = get_r_g_b_max(img_as_rgb)  # 3
    min_R, min_G, min_B = get_r_g_b_min(img_as_rgb)  # 6
    mean_R, mean_G, mean_B = get_r_g_b_mean(img_as_rgb)  # 9
    max_mean_R, max_mean_G, max_mean_B = get_r_g_b_maxmean(img_as_rgb)  # 12
    min_mean_R, min_mean_G, min_mean_B = get_r_g_b_minmean(img_as_rgb)  # 15

    max_hue, max_sat, max_intensity = get_hue_sat_intensity_max(img_as_hsv)  # 3
    min_hue, min_sat, min_intensity = get_hue_sat_intensity_min(img_as_hsv)  # 6
    mean_hue, mean_sat, mean_intensity = get_hue_sat_intensity_mean(img_as_hsv)  # 9

    max_mean_hue, max_mean_sat, max_mean_intensity = get_h_s_v_maxmean(img_as_hsv)  # 12
    min_mean_hue, min_mean_sat, min_mean_intensity = get_h_s_v_minmean(img_as_hsv)  # 15

    mean_R = mean_R / max_R
    mean_G = mean_G / max_G
    mean_B = mean_B / max_B

    mean_hue = mean_hue / max_hue
    mean_sat = mean_sat / max_sat
    mean_intensity = mean_intensity / max_intensity

    max_mean_R = max_mean_R / max_R
    max_mean_G = max_mean_G / max_G
    max_mean_B = max_mean_B / max_B

    max_mean_hue = max_mean_hue / max_hue
    max_mean_sat = max_mean_sat / max_sat
    max_mean_intensity = max_mean_intensity / max_intensity

    min_mean_R = min_mean_R / max_R
    min_mean_G = min_mean_G / max_G
    min_mean_B = min_mean_B / max_B

    min_mean_hue = min_mean_hue / max_hue
    min_mean_sat = min_mean_sat / max_sat
    min_mean_intensity = min_mean_intensity / max_intensity


    img_featureSet_tuple = (mean_R, mean_G, mean_B, mean_hue, mean_sat, mean_intensity, max_mean_R, max_mean_G,
                                max_mean_B, max_mean_hue, max_mean_sat, max_mean_intensity, min_mean_R, min_mean_G,
                                min_mean_B, min_mean_hue, min_mean_sat, min_mean_intensity, float(max_R), float(max_G), float(max_B),
                                float(max_hue), float(max_sat), float(max_intensity), float(min_R), float(min_G), float(min_B), float(min_hue), float(min_sat),
                                float(min_intensity))
    return img_featureSet_tuple


def cal_feature_for_image_withlabelversion(in_image, label, img_name):
    img_as_rgb = in_image
    img_as_hsv = cv2.cvtColor(img_as_rgb, cv2.COLOR_RGB2HSV)  # Converted, HSV Version
    max_R, max_G, max_B = get_r_g_b_max(img_as_rgb)  # 3
    min_R, min_G, min_B = get_r_g_b_min(img_as_rgb)  # 6
    mean_R, mean_G, mean_B = get_r_g_b_mean(img_as_rgb)  # 9
    max_mean_R, max_mean_G, max_mean_B = get_r_g_b_maxmean(img_as_rgb)  # 12
    min_mean_R, min_mean_G, min_mean_B = get_r_g_b_minmean(img_as_rgb)  # 15

    max_hue, max_sat, max_intensity = get_hue_sat_intensity_max(img_as_hsv)  # 3
    min_hue, min_sat, min_intensity = get_hue_sat_intensity_min(img_as_hsv)  # 6
    mean_hue, mean_sat, mean_intensity = get_hue_sat_intensity_mean(img_as_hsv)  # 9

    max_mean_hue, max_mean_sat, max_mean_intensity = get_h_s_v_maxmean(img_as_hsv)  # 12
    min_mean_hue, min_mean_sat, min_mean_intensity = get_h_s_v_minmean(img_as_hsv)  # 15

    mean_R = mean_R / max_R
    mean_G = mean_G / max_G
    mean_B = mean_B / max_B

    mean_hue = mean_hue / max_hue
    mean_sat = mean_sat / max_sat
    mean_intensity = mean_intensity / max_intensity

    max_mean_R = max_mean_R / max_R
    max_mean_G = max_mean_G / max_G
    max_mean_B = max_mean_B / max_B

    max_mean_hue = max_mean_hue / max_hue
    max_mean_sat = max_mean_sat / max_sat
    max_mean_intensity = max_mean_intensity / max_intensity

    min_mean_R = min_mean_R / max_R
    min_mean_G = min_mean_G / max_G
    min_mean_B = min_mean_B / max_B

    min_mean_hue = min_mean_hue / max_hue
    min_mean_sat = min_mean_sat / max_sat
    min_mean_intensity = min_mean_intensity / max_intensity

    img_featureSet_tuple = (mean_R, mean_G, mean_B, mean_hue, mean_sat, mean_intensity, max_mean_R, max_mean_G,
                                max_mean_B, max_mean_hue, max_mean_sat, max_mean_intensity, min_mean_R, min_mean_G,
                                min_mean_B, min_mean_hue, min_mean_sat, min_mean_intensity, float(max_R), float(max_G), float(max_B),
                                float(max_hue), float(max_sat), float(max_intensity), float(min_R), float(min_G), float(min_B), float(min_hue), float(min_sat),
                                float(min_intensity), label)
    return img_featureSet_tuple


if __name__ == "__main__":
    print("-----Main Starts-----")

    files_list = utility.read_files_list("/Users/shoaibanwar/Downloads/step1_chars_segmented", "jpg")
    # files_list2 = utility.read_files_list("/Users/shoaibanwar/Desktop/char_new_data_set/Pakistani+Brazilian+euorpe wale character/characters/", "png")
    #
    # for pic in files_list2:
    #     files_list.append(pic)

    negative_label = 0
    positive_label = 1

    training_examples = list()
    for img_name in files_list:  # img_name is fully-qualified i.e. with path
        print(img_name)
        img = cv2.imread(img_name)  # read the image, RGB Version
        example_featureSet_tuple = cal_feature_for_image_withlabelversion(img, positive_label, img_name)

        training_examples.append(example_featureSet_tuple)

    df_training_data = pd.DataFrame(training_examples, columns=['mean_R', 'mean_G', 'mean_B', 'mean_hue', 'mean_sat',
                                                                'mean_intensity', 'max_mean_R', 'max_mean_G', 'max_mean_B',
                                                                'max_mean_hue', 'max_mean_sat', 'max_mean_intensity',
                                                                'min_mean_R', 'min_mean_G', 'min_mean_B', 'min_mean_hue',
                                                                'min_mean_sat', 'min_mean_intensity', 'max_R', 'max_G',
                                                                'max_B', 'max_hue', 'max_sat', 'max_intensity', 'min_R',
                                                                'min_G', 'min_B', 'min_hue', 'min_sat', 'min_intensity',
                                                                'label'])
    # shuffle the rows
    # -- df_training_data.sample(frac=1)
    # write the features to csv
    df_training_data.to_csv("/Users/shoaibanwar/PycharmProjects/neural_test/fyp_csvs/positive_images/aus_pos.csv", index=False)
    # print(training_examples)
    print("--- Done --")