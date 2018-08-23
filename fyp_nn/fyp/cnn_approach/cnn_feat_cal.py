from fyp import utility
import cv2 as cv
import numpy as np
import pandas as pd


# Reads images and calculates features of images (resizes them to 22x22) as grayscale pixels.
def cal_features_for_grayscale_model(list_of_images):
    print("Function: cal_features_for_grayscale_model(), Starts")

    processed_images_list = list()
    for image in list_of_images:
        print(image)
        img_read = cv.imread(image)
        img_read = utility.resize(img_read, 22, 22)
        img_read = utility.grayscale(img_read)
        processed_images_list.append(img_read)

    images_as_1d_nparrs_list = list()
    for image in processed_images_list:
        image = utility.reshape_image_to_1d(image)
        images_as_1d_nparrs_list.append(image)

    data_frame = pd.DataFrame(images_as_1d_nparrs_list)
    print("Function: cal_features_for_grayscale_model(), Done")
    return data_frame


# Reads images and calculates features of images (resizes them to 22x22) as color pixels.
def cal_features_for_color_model(list_of_images, rescale_for_activation_func="relu"):
    print("Function: cal_features_for_color_model(), Starts")

    processed_images_list = list()
    for image in list_of_images:
        print(image)
        img_read = cv.imread(image)
        img_read = utility.resize(img_read, 22, 22)
        processed_images_list.append(img_read)

    images_as_1d_nparrs_list = list()
    for image in processed_images_list:
        # Make the image Linear (one dimensional), that is, an array with shape (22x22x3,)
        image = image.reshape((image.shape[0] * image.shape[1] * image.shape[2],))

        # --- Rescale image pixels to activations function bounds ---
        if rescale_for_activation_func == "sigmoid" or rescale_for_activation_func == "relu":
            image = np.divide(image, 255)

        images_as_1d_nparrs_list.append(image)

    data_frame = pd.DataFrame(images_as_1d_nparrs_list)
    print("Function: cal_features_for_color_model(), Done")
    return data_frame


if __name__ == "__main__":
    print("Main Runs")

    images_list = utility.read_files_list("/Volumes/D/Study/fyp extras/FYP_DATASET/neg_data/neg_test/", "jpg")
    images_list1 = utility.read_files_list("/Volumes/D/Study/fyp extras/FYP_DATASET/neg_data/neg_test/", "png")

    for item in images_list1:
        images_list.append(item)

    data_frame_with_features = cal_features_for_color_model(images_list, "relu")

    data_frame_with_features.to_csv("/Users/shoaibanwar/PycharmProjects/neural_test/csvs_pixelfeatures_RGB/fyp_color_model_files/rescaled_neg_test_22x22x3.csv", index=False,
                      header=False)
    print("Done")