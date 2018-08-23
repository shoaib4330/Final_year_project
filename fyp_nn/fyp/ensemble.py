import cv2
from sklearn.cluster import KMeans
import numpy as np


def object_by_contours(image_in):
    image_input = image_in.copy()
    countours = cv2.findContours(image_input, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0))
    return None


def kmeans_color_segmented_images(image_in):
    image_in = cv2.cvtColor(image_in, cv2.COLOR_RGB2GRAY)
    _rows = image_in.shape[0]
    _cols = image_in.shape[1]
    image_in = image_in.reshape((_rows * _cols, 1))
    cpts = KMeans(n_clusters=2, random_state=0, n_init=15).fit(image_in)

    f_img = cpts.labels_
    f_img = (f_img + 1)
    f_img = f_img * 40
    f_img = f_img.reshape((_rows, _cols))

    unique_labels = np.unique(cpts.labels_)
    unique_labels = unique_labels + 1
    images_list = list()
    for a_label in unique_labels:
        curr_img = f_img.copy()
        for row in range(curr_img.shape[0]):
            for col in range(curr_img.shape[1]):
                if curr_img[row][col] == a_label * 40:
                    continue
                else:
                    curr_img[row][col] = 0
        curr_img = curr_img.astype(np.uint8)
        images_list.append(curr_img)
    return images_list


if __name__ == "__main__":
    print("---Main Runs---")

    img = cv2.imread("/Users/shoaibanwar/PycharmProjects/neural_test/resources/car_extracted.png")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_rows = img.shape[0]
    img_cols = img.shape[1]
    img = img.reshape((img_rows * img_cols, 1))
    kmeans = KMeans(n_clusters = 3, random_state=0, n_init=15).fit(img)
    print(kmeans.labels_)

    f_img = kmeans.labels_
    f_img = (f_img + 1)
    f_img = f_img * 40
    f_img = f_img.reshape((img_rows, img_cols))

    all_unique_labels = np.unique(kmeans.labels_)
    all_unique_labels = all_unique_labels + 1

    cc = 0
    for label in all_unique_labels:
        print("--runsss---")

        curr_img = f_img.copy()
        for row in range(curr_img.shape[0]):
            for col in range(curr_img.shape[1]):
                if curr_img[row][col] == label * 40:
                    continue
                else:
                    curr_img[row][col] = 0

        # object_by_contours(curr_img)
        curr_img = curr_img.astype(np.uint8)
        # curr_img = cv2.bilateralFilter(curr_img, 11, 17, 17)
        # edged_img = cv2.Canny(curr_img, 100, 200)
        cv2.imwrite("/Users/shoaibanwar/PycharmProjects/neural_test/resources/exrs/"+"exrs_"+str(cc)+".jpg", curr_img)
        cc+=1
        cv2.imshow("k_mean_wind", curr_img)
        cv2.waitKey(0)

    # kmeans.predict([[0, 0], [4, 4]])
    # kmeans.cluster_centers_