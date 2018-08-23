import numpy as np
import cv2 as cv
from fyp.utility import *
from fyp.feature_calucation import cal_feature_for_image


def generate_slices(image, width, height):
    # slide a window across the image
    for y in range(0, image.shape[0], height):
        for x in range(0, image.shape[1], width):
            # yield the current window
            yield (x, y, image[y:y + height, x:x + width])


if __name__ == "__main__":
    print("--Slicer Main Runs--")

    sliced_region_width = 21
    sliced_region_height = 33

    list_images = read_files_list("/Users/shoaibanwar/Desktop/neg/images", "jpg")
    for index, image in enumerate(list_images):
        img_to_slice = cv.imread(image)
        slices = generate_slices(img_to_slice, sliced_region_width, sliced_region_height)
        name = 'slice_sub_' + str(index)
        for inn_index, region in enumerate(slices):
            img_rgb = region[2]
            img_r = img_rgb[:, :, 0]
            subregion_width = len(img_r[0, :])
            subregion_height = len(img_r[:, 0])
            if subregion_height < sliced_region_height or subregion_width < sliced_region_width:
                continue
            else:
                print("slice-->", name+"_"+str(inn_index))
                cv.imwrite("/Users/shoaibanwar/Desktop/neg/output_slices/"+name+"_"+str(inn_index)+".jpg", region[2])
                # features_list = cal_feature_for_image(region[2])
