import glob
import cv2 as cv


def read_files_list(path, extension):
    m_path = path+"/*."+extension
    file_list = glob.glob(m_path)
    return file_list


def grayscale(in_image):
    img_gray = cv.cvtColor(in_image, cv.COLOR_BGR2GRAY)
    return img_gray


def resize(in_image, width, height):
    img_resized = cv.resize(in_image, (width, height))
    return img_resized


def reshape_image_to_1d (in_array):
    reshaped_array = in_array.reshape((in_array.shape[0] * in_array.shape[1], ))
    return reshaped_array