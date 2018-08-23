import cv2 as cv
import keras
import numpy as np
from fyp import utility
from module_car_extraction import cardetect
from east_detection.detect_text import detect_plate_region, get_det_doer, draw_illu
import Constants as cons

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    image = image.copy()
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, x + windowSize[0], y + windowSize[1] ,image[y:y + windowSize[1], x:x + windowSize[0]])


def load_model(model_name_abs_qualified, weights_path_abs_qualified):
    model_file = open(model_name_abs_qualified, 'r')
    model = model_file.read()
    model_file.close()
    loaded_model = keras.models.model_from_json(model)
    loaded_model.load_weights(weights_path_abs_qualified)
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model


def predict_region(in_region, classifier):
    in_region = utility.resize(in_region, 22, 22)
    # in_region = utility.grayscale(in_region)
    # in_region = utility.reshape_image_to_1d(in_region)
    in_region = in_region.reshape((in_region.shape[0] * in_region.shape[1] * in_region.shape[2],))
    in_region = np.divide(in_region, 255)
    abc = [in_region]
    in_region = np.asanyarray(abc)
    prediction_label = classifier.predict_classes(in_region)
    return prediction_label


# Check if a regions passed, is a plate? this function says yes if a region
# has >=4 char regions in it
def is_region_a_plate(in_region, in_classifier):
    char_regions = sliding_window(in_region, 4, [22, 30])
    char_count = 0

    for a_region in char_regions:
        a_label = predict_region(a_region[4], in_classifier)
        if a_label == 1:
            char_count += 1

    if char_count >= 5:
        print("A plate found")
        return True
    else:
        return False


def declare_plate(in_img, in_regions):
    return None


car_possibility_0_min_size = (14, 18)
car_possibility_0_max_size = (17, 22)


car_width_height_dims_possibility_0 = (450, 350)
car_width_height_dims_possibility_1 = (450, 350)


def process_car_with_ANPR(in_car, min_size, max_size, in_classification_model):
    if in_car is None:
        print("INFO: Input Image is None")
        return None
    regions_to_detect_as_chars = sliding_window(in_car, 4, [10, 13])
    for a_region in regions_to_detect_as_chars:
        label = predict_region(a_region[4], in_classification_model)
        # label = 1
        if label == 1:
            cv.rectangle(in_car, (a_region[0], a_region[1]), (a_region[2], a_region[3]), (255, 0, 0), 1)
    return in_car


def video_processor(in_stream_path, in_classification_model):
    # open video source
    cap = cv.VideoCapture(in_stream_path)
    while cap.isOpened():
        # read a frame
        ret, frame = cap.read()
        # extract car from frame
        cars_extracted = cardetect.extract_cars(frame)
        # Process each car
        for car in cars_extracted:
            car_dims = car.shape    # gives (rows/height, cols/width)
            if not (car_dims[0] >= car_width_height_dims_possibility_0[0] and car_dims[1] >= car_width_height_dims_possibility_0[1]):
                continue
            c_height = int (car_dims[0]/2)
            c_widht = int(car_dims[1] / 2)

            #car = cv.resize(car, (c_widht, c_height))
            #max_tuple = (int(car_possibility_0_max_size[0]/2), int(car_possibility_0_max_size[1]/2))
            #min_tuple = (int(car_possibility_0_min_size[0]/2), int(car_possibility_0_min_size[1]/2))
            max_tuple = car_possibility_0_max_size
            min_tuple = car_possibility_0_min_size
            marked_frame = process_car_with_ANPR(car, min_tuple, max_tuple, in_classification_model)
            cv.imshow("stream", marked_frame)
            cv.waitKey(10)
    cap.release()
    return None


def video_processor_robust(in_stream_path, pred):
    # open video source
    cap = cv.VideoCapture(in_stream_path)
    while cap.isOpened():
        # read a frame
        ret, frame = cap.read()
        # extract car from frame
        cars_extracted = cardetect.extract_cars(frame)
        # Process each car
        for car in cars_extracted:
            marked_frame = pred(car)
            rected_frame = draw_illu(car.copy(), marked_frame)
            cv.imshow("stream", rected_frame)
            cv.waitKey(1)
    cap.release()
    return None


def run_on_image(in_frame, index, in_pred):
    # cars_extracted = cardetect.extract_cars(in_frame)
    # for car in cars_extracted:
    marked_frame = in_pred(in_frame)
    rected_frame = draw_illu(in_frame.copy(), marked_frame)
    cv.imshow(str(index), rected_frame)
    cv.waitKey(0)


def demo_images(in_pred):
    images_list = [r"C:\Users\VenD\Desktop\ANPR\fyp_nn\test_images\car_extracted.png",
                   r"C:\Users\VenD\Desktop\ANPR\fyp_nn\test_images\corolla.jpg",
                   r"C:\Users\VenD\Desktop\ANPR\fyp_nn\test_images\cultus.jpg",
                   r"C:\Users\VenD\Desktop\ANPR\fyp_nn\test_images\red_custom_car.jpg"]
    i = 0
    for an_image in images_list:
        img = cv.imread(an_image)
        run_on_image(img, i, in_pred)
    return None


def demo_video(in_pred):
    vid_list = [r"C:\Users\VenD\Desktop\ANPR\fyp_nn\test_videos\a1.3gp"]
    i = 0
    for vid in vid_list:
        video_processor_robust(vid, in_pred)
    return None


if __name__ == "__main__":
    print("---Pixel Detector Script Runs---")

    pred = get_det_doer(predictor_path= cons.predictor_path)

    demo_images(pred)

    demo_video(pred)
    # video_processor_robust("/home/abdul/PycharmProjects/fyp_nn/test_videos/test_video_plate.3gp", pred)

    exit(0)
