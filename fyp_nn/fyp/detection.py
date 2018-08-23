import cv2 as cv
import keras
from fyp.feature_calucation import cal_feature_for_image
import numpy as np
from fyp.normalize_test_data import normalize_feature_set



# windowSize is = [height, width]
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
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


def predict_region(region, classifier):
    tuple_feature = cal_feature_for_image(region)
    features_aslist = list(tuple_feature)
    # First normalize the extracted features
    features_aslist = normalize_feature_set(features_aslist)
    feature_asarray = np.asarray([features_aslist])
    prediction_label = classification_model.predict_classes(feature_asarray)
    return prediction_label


if __name__ == "__main__":
    print("---detection_script Runs---")

    plate_width_to_height_ratio = 2.1
    standard_car_image_height = 370
    standard_car_image_width = 390
    standard_plate_height = 47

    standard_plate_char_height = 16
    standard_plate_char_width = 12

    # img = cv.imread("/Users/shoaibanwar/PycharmProjects/neural_test/resources/car_extracted.png")
    # img = cv.resize(img, (200, 200))
    # cv.imshow("view", img)
    # cv.waitKey(0)
    # # img = cv.resize(img, (500, 500))
    # # cv.imshow("view", img)
    # # cv.waitKey(0)
    # potential_plates = sliding_window(img, 50, [126, 60])
    #
    # for plate in potential_plates:
    #     images = kmeans_color_segmented_images(plate[2])
    #     for i_m_g in images:
    #         cv.imshow("image-AA", i_m_g)
    #         cv.waitKey(0)
    # cv.waitKey(0)

    # RGB-HSI Model: load the model_additional_layer that classifier if region is a character or not
    classification_model = load_model("/Users/shoaibanwar/PycharmProjects/neural_test/fyp_models/modelv4/model.json",
                                       "/Users/shoaibanwar/PycharmProjects/neural_test/fyp_models/modelv4/model.h5")

    # print("pred_nonchar", predict_region(cv.imread("/Users/shoaibanwar/PycharmProjects/neural_test/resources/non_char_0.png"), classification_model))
    # print("pred_char", predict_region(cv.imread("/Users/shoaibanwar/PycharmProjects/neural_test/resources/char_0.png"),
    #                               classification_model))
    # cv.waitKey(0)

    img_to_detect_plate = cv.imread("/Users/shoaibanwar/Downloads/DVLA-number-plates-2017-67-new-car-847566.jpg")
    # img_to_detect_plate = cv.resize(img_to_detect_plate, (213, 200))
    regions_to_test = sliding_window(img_to_detect_plate, 8, [19, 28])

    for region in regions_to_test:
        label = predict_region(region[4], classification_model)
        if label == 1:
            cv.rectangle(img_to_detect_plate, (region[0], region[1]), (region[2], region[3]), (255, 0, 0), 1)
            # cv.namedWindow(str(label))
            # cv.imshow(str(label), region[4])
            # cv.waitKey(0)
        print("Prediction: ", label)

    cv.imshow("ROIed Image", img_to_detect_plate)
    cv.waitKey(0)