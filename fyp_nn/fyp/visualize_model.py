from keras.utils import plot_model
from fyp.detection import load_model


if __name__ == "__main__":
    model = load_model("/Users/shoaibanwar/PycharmProjects/neural_test/fyp_models/model.json",
                        "/Users/shoaibanwar/PycharmProjects/neural_test/fyp_models/model.h5")

    plot_model(model, to_file='/Users/shoaibanwar/PycharmProjects/neural_test/fyp_models/model.png',
               show_shapes = True)