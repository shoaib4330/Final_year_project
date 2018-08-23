#Automatic Number Plate Recognition Using Multiple Cameras

This project perform the following tasks:

1) Takes as input a video Stream
2) Goes through a first Model, that extracts Front/Backs of any cars present inside video frames.
3) Extracted Front/Backs of Vehicles are preprocessed (most importantly resize to bring to low resolution).

Next phase is to perform number plate detection.

This project has two repositories, This repository contains modules and works on detection of Custom Number Plates, whereas the other repository
is trained to recognize Standard Pakistani-Punjab based number plates.

To setup and get this project going, do the following:

1) Download Anaconda (Make Sure you Anaconda Python 3.5 version)
2) Anaconda creates an environment for every project.
3) conda create --name myenv (replace myenv with your desired name of environment)

It is time to install dependencies to the environment that are needed for our project.

4) Install dependencies using: conda install {dependency}
5) Following are the dependencies:
	- Tensorflow
	- Matplotlib
	- Shapely
	- Keras
	- Pandas
	- Numpy
	- Scipy
	- OpenCv
	- (All above dependencies can be added using "conda create install {dependency-name}"

#Dependencies are setup.
- Now it is time to set some configuration parameter.
- Project Root Directory is named as "fyp_nn"
- Under this directory there is a file name "Constants.py", This file declares absolute paths to all the required models and files (since pyCharm does not
support Relative Paths).

Following paths are to be set according to your machine:

All you need to do is, prepend "C:\Users\VenD\Desktop\ANPR\\" with your own system specific project path.
--------------------------------
---------------- Car Detection Module Constants and Configuration parameters ----------------------------
--------------------------------
prototxt_path = r"C:\Users\VenD\Desktop\ANPR\\fyp_nn\module_car_extraction\prototxt.txt"
caffemodel_path = r"C:\Users\VenD\Desktop\ANPR\\fyp_nn\module_car_extraction\model.caffemodel"

--------------------------------
---------------- East-Text-Detection Module Constants and Configuration parameters ----------------------------
--------------------------------
predictor_path = r"C:\Users\VenD\Desktop\ANPR\\fyp_nn\east_resources\east_icdar2015_resnet_v1_50_rbox"
check_point_path = r"C:\Users\VenD\Desktop\ANPR\fyp_nn\east_resources\east_icdar2015_resnet_v1_50_rbox"
output_path = r"C:\Users\VenD\Desktop\ANPR\fyp_nn\output"
test_images_path = r"C:\Users\VenD\Desktop\ANPR\fyp_nn\test_images"

- There is a directory named test_video under project root directory, add any test video/videos in this directory that has traffic (vehicles) in it.
- These video can be found in burnded dvd's submitted to the department. 

You're good to go.
