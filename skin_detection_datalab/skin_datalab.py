#!/usr/bin/env python

# Test skin detection algorithm for datalab project


import sys
sys.path[0] = "/usr/local/lib/python2.7/dist-packages"
import pdb
import glob
import colorgram
import numpy as np
import cv2
from deepgaze.color_detection import RangeColorDetector

#Firs image boundaries
min_range = np.array([0, 48, 70], dtype = "uint8") #lower HSV boundary of skin color
max_range = np.array([20, 150, 255], dtype = "uint8") #upper HSV boundary of skin color
my_skin_detector = RangeColorDetector(min_range, max_range) #Define the detector object

image_count = len(glob.glob1("/home/arturo/GitHub/deepgaze/skin_detection_datalab","*.jpg"))

for i in range(1, image_count + 1):
	image_name = "img (" + str(i) + ").jpg" 
	image = cv2.imread(image_name) #Read the image with OpenCV
	image_filtered = my_skin_detector.returnFiltered(image, morph_opening=False, blur=False, kernel_size=3, iterations=1)
	new_image_name = "filtered_" + image_name
	cv2.imwrite(new_image_name, image_filtered) #Save the filtered image

# image = cv2.imread("vale.jpg") #Read the image with OpenCV
# #We do not need to remove noise from this image so morph_opening and blur are se to False
# image_filtered = my_skin_detector.returnFiltered(image, morph_opening=False, blur=False, kernel_size=3, iterations=1)
# cv2.imwrite("vale_filtered.jpg", image_filtered) #Save the filtered image

# #Second image boundaries
# min_range = np.array([0, 58, 50], dtype = "uint8") #lower HSV boundary of skin color
# max_range = np.array([30, 255, 255], dtype = "uint8") #upper HSV boundary of skin color
# image = cv2.imread("tomb_rider_2.jpg") #Read the image with OpenCV
# my_skin_detector.setRange(min_range, max_range) #Set the new range for the color detector object
# #For this image we use one iteration of the morph_opening and gaussian blur to clear the noise
# image_filtered = my_skin_detector.returnFiltered(image, morph_opening=True, blur=True, kernel_size=3, iterations=1)
# cv2.imwrite("tomb_rider_2_filtered.jpg", image_filtered) #Save the filtered image
