#!/usr/bin/env python

# Test skin detection algorithm for datalab project

import colorgram
import sys
from face_detect import detect_faces
sys.path[0] = "/usr/local/lib/python2.7/dist-packages"
import glob
import colorgram
import numpy as np
import cv2
from deepgaze.color_detection import RangeColorDetector
from PIL import Image
import os.path
import math

# min_range = np.array([0, 48, 70], dtype = "uint8") #lower HSV boundary of skin color
# max_range = np.array([20, 150, 255], dtype = "uint8") #upper HSV boundary of skin color

#First image boundaries
min_range = np.array([0, 48, 70], dtype = "uint8") #lower HSV boundary of skin color
max_range = np.array([20, 150, 255], dtype = "uint8") #upper HSV boundary of skin color
my_skin_detector = RangeColorDetector(min_range, max_range) #Define the detector object


# image_count = len(glob.glob1("/home/arturo/GitHub/deepgaze/skin_detection_datalab/diputados_pri","*.jpg")) # People in folder


folder_name = "/home/arturo/GitHub/deepgaze/skin_detection_datalab/images"

data_path = os.path.join(folder_name,'*g')

files = glob.glob(data_path)

number_of_files = len(files)

# import pdb; pdb.set_trace()
all_rgb = [None] * number_of_files
proportions = [None] * number_of_files

# file index
j = 0

for image_name in files:
	try:
		detect_faces(image_name)
	except:
		pass
	if(os.path.exists(image_name[:52] + 'results/face_' + image_name[59:])):
		image = cv2.imread(image_name[:52] + 'results/face_' + image_name[59:]) #Read the image with OpenCV
		# filtered_image_name = image_name[:52] + 'results/face_filtered_' + image_name[66:]
	else:
		image = cv2.imread(image_name)
		# filtered_image_name = image_name[:52] + 'results/face_filtered_' + image_name[66:]
	filtered_image_name = image_name[:52] + 'results/face_filtered_' + image_name[59:]
	image_filtered = my_skin_detector.returnFiltered(image, morph_opening = True, blur = False, kernel_size = 3, iterations = 3)
	image_filtered = cv2.GaussianBlur(image_filtered,(15,15),10)
	cv2.imwrite(filtered_image_name, image_filtered) #Save the filtered image
	# Extract 3 colors from an image.
	colors = colorgram.extract(filtered_image_name, 10)
	# colorgram.extract returns Color objects, which let you access
	# RGB, HSL, and what proportion of the image was that color.

	#import pdb;pdb.set_trace()
	limits = [30, 30, 30]
	out_of_bounds = [29, 29, 29]
	i = 0
	while((out_of_bounds[0] < limits[0] or out_of_bounds[1] < limits[1] or out_of_bounds[2] < limits[2]) and i <= 3):
		rgb = colors[i].rgb
		out_of_bounds = rgb
		proportion = colors[i].proportion
		i = i + 1	

	if proportion >= .15:		
		try:
			array = np.zeros([100, 200, 3], dtype=np.uint8)
			array[:,:100] = [rgb[0], rgb[1], rgb[2]]  
			array[:,100:] = [rgb[0], rgb[1], rgb[2]] 

			img = Image.fromarray(array)
			image_color_name = image_name[:52] + 'results/colour_' + image_name[59:]
			img.save(image_color_name)

			# store rbg values in list: 
			proportions[j] = proportion
			all_rgb[j] = rgb
			j = j + 1
		except:
			pass

# get average rgb value out of all files and create image for visualization

all_rgb = [rgb for rgb in all_rgb if rgb is not None]

r = 0
g = 0
b = 0

for sample in all_rgb:
	r = r + sample[0]**2
	g = g + sample[1]**2
	b = b + sample[2]**2

r = math.sqrt(r / j)
g = math.sqrt(g / j)
b = math.sqrt(b / j)

average_rgb = [r, g, b]

array = np.zeros([100, 200, 3], dtype=np.uint8)
array[:,:100] = [average_rgb[0], average_rgb[1], average_rgb[2]]  
array[:,100:] = [average_rgb[0], average_rgb[1], average_rgb[2]] 

img = Image.fromarray(array)
image_color_name = '/home/arturo/GitHub/deepgaze/skin_detection_datalab/results/average_colour.jpg'
img.save(image_color_name)