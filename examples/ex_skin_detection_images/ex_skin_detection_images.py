#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#In this example I use the range detector class to detect skin in two pictures
#The range detector find which pixels are included in a specific range.
#The hardest part is to find the correct boundaries for the range and tune
#the detector with the right morphing operation in order to have clean results
#and remove noise. The filter use HSV color representation (https://en.wikipedia.org/wiki/HSL_and_HSV)
import colorgram
import sys
sys.path[0] = "/usr/local/lib/python2.7/dist-packages"

import numpy as np
import cv2
from deepgaze.color_detection import RangeColorDetector

#Firs image boundaries
min_range = np.array([0, 48, 70], dtype = "uint8") #lower HSV boundary of skin color
max_range = np.array([20, 150, 255], dtype = "uint8") #upper HSV boundary of skin color
my_skin_detector = RangeColorDetector(min_range, max_range) #Define the detector object
image = cv2.imread("rodo.jpg") #Read the image with OpenCV
#We do not need to remove noise from this image so morph_opening and blur are se to False
image_filtered = my_skin_detector.returnFiltered(image, morph_opening=False, blur=False, kernel_size=3, iterations=1)
cv2.imwrite("rodo_filtered.jpg", image_filtered) #Save the filtered image


# Extract colour from filtered picture: 

# Extract 6 colors from an image.
colors = colorgram.extract('rodo_filtered.jpg', 6)

# colorgram.extract returns Color objects, which let you access
# RGB, HSL, and what proportion of the image was that color.
first_color = colors[1]
rgb = first_color.rgb # e.g. (255, 151, 210)
hsl = first_color.hsl # e.g. (230, 255, 203)
proportion  = first_color.proportion # e.g. 0.34

# RGB and HSL are named tuples, so values can be accessed as properties.
# These all work just as well:
red = rgb[0]
red = rgb.r
saturation = hsl[1]
saturation = hsl.s

# import pdb; pdb.set_trace()

from PIL import Image

array = np.zeros([100, 200, 3], dtype=np.uint8)
array[:,:100] = [rgb[0], rgb[1], rgb[2]] #Orange left side
array[:,100:] = [rgb[0], rgb[1], rgb[2]]   #Blue right side

img = Image.fromarray(array)
img.save('rodo_color.png')

# #Second image boundaries
# min_range = np.array([0, 58, 50], dtype = "uint8") #lower HSV boundary of skin color
# max_range = np.array([30, 255, 255], dtype = "uint8") #upper HSV boundary of skin color
# image = cv2.imread("tomb_rider_2.jpg") #Read the image with OpenCV
# my_skin_detector.setRange(min_range, max_range) #Set the new range for the color detector object
# #For this image we use one iteration of the morph_opening and gaussian blur to clear the noise
# image_filtered = my_skin_detector.returnFiltered(image, morph_opening=True, blur=True, kernel_size=3, iterations=1)
# cv2.imwrite("tomb_rider_2_filtered.jpg", image_filtered) #Save the filtered image
