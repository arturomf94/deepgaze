from random import randint
import cv2
import sys
import os
      
CASCADE="Face_cascade.xml"
FACE_CASCADE=cv2.CascadeClassifier(CASCADE)

def detect_faces(image_path):
	# import pdb;pdb.set_trace()
	image=cv2.imread(image_path)
	image_grey=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=15,minSize=(25,25),flags=0)

	image_path = image_path[:52] + 'results/face_' + image_path[59:]

	for x,y,w,h in faces:
	    sub_img=image[y-12:y+h+12,x-12:x+w+12]
	    # os.chdir("Extracted")
	    cv2.imwrite(image_path, sub_img)
	    # os.chdir("../")
	    cv2.rectangle(image,(x,y),(x+w,y+h),(255, 255,0),2)

	#cv2.imshow("Faces Found",image)
	#if (cv2.waitKey(0) & 0xFF == ord('q')) or (cv2.waitKey(0) & 0xFF == ord('Q')):
	#	cv2.destroyAllWindows()

# if __name__ == "__main__":
	
# 	if not "Extracted" in os.listdir("."):
# 		os.mkdir("Extracted")
    
# 	if len(sys.argv) < 2:
# 		print "Usage: python Detect_face.py 'image path'"
# 		sys.exit()

# 	detect_faces(sys.argv[1])