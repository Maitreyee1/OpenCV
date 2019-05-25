# -*- coding: utf-8 -*-
"""
Created on Sat May 25 16:16:51 2019

@author: maitr
"""

# =============================================================================
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# img_raw = cv2.imread('image.jpg')
# 
# print(type(img_raw))
# 
# 
# img_raw.shape
# 
# #plt.imshow(img_raw)
# 
# img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
# plt.imshow(img_rgb)
# 
# =============================================================================
# =============================================================================
# import cv2
# img = cv2.imread('image.jpg')
# while True:
#     cv2.imshow('mandrill',img)
# 
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
# 
# 
# cv2.destroyAllWindows()
# 
# 
# =============================================================================

import numpy as np
import cv2
import matplotlib.pyplot as plt
#%matplotlib inline

def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


#Loading the image to be tested
test_image = cv2.imread('baby2.png')
#test_image=convertToRGB(test_image)

#Converting to grayscale
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

# Displaying the grayscale image
plt.imshow(test_image_gray, cmap='gray')
#Since we know that OpenCV loads an image in BGR format, so we need to convert it into RBG format to be able to display its true colors. Let us write a small function for that.


haar_cascade_face = cv2.CascadeClassifier('D:\Study\Face_Detection\data\haarcascades\haarcascade_frontalface_default.xml')

faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.2, minNeighbors = 5);

# Let us print the no. of faces found
print('Faces found: ', len(faces_rects))

for (x,y,w,h) in faces_rects:
     cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#convert image to RGB and show image

plt.imshow(convertToRGB(test_image))




