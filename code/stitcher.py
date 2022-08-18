# Importing required libraries
import cv2
import numpy as np
import argparse
import os

# Create the parser
parser = argparse.ArgumentParser(description= "Image Mosaicing")
# Adding arguments
parser.add_argument('path_to_directory_containing_2_images', nargs='+',type=str)
args = parser.parse_args()

# Images path 
images = []
imgs_path = "../" + str(args.path_to_directory_containing_2_images[0])
images_list = os.listdir(imgs_path)
for image in images_list:
    img_path = imgs_path + "\\"+ image
    img = cv2.imread(img_path)
    images.append(img)


# cropping
def cropify(result):
    stitched_img = cv2.copyMakeBorder(result, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))
    print(stitched_img.shape)
    
    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray, 0, 255 , cv2.THRESH_BINARY)[1]
    
    contours,hierarchy = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areaOI = max(contours, key=cv2.contourArea)
    
    mask = np.zeros(thresh_img.shape, dtype="uint8")
    x, y, w, h = cv2.boundingRect(areaOI)
    cv2.rectangle(mask, (x,y), (x + w, y + h), 255, -1)
    
    minRectangle = mask.copy()
    sub = mask.copy()
    
    while cv2.countNonZero(sub) > 0:
        minRectangle = cv2.erode(minRectangle, None)
        sub = cv2.subtract(minRectangle, thresh_img)
    
    contours,hierarchy = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areaOI = max(contours, key=cv2.contourArea)
    
    x, y, w, h = cv2.boundingRect(areaOI)
    
    stitched_img = stitched_img[y:y + h, x:x + w]
    return stitched_img


# OpenCV stitcher API
stitcher = cv2.Stitcher_create()
(status, stitched_img) = stitcher.stitch(images)

if (status == cv2.STITCHER_OK):
    print("Panorama generated")
    stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))

    stitch_img = cropify(stitched_img)

    cv2.imshow("Stitched Image", stitch_img)
    cv2.imwrite("../results/StitcherAPI.jpg",stitch_img)
    cv2.waitKey(0)

else:
    print("Images could not be stitched!")
    print("Likely not enough keypoints being detected!")







