import os
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description= "Image Mosaicing")
parser.add_argument('path_to_directory_containing_2_images', nargs='+',type=str)
parser.add_argument('-normalize', type=int,choices=[0,1])
args = parser.parse_args()

imgs_path = "../" + str(args.path_to_directory_containing_2_images[0])
images = os.listdir(imgs_path)
   

""" Train Image--> Right image
    Query Image--> Left image"""
    
pointsA, pointsB = [],[] # pointsA-->train img, pointsB--> query img
# reading the images
train_img = cv2.imread(os.path.join(imgs_path,images[0]))
query_img = cv2.imread(os.path.join(imgs_path,images[1]))
queryImg = query_img.copy()
trainImg = train_img.copy()

""" Generating Mouse clicking function """
def query_click_event(event, x, y, flags, params):

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(query_img, (x,y), 3, (0,0,255), -1)
        pointsB.append([x, y])

        cv2.imshow('Query image', query_img)
def train_click_event(event, x, y, flags, params):

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(train_img, (x,y), 3, (0,0,255), -1)
        pointsA.append([x, y])

        cv2.imshow('Train image', train_img)
# selecting point correspondences        
cv2.namedWindow('Query image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Query image', 640, 480)
cv2.imshow('Query image', query_img)
cv2.moveWindow('Query image', 150,175)
cv2.setMouseCallback('Query image', query_click_event)

cv2.namedWindow('Train image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Train image', 640, 480)
cv2.imshow('Train image', train_img)
cv2.moveWindow('Train image', 791,174)
cv2.setMouseCallback('Train image', train_click_event)
cv2.waitKey()   
cv2.destroyAllWindows()

# to find the scaling constant
def get_scaling_value(points):
    mean = np.mean(points,0)
    scale = (points-mean)**2
    scale = 0.5 * np.sum(scale,axis=1)
    scale = np.sqrt(np.mean(scale))
    return scale
# points normalization  
def normalize_image_points(points):
    """
    Input: 2D list with x,y image points
    Output:
    """
    points = np.array(points) 
    mean = np.mean(points,0)
    # define similarity transformation
    # no rotation, scaling using sdv and setting centroid as origin
    s = get_scaling_value(points)
    T = np.array([[1/s, 0, -(mean[0])/s],
                  [0, 1/s, -(mean[1])/s],
                  [0,   0, 1]])
    # print(T)
    points = np.dot(T, np.concatenate((points.T, np.ones((1, points.shape[0])))))

    # retrieve normalized image in the original input shape 
    points = points[0:2].T
    return points, T
# calculating homography matrix
def homography_matrix(pointsA,pointsB):
    H,_ = cv2.findHomography(pointsA, pointsB)
    return H
# without  normalization
if int(args.normalize) == 0:        
    
    path = "../results/campus_no_normalization.jpg" # for saving the mosaiced image
    # set data points to numpy arrays
    pointsA = np.array(pointsA)
    pointsB = np.array(pointsB)
    print(pointsA.shape)
    H = homography_matrix(pointsA,pointsB)
    
# with  normalization     
else:
    path = "../results/campus_with_normalization.jpg"
    pointsA,T1 = normalize_image_points(pointsA)
    pointsB,T2 = normalize_image_points(pointsB)
    Hn = homography_matrix(pointsA,pointsB) # normalization homography matrix
    H = np.dot(np.dot(np.linalg.inv(T2),Hn),T1) # denormalization
 
# cropping
def cropify(result):
    cropped_img = cv2.copyMakeBorder(result, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))
    
    gray = cv2.cvtColor( cropped_img, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray, 0, 255 , cv2.THRESH_BINARY)[1]
    # finding the contours
    contours,hierarchy = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # finding the largest contours by area
    areaOI = max(contours, key=cv2.contourArea)
    
    mask = np.zeros(thresh_img.shape, dtype="uint8")
    # bounding a rectangle with the largest contour
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
    
    cropped_img =  cropped_img[y:y + h, x:x + w]
    
    return  cropped_img

def img_stitching(H, trainImg, queryImg):
    # Apply panorama correction
    width = trainImg.shape[1] + queryImg.shape[1]
    height = trainImg.shape[0]
    
    result = cv2.warpPerspective(trainImg, H, (width, height))
    result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg
    stitched_img = cropify(result)
    return stitched_img

stitched_img = img_stitching(H, trainImg, queryImg)
cv2.imshow('Panorama',stitched_img)
cv2.imwrite(path,stitched_img)
cv2.waitKey()   
cv2.destroyAllWindows()