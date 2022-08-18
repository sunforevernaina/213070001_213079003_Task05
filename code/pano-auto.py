import cv2
import matplotlib.pyplot as plt
import numpy as np
from ransac import Ransac
import argparse
import os
parser = argparse.ArgumentParser(description= "Image Mosaicing")
parser.add_argument('path_to_directory_containing_2_images', nargs='+',type=str)
parser.add_argument('-mode', type=str, required=True,choices = ['custom-ransac','auto-ransac'], help="select mode")
parser.add_argument('-normalize', type=int, choices=[0,1], required=True, help="normalization")
args = parser.parse_args()

imgs_path = "../" + str(args.path_to_directory_containing_2_images[0])
images = os.listdir(imgs_path)

def get_scaling_value(points):
    mean = np.mean(points,0)
    scale = (points-mean)**2
    scale = 0.5 * np.sum(scale,axis=1)
    scale = np.sqrt(np.mean(scale))
    return scale
    
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
    # print('dsjfhsdfh',s)
    T = np.array([[1/s, 0, -(mean[0])/s],
                  [0, 1/s, -(mean[1])/s],
                  [0,   0, 1]])
    # print(T)
    points = np.dot(T, np.concatenate((points.T, np.ones((1, points.shape[0])))))

    # retrieve normalized image in the original input shape 
    points = points[0:2].T
    return points, T


# Read image and convert them to gray!!
def read_image(path):
    img = cv2.imread(path)
    img_gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_gray, img, img_rgb

left_gray, left_origin, left_rgb = read_image(os.path.join(imgs_path,images[1]))
right_gray, right_origin, right_rgb = read_image(os.path.join(imgs_path,images[0]))

def SIFT(img):
    siftDetector= cv2.xfeatures2d.SIFT_create()

    kp, des = siftDetector.detectAndCompute(img, None)
    return kp, des

def plot_sift(gray, rgb, kp):
    tmp = rgb.copy()
    img = cv2.drawKeypoints(gray, kp, tmp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img

# Better result when using gray
kp_left, des_left = SIFT(left_gray)
kp_right, des_right = SIFT(right_gray)

def matcher(kp1, des1, img1, kp2, des2, img2, threshold):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    good1 = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append([m])
            good1.append(m)

    matches1 = []
    for pair in good:
        matches1.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))
    matches1 = np.array(matches1)
    return matches1,good

match,good = matcher(kp_left, des_left, left_rgb, kp_right, des_right, right_rgb, 0.75)
print("Number of SIFT Matches:",len(match))
def plot_matches(match, total_img,path,title):
    match_img = total_img.copy()
    offset = total_img.shape[1]/2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8')) #　RGB is integer type
    ax.plot(match[:, 0], match[:, 1], 'xr')
    ax.plot(match[:, 2] + offset, match[:, 3], 'xr')
     
    img = ax.plot([match[:, 0], match[:, 2] + offset], [match[:, 1], match[:, 3]],
            'r', linewidth=0.5)
    fig.savefig(path)
    plt.title(title, fontsize="18")
    plt.show()
total_img = np.concatenate((left_rgb, right_rgb), axis=1)
plot_matches(match, total_img,"../results/SIFT_MATCHES",'SIFT_MATCHES')# Good matches


# cropping
def cropify(result):
    stitched_img = cv2.copyMakeBorder(result, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))
    print(stitched_img.shape)
    
    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray, 0, 255 , cv2.THRESH_BINARY)[1]
    
    #contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours,hierarchy = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)
    
    mask = np.zeros(thresh_img.shape, dtype="uint8")
    x, y, w, h = cv2.boundingRect(areaOI)
    cv2.rectangle(mask, (x,y), (x + w, y + h), 255, -1)
    
    minRectangle = mask.copy()
    sub = mask.copy()
    
    while cv2.countNonZero(sub) > 0:
        minRectangle = cv2.erode(minRectangle, None)
        sub = cv2.subtract(minRectangle, thresh_img)
    
    
    #contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours,hierarchy = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)
    
    x, y, w, h = cv2.boundingRect(areaOI)
    
    stitched_img = stitched_img[y:y + h, x:x + w]
    return stitched_img

#-----------------------------------------------------------------------------------------------------"

#===CUSTOM RANSAC===
if (args.mode == 'custom-ransac'):
    inliers = Ransac(match,4)       

    print("Number of SIFT+RANSAC Matches:",len(inliers))
    plot_matches(inliers, total_img,"../results/SIFT+RANSAC Matches","SIFT+RANSAC Matches") # show inliers matches 
    src = np.float32(inliers[:,0:2])    
    dst = np.float32(inliers[:,2:4])

    if int(args.normalize)==0:        
        # set data points to numpy arrays
        src = np.array(src)
        dst = np.array(dst)
        H,_ = cv2.findHomography(src,dst)
        print("Homography Matrix Custom",H,sep="\n")
        
    
    else:
        src,T1 = normalize_image_points(src)
        dst,T2 = normalize_image_points(dst)
        H,_ = cv2.findHomography(src,dst)
        print("Homography Matrix Custom",H,sep="\n")
    
   
    
      # Apply panorama correction
    result = cv2.warpPerspective(right_origin,H,(left_origin.shape[1] + right_origin.shape[1], left_origin.shape[0]))
    result[0:right_origin.shape[0], 0:right_origin.shape[1]] = right_origin

    stitched_img = cropify(result)
    cv2.imwrite('../results/Custom_RANSAC_Panorama.jpg',stitched_img)
    cv2.imshow('Custom RANSAC Panorama', stitched_img)
    cv2.waitKey()   
    cv2.destroyAllWindows()



#---------------------------------------------------------------------------------------------------
#===AUTO-RANSAC===
else:
    #Storing coordinates of points corresponding to the matches found in both the images
        BaseImage_pts = []
        SecImage_pts = []
        for Match in good:
            BaseImage_pts.append(kp_left[Match[0].queryIdx].pt)
            SecImage_pts.append(kp_right[Match[0].trainIdx].pt)
        

        # Changing the datatype to "float32" for finding homography
        BaseImage_pts = np.float32(BaseImage_pts)
        SecImage_pts = np.float32(SecImage_pts)
        if int(args.normalize)==0:        
            # set data points to numpy arrays
            src = np.array(BaseImage_pts)
            dst = np.array(SecImage_pts)
            # Finding the homography matrix(transformation matrix).
            HomographyMatrix, status = cv2.findHomography(SecImage_pts, BaseImage_pts, cv2.RANSAC, 4.0)
            print("Homography Matrix Auto",HomographyMatrix,sep="\n")
        elif int(args.normalize)==1:
            src,T1 = normalize_image_points(BaseImage_pts)
            dst,T2 = normalize_image_points(SecImage_pts)
            # Finding the homography matrix(transformation matrix).
            HomographyMatrix, status = cv2.findHomography(SecImage_pts, BaseImage_pts, cv2.RANSAC, 4.0)
            print("Homography Matrix Auto",HomographyMatrix,sep="\n")
        
        else:
            print("Choose normalization values 0 or 1")
            
    
        # Apply panorama correction
        result = cv2.warpPerspective(right_origin,HomographyMatrix,(left_origin.shape[1] + right_origin.shape[1], left_origin.shape[0]))
        result[0:left_origin.shape[0], 0:left_origin.shape[1]] = left_origin

        stitched_img = cropify(result)
        cv2.imwrite('../results/Auto_RANSAC_Panorama.jpg',stitched_img)
        cv2.imshow('Auto RANSAC Panorama',stitched_img)
        cv2.waitKey()   
        cv2.destroyAllWindows()