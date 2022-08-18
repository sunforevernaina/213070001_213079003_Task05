import numpy as np
import random
import math


def homography(pairs):
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)
    H = H/H[2, 2] # standardize to let w*H[2,2] = 1
    return H

def get_error(points, H):
    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = points[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
    # Compute error
    errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2

    return errors

def Ransac(matches, threshold):
    num_iterations = math.inf
    iterations_done = 0
    num_sample = 4 #s

    prob_outlier = 0.5 # 50% outliers
    desired_prob = 0.99 # p
    
    T = (1- prob_outlier)* matches.shape[0] #T
    valid_inliers=[]
    valid_inliers_count=[]
    while num_iterations > iterations_done:
        idx = random.sample(range(len(matches)), 4) # 4 indices to find points for homography
        point = [matches[i] for i in idx ]
        points = np.array(point)
        H = homography(points)
        #  avoid dividing by zero 
        if np.linalg.matrix_rank(H) < 3:
            continue
        # count the inliers within the threshold
        errors = get_error(matches, H)
        
        threshold = math.sqrt(3.84)*errors.std() # t = sqrt(3.84)sigma
        index = np.where(errors < threshold)[0]
        inliers = matches[index]
        num_inliers = len(inliers)
        #check for inlier size
        if num_inliers>T:
            valid_inliers.append(inliers)
            valid_inliers_count.append(num_inliers)

        prob_outlier = 1 - num_inliers /len(matches)
        num_iterations = math.ceil(math.log(1 - desired_prob)/math.log(1 - (1 - prob_outlier)**num_sample))
        iterations_done = iterations_done + 1

    inliers = valid_inliers[valid_inliers_count.index(max(valid_inliers_count))]

    return inliers
            
   