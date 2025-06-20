# Author: Vaibhav Ganatra (t-vaganatra at microsoft dot com))

import cv2
import math
import numpy as np
import os
from typing import List

def calculate_frame_sharpness(img: np.ndarray) -> float:
    """
    calculates the sharpness of a frame using the variance of the Laplacian.
    """    
    # laplacian of image
    lap = cv2.Laplacian(img, cv2.CV_64F)

    # variance of laplacian
    return lap.var()

def crop_around_center(image: np.array, crop_dims: List =[0,0], center: List = None):
    """
    Crops the image around the center point with specified dimensions.
    """
    if center is None:
        height, width = image.shape[:2]
        c_y_min, c_x_min = height // 2 - crop_dims[1] // 2, width // 2 - crop_dims[1] // 2
        c_y_max, c_x_max = height // 2 + crop_dims[1] // 2, width // 2 + crop_dims[0] // 2
        image_crop = image[c_y_min:c_y_max, c_x_min:c_x_max]
    else:
        centerX, centerY = center
        c_y_min, c_x_min = centerY - crop_dims[1] // 2, centerX - crop_dims[1] // 2
        c_y_max, c_x_max = centerY + crop_dims[1] // 2, centerX + crop_dims[1] // 2
        image_crop = image[c_y_min:c_y_max, c_x_min:c_x_max]

    return image_crop

def plot_line(img: np.array, center: List, angle: float):                                          
    """
    function to plot_line at an angle
    """
    img = np.zeros_like(img)                                                
    x_min, y_min = 0, 0                                                     
    x_max, y_max = img.shape[1], img.shape[0]                               
                                                                            
    # finding extreme for this line                                         
    length = 1                                                              
    allowed_len = -1                                                        
    while True:                                                             
        x =  int(center[0] + length * math.cos(angle * np.pi / 180.0))  
        y =  int(center[1] + length * math.sin(angle * np.pi / 180.0))  
        if (x >= x_min and x < x_max) and (y >= y_min and y < y_max):   
            allowed_len = length                                            
        else:                                                               
            break                                                           
        length += 1                                                         
                                                                             
    x =  int(center[0] + allowed_len * math.cos(angle * np.pi / 180.0)) 
    y =  int(center[1] + allowed_len * math.sin(angle * np.pi / 180.0)) 
    img = img.astype(np.uint8)                                              
    img = cv2.line(img, (center[0], center[1]), (x,y), (255,255, 255), 1)
    return img

def process_mires(img: np.array, center: List, angle: float, weights=None):
    """
    getting the mires given the img and the line at an angle
    """    
    mask = plot_line(img.copy(), center, angle)
    line = mask.copy()
    mask[mask > 0] = 1
    masked_img = img * mask
    connectivity = 4  # You need to choose 4 or 8 for connectivity type
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(masked_img , connectivity , cv2.CV_32S)
    centroids = [list(x) for x in list(centroids)]
    if weights is not None:
        centroids = []
        for idx in np.unique(labels):
            y,x = np.argwhere(labels==idx)[:,0], np.argwhere(labels==idx)[:,1]
            centroids.append([np.average(x, weights=weights[y,x]), np.average(y, weights=weights[y,x]), np.mean(weights[y,x])])
            # plt.scatter(x,y, label=idx)

    centroids_new = []
    for centroid in centroids:
        r = math.sqrt((center[0]-centroid[0])**2 + (center[1]-centroid[1])**2)
        intensity = centroid[2]
        centroid = centroid[:2]
        centroid.append(r)
        centroid.append(intensity)
        centroids_new.append(centroid)
    return list(centroids_new[1:]), line

def dict_list(inp):
    """
    Converts a numpy array or list to a JSON serializable format.
    """
    if isinstance(inp, np.ndarray) or isinstance(inp, np.array):
        inp = inp.tolist()
        return inp
    raise Exception("Not JSON serializable")

def plot_supports(frame: np.array, left_angle: float, right_angle: float, center: List, out_dir: str):
    """
    """
    LENGTH = 250

    x,y = math.cos(math.radians(left_angle - 4.5)) * LENGTH, math.sin(math.radians(left_angle-4.5)) * LENGTH
    cv2.line(frame, (int(center[0]), int(center[1])), (int(center[0] + x), int(center[1] + y)), (0,0,255), 1)

    x,y = math.cos(math.radians(left_angle + 4.5)) * LENGTH, math.sin(math.radians(left_angle+4.5)) * LENGTH
    cv2.line(frame, (int(center[0]), int(center[1])), (int(center[0] + x), int(center[1] + y)), (0,0,255), 1)

    x,y = math.cos(math.radians(right_angle - 4.5)) * LENGTH, math.sin(math.radians(right_angle-4.5)) * LENGTH
    cv2.line(frame, (int(center[0]), int(center[1])), (int(center[0] + x), int(center[1] + y)), (0,0,255), 1)

    x,y = math.cos(math.radians(right_angle + 4.5)) * LENGTH, math.sin(math.radians(right_angle+4.5)) * LENGTH
    cv2.line(frame, (int(center[0]), int(center[1])), (int(center[0] + x), int(center[1] + y)), (0,0,255), 1)

    cv2.imwrite(os.path.join(out_dir, 'frame_supports.png'), frame)

def fetch_break_thresholds(mire_num: float):
    if mire_num <= 7:
        threshold = 350
    elif mire_num <= 17: # change to 17
        threshold = 300
    else:
        threshold = 260
    
    return threshold
