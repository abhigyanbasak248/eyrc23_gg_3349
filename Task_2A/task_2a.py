'''
*****************************************************************************************
*
*        		===============================================
*           		Geo Guide (GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 2A of Geo Guide (GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			[ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:			task_2a.py
# Functions:		detect_ArUco_details
# 					[ Comma separated list of functions in this file ]


####################### IMPORT MODULES #######################
## You are not allowed to make any changes in this section. ##
## You have to implement this task with the five available  ##
## modules for this task                                    ##
##############################################################
import numpy as np
import cv2
import math
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def calc_centre(ArUco_corners):
    centre = []
    for corner in ArUco_corners.values():
        tl = corner[0]
        tr = corner[1]
        br = corner[2]
        bl = corner[3]
        x = int((tl[0] + tr[0] + br[0] + bl[0])/4)
        y = int((tl[1] + tr[1] + br[1] + bl[1])/4)
        centre.append([x, y])
    return centre

def calc_degree(Detected_ArUco_markers):
    ArUco_marker_angles = {}
    for key in Detected_ArUco_markers:
        corners = Detected_ArUco_markers[key]
        tl = corners[1]
        tr = corners[0]
        br = corners[2]
        bl = corners[3]
        top = (tl[0]+tr[0])/2, -((tl[1]+tr[1])/2)
        centre = (tl[0]+tr[0]+bl[0]+br[0])/4, -((tl[1]+tr[1]+bl[1]+br[1])/4)
        try:
            angle = 90 - round(math.degrees(np.arctan((top[1]-centre[1])/(top[0]-centre[0]))))
        except:
            # add some conditions for 90 and 270
            if(top[1]>centre[1]):
                angle = 90
            elif(top[1]<centre[1]):
                angle = 270
        if(top[0] >= centre[0] and top[1] < centre[1]):
            angle = 360 + angle
        elif(top[0]<centre[0]):
            angle = 180 + angle
        if ((360 - angle) < angle):
            angle = -(360 - angle)
        ArUco_marker_angles.update({key: angle})
    return ArUco_marker_angles



##############################################################

def detect_ArUco_details(image):

    """
    Purpose:
    ---
    This function takes the image as an argument and returns two dictionaries where one
    contains details regarding the center coordinates and orientation of the marker
    and the second dictionary contains values of the 4 corner coordinates of the marker. 
    
    First output: The dictionary `ArUco_details_dict` should should have the id of the marker 
    as the key and the value corresponding to that id should be a list containing the following details
    in this order: [[center_x, center_y], angle from the vertical]     
    This order should be strictly maintained in the output
    Datatypes:
    1. id - int
    2. center coordinates - int
    3. angle - int, x and y coordinates should be combined as a list for each corner

    Second output: The dictionary `ArUco_corners` should contain the id of the marker as key and the
    corresponding value should be an array of the coordinates of 4 corner points of the markers
    Datatypes:
    1. id - int
    2. corner coordinates - each coordinate value should be float, x and y coordinates should 
    be combined as a list for each corner

    Input Arguments:
    ---
    `image` :	[ numpy array ]
            numpy array of image returned by cv2 library
    Returns:
    ---
    `ArUco_details_dict` : { dictionary }
            dictionary containing the details regarding the ArUco marker

    `ArUco_corners` : { dictionary }
            dictionary containing the details regarding the corner coordinates of the ArUco marker
    
    Example call:
    ---
    ArUco_details_dict, ArUco_corners = detect_ArUco_details(image)

    Example output for 2 markers in an image:
    ---
    * ArUco_details_dict = {9: [[311, 490], 0], 3: [[158, 175], -22]}
    * ArUco_corners = 
       {9: array([[211., 389.],
       [412., 389.],
       [412., 592.],
       [211., 592.]], dtype=float32), 
       3: array([[109.,  46.],
       [284., 118.],
       [207., 304.],
       [ 33., 232.]], dtype=float32)}
    """    
    ArUco_details_dict = {}
    ArUco_corners = {}
    corners = []
    ids = []
    ##############	ADD YOUR CODE HERE	##############
    for i in ARUCO_DICT.values():
        dictionary = cv2.aruco.getPredefinedDictionary(i)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(image)
        if (corners != ()):
            ids = ids.tolist()
            corners = list(corners)
            for i,j in enumerate(ids):
                for k in j:
                    ArUco_corners[k] = np.squeeze(corners[i])
    centre = calc_centre(ArUco_corners)
    degree = calc_degree(ArUco_corners)
    # print(centre, degree)
    for i,j in enumerate(ArUco_corners.keys()):
        # print(i, j)
        ArUco_details_dict[j] = [centre[i], degree[j]]
    for ids, details in ArUco_details_dict.items():
        center = details[0]
        # print(center)
    # print(ArUco_corners)
    # print(ArUco_details_dict)
    ##################################################
    
    return ArUco_details_dict, ArUco_corners 

######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THE CODE BELOW #########	

def mark_ArUco_image(image,ArUco_details_dict, ArUco_corners):

    for ids, details in ArUco_details_dict.items():
        center = details[0]
        cv2.circle(image, center, 5, (0,0,255), -1)

        corner = ArUco_corners[int(ids)]
        cv2.circle(image, (int(corner[0][0]), int(corner[0][1])), 5, (50, 50, 50), -1)
        cv2.circle(image, (int(corner[1][0]), int(corner[1][1])), 5, (0, 255, 0), -1)
        cv2.circle(image, (int(corner[2][0]), int(corner[2][1])), 5, (128, 0, 255), -1)
        cv2.circle(image, (int(corner[3][0]), int(corner[3][1])), 5, (25, 255, 255), -1)

        tl_tr_center_x = int((corner[0][0] + corner[1][0]) / 2)
        tl_tr_center_y = int((corner[0][1] + corner[1][1]) / 2) 

        cv2.line(image,center,(tl_tr_center_x, tl_tr_center_y),(255,0,0),5)
        display_offset = int(math.sqrt((tl_tr_center_x - center[0])**2+(tl_tr_center_y - center[1])**2))
        cv2.putText(image,str(ids),(center[0]+int(display_offset/2),center[1]),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        angle = details[1]
        cv2.putText(image,str(angle),(center[0]-display_offset,center[1]),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return image

if __name__ == "__main__":

    # path directory of images in test_images folder
    img_dir_path = "public_test_cases/"

    marker = 'aruco'

    for file_num in range(0,2):
        img_file_path = img_dir_path +  marker + '_' + str(file_num) + '.png'

        # read image using opencv
        img = cv2.imread(img_file_path)

        print('\n============================================')
        print('\nFor '+ marker  +  str(file_num) + '.png')
   
        ArUco_details_dict, ArUco_corners = detect_ArUco_details(img)
        print("Detected details of ArUco: " , ArUco_details_dict)

        # displaying the marked image
        img = mark_ArUco_image(img, ArUco_details_dict, ArUco_corners) 
        cv2.imshow("Marked Image",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
