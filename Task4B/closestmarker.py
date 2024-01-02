import cv2 as cv
from cv2 import aruco
import numpy as np

# Loading all pre-built libraries
ARUCO_DICT = {
	"DICT_4X4_50": cv.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11
}

# Capture video from webcam (change the argument to your video file if needed)
cap = cv.VideoCapture(2)

# Moving ID
moving_id = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Defining lists
    corners = []
    ID = []
    centres = []
    z_rot = []

    # Store the center of the marker with ID 100
    center_id_100 = np.array([0.0, 0.0])

    # Checking the frame with each pre-loaded library
    for dict_name, dict_id in ARUCO_DICT.items():
        dictionary = aruco.getPredefinedDictionary(dict_id)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
        marker_corners, markers_ID, reject = detector.detectMarkers(frame)
        if marker_corners is not None and markers_ID is not None:
            corners.append(marker_corners)
            ID.append(markers_ID)

            # Print the name of the dictionary used
            print(f"Used Dictionary: {dict_name}")

            # Store the center of the marker with ID 100
            for idx, id_value in enumerate(markers_ID):
                if id_value == moving_id:
                    center_id_100 = np.mean(marker_corners[idx][0], axis=0)
                    break

            break

    # Calculating centre and angle of each marker
    for mark in corners:
        for dv in mark:
            for c0, c1, c2, c3 in dv:
                mx = (c0[0] + c1[0] + c2[0] + c3[0]) / 4
                my = (c0[1] + c1[1] + c2[1] + c3[1]) / 4
                c = [mx, my]
                centres.append(c)
                vector = c0 - c
                angle = np.arctan2(vector[1], vector[0])
                angle *= 180 / np.pi
                z_rot.append(abs(angle))

    # Find the minimum distance ID
    min_distance_id = None
    min_distance = float('inf')

    for idx, centre in enumerate(centres):
        distance = np.linalg.norm(np.array(center_id_100) - np.array(centre))
        if distance < min_distance and ID[0][idx] != moving_id:
            min_distance = distance
            min_distance_id = ID[0][idx]

    # Updating frame to show the values and data found
    if len(marker_corners) > 0:
        aruco.drawDetectedMarkers(frame, marker_corners, markers_ID, (0, 255, 0))

    for centre in centres:
        cv.circle(frame, (int(centre[0]), int(centre[1])), 5, (255, 0, 255), -1)

    # Print the minimum distance ID
    if min_distance_id is not None:
        print(f"Minimum distance ID: {min_distance_id}")

    cv.imshow("Frame", frame)

    # Press 'q' to exit the loop
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv.destroyAllWindows()