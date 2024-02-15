import cv2 as cv
from cv2 import aruco
import numpy as np

# Loading all pre-built dictionaries
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

# Opening video capture (you can change 'your_video.mp4' to the path of your video file)
cap = cv.VideoCapture(3)

# Get the width and height of the original video
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Set the desired width and calculate the corresponding height to maintain the aspect ratio
desired_width = 800  # Change this value to your preferred width
desired_height = int((desired_width / width) * height)

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    # Resize the frame to fit the screen
    frame = cv.resize(frame, (desired_width, desired_height))

    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Defining lists
    corners = []
    ID = []
    centres = []
    z_rot = []

    # Store the center of the marker with ID 100
    center_id_100 = np.array([0.0, 0.0])

    # Checking the frame with each pre-loaded library
    for i in ARUCO_DICT.values():
        dictionary = aruco.getPredefinedDictionary(i)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
        marker_corners, markers_ID, reject = detector.detectMarkers(frame)
        if marker_corners is not None and markers_ID is not None:
            corners.append(marker_corners)
            ID.append(markers_ID.flatten())  # Flatten the ID array

            break

    # Calculating centre and angle of each marker
    for mark in corners:
        for dv in mark:
            for c0, c1, c2, c3 in dv:
                mx = (c0[0] + c1[0] + c2[0] + c3[0]) / 4
                my = (c0[1] + c1[1] + c2[1] + c3[1]) / 4
                c = [mx, my]
                centres.append(c)

    # Updating frame to show the values and data found
    if len(marker_corners) > 0:
        aruco.drawDetectedMarkers(frame, marker_corners, markers_ID, (0, 255, 0))

    for centre in centres:
        cv.circle(frame, (int(centre[0]), int(centre[1])), 5, (255, 0, 255), -1)

    cv.imshow("Frame", frame)

    # Press 'q' to exit the loop
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
