import cv2 as cv
from cv2 import aruco
import numpy as np
import csv

csv_name = 'lat_long.csv'
lat_lon = {}

with open(csv_name, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        ar_id, lat, lon = row
        lat_lon[ar_id] = [lat, lon]

def write_csv(loc, csv):
    with open(csv, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["lat", "lon"])
        for coordinates in loc.values():
            writer.writerow(coordinates)

def tracker(ar_id):
    ar_id = str(ar_id)
    if ar_id in lat_lon:
        lat, lon = lat_lon[ar_id]
        write_csv({ar_id: [lat, lon]}, "live_data.csv")
        return [lat, lon]
    else:
        return [0, 0]

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


cap = cv.VideoCapture(0)
moving_id = 100

while True:
    ret, frame = cap.read()
    corners = []
    ID = []
    centres = []

    for i in ARUCO_DICT.values():
        dictionary = aruco.getPredefinedDictionary(i)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
        marker_corners, markers_ID, reject = detector.detectMarkers(frame)
        if marker_corners is not None and markers_ID is not None:
            corners.append(marker_corners)
            ID.append(markers_ID)
            break

    moving_id_index = None
    min_distance = float('inf')
    closest_marker_id = None

    for mark, current_id in zip(corners, ID):
        for dv in mark:
            for c0, c1, c2, c3 in dv:
                mx = (c0[0] + c1[0] + c2[0] + c3[0]) / 4
                my = (c0[1] + c1[1] + c2[1] + c3[1]) / 4
                c = np.array([mx, my])
                centres.append(c)

                if current_id == moving_id:
                    continue

                if moving_id_index is not None:
                    distance = np.linalg.norm(c - np.array(centres[moving_id_index]))
                    if distance < min_distance:
                        min_distance = distance
                        closest_marker_id = current_id

    if closest_marker_id is not None:
        lat, lon = tracker(closest_marker_id)

    if len(marker_corners) > 0:
        aruco.drawDetectedMarkers(frame, marker_corners, markers_ID, (0, 255, 0))

    for centre in centres:
        cv.circle(frame, (int(centre[0]), int(centre[1])), 5, (255, 0, 255), -1)

    cv.imshow("Frame", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
