import cv2 as cv
from cv2 import aruco
import numpy as np
import csv

def read_csv(csv_name):
    lat_lon = {}

    with open(csv_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            ar_id, lat, lon = row
            # Convert ArUco ID to string
            ar_id = str(ar_id)
            lat_lon[ar_id] = [lat, lon]

    return lat_lon

def write_csv(loc, csv_name):
    with open(csv_name, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["lat", "lon"])
        for coordinates in loc.values():
            writer.writerow(coordinates)

def tracker(ar_id, lat_lon):
    ar_id = str(ar_id)

    if ar_id in lat_lon:
        lat, lon = lat_lon[ar_id]
        write_csv({ar_id: [lat, lon]}, "live_data.csv")
        return [lat, lon]
    else:
        print(f"ArUco ID {ar_id} not found in lat_lon dictionary.")
        print("Available ArUco IDs:", lat_lon.keys())
        return [0, 0]

# Loading all pre-built libraries
ARUCO_DICT = {
    "DICT_4X4_50": cv.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
    "DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11
}

cap = cv.VideoCapture(2)

moving_id = 100

map_ids = [23,24,22,49,50,51,52,53,54,48,47,46,45,44,43,10,8,12,9,11,13,14,15,16,17,18,19,20,21,25,26,27,28,29,34,33,32,31,30,42,41,40,39,35,38,37,36]
stored_centres = {}

lat_long = read_csv('lat_long.csv')

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

desired_width = 945  # Change this value to your preferred width
desired_height = int((desired_width / width) * height)

while True:
    ret, frame = cap.read()

    frame = cv.resize(frame, (960, 985))

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
   
    all_corners = []
    all_ID = []
    all_centres = []
    center_id_100 = np.array([0.0, 0.0])

    for dict_name, dict_value in ARUCO_DICT.items():
        dictionary = aruco.getPredefinedDictionary(dict_value)
        parameters = aruco.DetectorParameters()
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        parameters.cornerRefinementWinSize = 5
        detector = aruco.ArucoDetector(dictionary, parameters)
        marker_corners, markers_ID, _ = detector.detectMarkers(gray)

        if marker_corners is not None and markers_ID is not None:
            all_corners.extend(marker_corners)
            all_ID.extend(markers_ID.flatten())

            for idx, id_value in enumerate(markers_ID.flatten()):
                if id_value in map_ids and len(marker_corners) != len(map_ids):
                    # area = cv.contourArea(marker_corners[idx][0])
                    # if 500 < area < 5000:  # Adjust the area range based on your marker size
                    center = np.mean(marker_corners[idx][0], axis=0)
                    stored_centres[id_value] = center

                if id_value == moving_id:
                    center_id_100 = np.mean(marker_corners[idx][0], axis=0)
                    break

        min_distance_id = None
        min_distance = float('inf')

        for stored_id, stored_center in stored_centres.items():
            distance = np.linalg.norm(np.array(center_id_100) - np.array(stored_center))
            if distance < min_distance and stored_id != moving_id and distance < 100:  # Adjust the distance threshold
                min_distance = distance
                min_distance_id = stored_id

        if min_distance_id is not None:
            print(f"Minimum distance ID: {min_distance_id} {min_distance}")
            printll = tracker(min_distance_id, lat_long)
            print(printll)

    cv.imshow("Frame", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# cap.release()
cv.destroyAllWindows()
