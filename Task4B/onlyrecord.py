import cv2 as cv

cap = cv.VideoCapture(2)

# Get the width and height of the original video
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Set the desired width and calculate the corresponding height to maintain the aspect ratio
desired_width = 1400  # Change this value to your preferred width
desired_height = int((desired_width / width) * height)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to fit the screen
    frame = cv.resize(frame, (desired_width, desired_height))
    frame=cv.flip(frame,-1)
    cv.imshow("Video", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the video
        break

cap.release()
cv.destroyAllWindows()
