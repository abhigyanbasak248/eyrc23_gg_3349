import cv2
cap = cv2.VideoCapture(2)

#460, 555
while True:
    ret, frame = cap.read()
    frame = frame[:, 100:600]
    frame = cv2.resize(frame, (960, 985))
    frame = cv2.circle(frame, (240, 850), 5, (0,0,255), -1)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()