import cv2
import torch
import time
from ultralytics import YOLO
import json
model = YOLO('/home/shagnik/Downloads/best.pt')
cap = cv2.VideoCapture(2)

x = 0
start = time.time()

labels=["blank","combat","destroyed_buildings","fire","human_aid_rehabilitation","military_vehicles"]
mid_y=[]
ca=[]
i=1
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = frame[:, 100:600]
    results = model(frame)
    for r in results:
        for ra in r.boxes.data:
            # Extracting bounding box coordinates
            x_min, y_min, x_max, y_max = map(int, ra[:4])
            c=int(ra[5])
            # Setting the color (BGR format, e.g., (0, 255, 0) for green)
            color = (0, 255, 0)  # You can change this color as per your preference
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame,labels[c],(x_min,y_min-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2,3)
            mid_y.append((y_min+y_max)/2)
            ca.append(c)
        if True:
            break
    if len(mid_y)>0 and i>0:
        g=list(zip(mid_y,ca))
        g=sorted(g,key=lambda x:x[0])
        s_y,s_c=zip(*g)
        print(s_c)
        s_c=list(s_c)
        n=[]
        for j in range(0,5):
            n.append(labels[s_c[j]])
        print(n)
        ele_dict=dict(zip(['E','D','C','B','A'],n))
        # new_dict = {}
        if(len(s_y))==5:
            # for key, value in zip(ele_dict.keys(), ele_dict.values()):
            #     if (value != "blank"):
            #         new_dict[key] = value
            # print(new_dict)
            print(ele_dict)
            file_path="file.txt"
            with open(file_path,"w") as file:
                json.dump(ele_dict,file)
        i=-1
    frame= cv2.resize(frame, (960, 985))
    cv2.imshow('frames', frame)

    x += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
