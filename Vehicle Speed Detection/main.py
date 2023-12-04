import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import time

model=YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('veh2.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0

tracker=Tracker()

cy1=322
cy2=368
offset=4

vh_down={}
vh_up = {}
counter=[]
counter1=[]

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)

 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
             
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]

        if 'car' in c:
            list.append([x1,y1,x2,y2])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame,str(c),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1)
    bbox_id=tracker.update(list)


    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2

        #condition for vehicle going down
        if cy1<(cy+offset) and cy1>(cy-offset):
            vh_down[id] = cy
            vh_down[id] = time.time()
        if id in vh_down:
            if cy2<(cy+offset) and cy2>(cy-offset):
                elapsed_time=time.time() - vh_down[id]
                if counter.count(id)==0:
                    counter.append(id)
                    distance = 10
                    a_speed_ms = distance/elapsed_time
                    a_speed_km = a_speed_ms * 3.6
                    #cv2.circle(frame,(cx,cy),2,(0,0,255),-1)
                    #cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                    cv2.putText(frame,str(int(a_speed_km))+ 'Km/h',(x4,y4), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,255,255),1)
                    print("////////////////")
                    print(' ')
                    print(' ')
                    print("Car id: " + str(id) + " Speed: "+str(int(a_speed_km))+ 'Km/h')
                    print(' ')
                    print(' ')
                    print("////////////////")
                #now some vehicles are being counted twice---
                if id not in counter:
                    counter.append(id)


        #condition for vehicle going up
        if cy2<(cy+offset) and cy2>(cy-offset):
            vh_up[id] = cy
            vh_up[id] = time.time()
        if id in vh_up:
            if cy1<(cy+offset) and cy1>(cy-offset):
                elapsed_time=time.time() - vh_up[id]
                if counter.count(id)==0:
                    counter.append(id)
                    distance = 10
                    a_speed_ms = distance/elapsed_time
                    a_speed_km = a_speed_ms * 3.6
                    cv2.circle(frame,(cx,cy),2,(0,0,255),-1)
                    cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),1)
                    cv2.putText(frame,str(int(a_speed_km))+ 'Km/h',(x4,y4), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,255,255),2)
                    print("////////////////")
                    print(' ')
                    print(' ')
                    print("Car id: " + str(id) + " Speed: "+str(int(a_speed_km))+ 'Km/h')
                    print(' ')
                    print(' ')
                    print("////////////////")
                #now some vehicles are being counted twice---
                if id not in counter1:
                    counter1.append(id)

    cv2.line(frame,(267,cy1),(829,cy1),(255,255,255),1)
    cv2.line(frame,(167,cy2),(932,cy2),(255,255,255),1)
    
    d = len(counter)
    cv2.putText(frame,('Going down: ')+str(d),(450,40),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    d = len(counter1)
    cv2.putText(frame,('Going up: ')+str(d),(450,100),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

time.sleep(0.1)

cap.release()
cv2.destroyAllWindows()

