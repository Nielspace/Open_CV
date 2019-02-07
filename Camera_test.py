import numpy as np 
import cv2

face_casade = cv2.CascadeClassifier('OpenCV-Python-Series/src/cascades/data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while(True):

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = face_casade.detectMultiScale(gray, scaleFactor = 5, minNeighbors=5)

    for (x,y,w,h) in face:
        print (x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_frame = frame[y:y+h, x:x+w]
        img_item = "my-item.png"
        cv2.imwrite(img_item, roi_frame)

        color = (255,0,0)
        stroke = 2
        width = (x+w)
        height = (y+h)
        cv2.rectangle(frame, (x,y), (width,height),color,stroke)


    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()