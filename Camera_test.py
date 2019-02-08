import numpy as np 
import cv2
import pickle

face_casade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recog = cv2.face.LBPHFaceRecognizer_create()
recog.read('trainer.yml')

labels={"person_name":1}
with open('labels.pickel', 'rb') as f:
    og_labels=pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = face_casade.detectMultiScale(gray, scaleFactor = 1.8, minNeighbors=5)

    for (x,y,w,h) in face:
        #print (x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_frame = frame[y:y+h, x:x+w]

        id_, conf = recog.predict(roi_gray)
        if conf >= 45 and conf<=85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stoke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)


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