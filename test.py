import os
import cv2
import numpy as np 
from PIL import Image 
import pickle


face_casade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

image_dir = os.path.join(BASE_DIR, "image")
recog = cv2.face.LBPHFaceRecognizer_create()

x_train = []
y_labels = []
current_id = 0
label_id = {}


for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg') or file.endswith('jpeg'):

            path = os.path.join(root,file)
            labels = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            #print(labels, path)
            if not labels in label_id:
                label_id[labels] = current_id
                current_id +=1
            id_ = label_id[labels]
            pil_image = Image.open(path).convert('L')
            image_array = np.array(pil_image, "uint8")
            #print(image_array)
            face = face_casade.detectMultiScale(image_array)
            for (x,y,w,h) in face:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

#print(x_train)
#print(y_labels)

with open('labels.pickel', 'wb') as f:
    pickle.dump(label_id, f)

recog.train(x_train, np.array(y_labels))
recog.save("trainer.yml")