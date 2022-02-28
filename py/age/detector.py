from pathlib import Path
import cv2
import sys
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
from keras.models import load_model
from keras.preprocessing.image import img_to_array

classifier = load_model('/home/diablo/fr/robots/py/eag/models/emotion_little_vgg_2.h5')
face_classifier = cv2.CascadeClassifier('/home/diablo/fr/robots/py/eag/models/haarcascade_frontalface_default.xml')
pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"

modhash = 'fbe63257a054c1c5466cfd7bf14646d6'
emotion_classes = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}

def face_detector(img):
    gray = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces == ():
        return False ,(0,0,0,0), np.zeros((1,48,48,3), np.uint8), img
    
    allfaces = []   
    rects = []
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi = img[y:y+h, x:x+w]
        allfaces.append(roi)
        rects.append((x,w,y,h))
    return True, rects, allfaces, img

depth = 16
k = 8
weight_file = None
margin = 0.4
image_dir = None

if not weight_file:
    weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="models",file_hash=modhash, cache_dir=Path(sys.argv[0]).resolve().parent)

# load model and weights
img_size = 64
model = WideResNet(img_size, depth=depth, k=k)()
model.load_weights(weight_file)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    ret, rects, faces, image = face_detector(frame)
    preprocessed_faces_ag = []
    preprocessed_faces_emo = []
    
    if ret:
        for (i,face) in enumerate(faces):
            face_ag = cv2.resize(face, (64, 64), interpolation = cv2.INTER_AREA)
            preprocessed_faces_ag.append(face_ag)

            face_gray_emo = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_gray_emo = cv2.resize(face_gray_emo, (48, 48), interpolation = cv2.INTER_AREA)
            face_gray_emo = face_gray_emo.astype("float") / 255.0
            face_gray_emo = img_to_array(face_gray_emo)
            face_gray_emo = np.expand_dims(face_gray_emo, axis=0)
            preprocessed_faces_emo.append(face_gray_emo)
            
        #Age and Gender
        results = model.predict(np.array(preprocessed_faces_ag))
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()

        #Emotion 
        emo_labels = []
        for (i, face) in enumerate(faces):
            preds = classifier.predict(preprocessed_faces_emo[i])[0]
            emo_labels.append(emotion_classes[preds.argmax()])
        
        # draw results
        for (i, face) in enumerate(faces):
            label = "{}, {}, {}".format(int(predicted_ages[i]),"F" if predicted_genders[i][0] > 0.6 else "M",emo_labels[i])
            
        for (i, face) in enumerate(faces):
            label_position = (rects[i][0] + int((rects[i][1]/2)), abs(rects[i][2] - 10))
            cv2.putText(image, label, label_position , cv2.FONT_HERSHEY_PLAIN,1, (0,255,0), 2)

    cv2.imshow("Emotion Detector", image)
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()     
