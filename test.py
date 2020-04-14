import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
cap = cv2.VideoCapture(0)
ret, _ = cap.read()
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("facial_landmarks.h5")
print("Loaded model from disk")
while ret:
    ret, frame = cap.read()
    frame = cv2.resize(frame,(96,96))
    pre_bu = img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pre_bu=pre_bu/255.
    pre_bu = np.reshape( pre_bu , (1,96,96,1))
    predictions = model.predict(pre_bu)
    predictions = np.reshape( predictions[0 , 0 , 0 ] , ( 15 , 2 ) )*96
    for pred in predictions:
        frame = cv2.circle(frame, (pred[0],pred[1]), 2, (255, 0, 0) , 2) 
    frame = cv2.resize(frame,(96*2,96*2))   
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
