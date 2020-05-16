import numpy as np
import cv2

img = cv2.imread("faces.jpeg",1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_path = "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(face_path)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(40,40), maxSize=(100,100))

for (x, y, w, h) in faces:
	cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

eye_path = "haarcascade_eye.xml"

eye_cascade = cv2.CascadeClassifier(eye_path)

eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=20, minSize=(10,10), maxSize=(30,30))

for (x, y, w, h) in eyes:
	cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 2)

# smile_path = "haarcascade_smile.xml"

# smile_cascade = cv2.CascadeClassifier(smile_path)

# smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.04, minNeighbors=20, minSize=(5,5), maxSize=(70,70))
# print(len(smiles))

# for (x, y, w, h) in smiles:
# 	cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()