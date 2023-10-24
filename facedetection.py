import cv2
import numpy as np
import time

# Face classifier
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Open the camera
videoCam = cv2.VideoCapture(0)

if not videoCam.isOpened():
    print("Camera not detected")
    exit()

# Q button pressed
q_button_pressed = False
while (q_button_pressed == False):
    ret, frame = videoCam.read()

    if ret == True:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=2)

        for (x, y, w, h) in detected_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        face_count_text = "Number of Detected Faces = " + str(len(detected_faces))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, face_count_text, (0, 30), font, 1, (255, 0, 0), 1)

        cv2.imshow("Result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            q_button_pressed = True
            break

videoCam.release()
cv2.destroyAllWindows()
