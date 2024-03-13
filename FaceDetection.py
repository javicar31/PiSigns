import cv2
cap = cv2.VideoCapture(0)

import cvzone
from cvzone import FaceDetectionModule
face_detector = FaceDetectionModule.FaceDetector()

while True:
    success, img = cap.read()
    img, list_faces = face_detector.findFaces(img)
    cv2.imshow("Face Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break