import cv2
from cvzone.HandTrackingModule import HandDetector
import serial
import time

# Setup serial connection
# Make sure to replace '/dev/ttyACM0' with the correct port if needed
arduino = serial.Serial('/dev/ttyACM0', 9600)
time.sleep(2)  # Give the connection a second to settle

# Initialize the camera and set properties
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Initialize the hand detector
detector = HandDetector(detectionCon=0.6, maxHands=1)

while True:
    success, img = cap.read()
    if success:
        hands, img = detector.findHands(img)
        if hands:
            # Assuming detector.findHands() returns list of detected hands with landmarks
            hand = hands[0]  # Take the first detected hand
            fingers = detector.fingersUp(hand)
            if fingers == [1, 1, 1, 1, 1]:  # All fingers up
                arduino.write(b'1')  # Send '1' to turn LED on
            elif fingers == [0, 0, 0, 0, 0]:  # No finger up
                arduino.write(b'0')  # Send '0' to turn LED off
        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
arduino.close()
