import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)   # Height

# Adjusted detection confidence for potentially better detection accuracy
detector = HandDetector(detectionCon=0.6, maxHands=2)

def detect_gestures(hands):
    gestures = []
    for hand in hands:
        fingers = detector.fingersUp(hand)
        # Define gestures based on the fingers up pattern
        if fingers == [0, 1, 1, 0, 0]:  # Index and middle finger up
            gestures.append("Peace")
        elif fingers == [0, 1, 0, 0, 1]:  #Rock and Roll baby
            gestures.append("Rock&Roll")
        elif fingers == [1, 1, 1, 1, 1]:  # All fingers up
            gestures.append("Hi")
        elif fingers == [1, 0, 0, 0, 0]:  # Thumb up
            gestures.append("OK")
        elif fingers == [0, 0, 1, 0, 0]: # YoU KNOW WHAT THIS MEANS
            gestures.append("F****U")
        elif fingers == [0, 0, 0, 0, 1]:  # Thumb up
            gestures.append("OK")
        elif fingers == [0, 1, 0, 0, 0]:  # 1
            gestures.append("1")
        elif fingers == [1, 1, 1, 0, 0]:  # 3
            gestures.append("3")
        elif fingers == [1, 1, 0, 0, 0]:  # 2
            gestures.append("2")
        elif fingers == [0, 1, 1, 1, 1]:  # 4
            gestures.append("4")
        
        
        
        
    return gestures

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img, draw=True)
    
    if hands:
        gestures = detect_gestures(hands)
        for gesture in gestures:
            cv2.putText(img, gesture, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)
    
    cv2.imshow("Smart Camera", img)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
