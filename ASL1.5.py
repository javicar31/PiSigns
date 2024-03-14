#####################
# Author:Javicar31  #
# Created: Feb2024  #
#####################

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

detector = HandDetector(detectionCon=0.7, maxHands=1)

def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def detect_asl_letters(hand, img):
    landmarks = hand['lmList']
    # Ensure there are enough landmarks for calculations
    if len(landmarks) < 21:  # Check for sufficient landmarks
        return ""

    fingers = detector.fingersUp(hand)
    thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip = landmarks[4][:2], landmarks[8][:2], landmarks[12][:2], landmarks[16][:2], landmarks[20][:2]
    asl_letter = ""
    
    # S: Fist with thumb on the side
    if fingers == [0, 0, 0, 0, 0] and thumb_tip[0] < index_tip[0]:
        asl_letter = "S"
    
    # B: All fingers up and together, thumb across palm
    
    # C: All fingers not fully extended and slightly curved inwards to form a 'C' shape
    if fingers == [0, 1, 1, 1, 1]:
        base_of_palm = landmarks[0][:2]  # Typically the base of the palm
        tips = [index_tip, middle_tip, ring_tip, pinky_tip]
        extended_fingers = [fingers[i] for i in range(1, 5)]
        extended_tips = [tips[i] for i in range(4) if extended_fingers[i] == 1]

        # Calculate average distance from the tips of extended fingers to the base of the palm
        average_distance = sum([calculate_distance(tip, base_of_palm) for tip in extended_tips]) / len(extended_tips)

        if average_distance < 100:  # Threshold for recognizing 'C'
            asl_letter = "C"
        else:
            asl_letter = "B"
    if fingers == [1, 1, 1, 1, 1]:
         asl_letter = "C"

    # D: Index finger up, other fingers curled into the palm, thumb touching middle finger
    thumb_middle_distance = calculate_distance(thumb_tip, middle_tip)
    if fingers == [0, 1, 0, 0, 0] and thumb_middle_distance < 2:  # Adjust threshold as needed
        asl_letter = "G"
    
    # F: Index finger and thumb touching, other fingers up
    if fingers == [0, 0, 1, 1, 1]:
        asl_letter = "F"
    
    # G: Index finger and thumb parallel, other fingers curled
    if fingers == [0, 1, 0, 0, 0] and index_tip[1] < thumb_tip[1]:
        asl_letter = "D"
    
    # H: Index and middle fingers parallel, other fingers curled
    if fingers == [0, 1, 1, 0, 0] and index_tip[1] < middle_tip[1]:
        asl_letter = "H"
    
    # I: Pinky up, other fingers curled, thumb across palm
    if fingers == [0, 0, 0, 0, 1]:
        asl_letter = "I"
    
    # K: Middle and index finger up forming a V, thumb out
    if fingers == [1, 1, 1, 0, 0]: 
        asl_letter = "K"
    
    # L: Index finger and thumb up making an L shape
    if fingers == [1, 1, 0, 0, 0]:
        asl_letter = "L"

    # A: Fist with thumb in front of fingers
    if fingers == [0, 0, 0, 0, 0] and thumb_tip[0] > index_tip[0]:
        asl_letter = "A"
        
    if fingers == [0, 1, 1, 0, 0]:
        asl_letter = "U"

    # W: Index, middle, and ring fingers up, thumb tucked in
    if fingers == [0, 1, 1, 1, 0]:
        asl_letter = "W"
    
    # Y: Thumb and pinky up, other fingers curled
    if fingers == [1, 0, 0, 0, 1]:
        asl_letter = "Y"
    

    return asl_letter

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)

    if hands:
        for hand in hands:
            asl_letter = detect_asl_letters(hand, img)
            if asl_letter:
                # Display the detected letter on the image
                cv2.putText(img, asl_letter, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)

    cv2.imshow("ASL Letter Detector", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): #press "q" on keyboard to close 
        break

cap.release()
cv2.destroyAllWindows()
