from cvzone.PoseModule import PoseDetector
import cv2

# Initialize the webcam to the default camera (index 0)
cap = cv2.VideoCapture(0)

# Initialize the PoseDetector class. Here, we're using default parameters. For a deep dive into what each parameter signifies, consider checking the documentation.
detector = PoseDetector(staticMode=False,
                        modelComplexity=1,
                        smoothLandmarks=True,
                        enableSegmentation=False,
                        smoothSegmentation=True,
                        detectionCon=0.5,
                        trackCon=0.5)

# Loop to continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    success, img = cap.read()

    # Detect human pose in the frame
    img = detector.findPose(img)

    # Extract body landmarks and possibly a bounding box 
    # Set draw=True to visualize landmarks and bounding box on the image
    lmList, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=False)

    # If body landmarks are detected
    if lmList:
        # Extract the center of the bounding box around the detected pose
        center = bboxInfo["center"]

        # Visualize the center of the bounding box
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

        # Calculate the distance between landmarks 11 and 15 and visualize it
        length, img, info = detector.findDistance(lmList[11][0:2],
                                                  lmList[15][0:2],
                                                  img=img,
                                                  color=(255, 0, 0),
                                                  scale=10)

        # Calculate and visualize the angle formed by landmarks 11, 13, and 15
        # This can be used as an illustrative example of how posture might be inferred from body landmarks.
        angle, img = detector.findAngle(lmList[11][0:2],
                                        lmList[13][0:2],
                                        lmList[15][0:2],
                                        img=img,
                                        color=(0, 0, 255),
                                        scale=10)

        # Check if the calculated angle is close to a reference angle of 50 degrees (with a leeway of 10 degrees)
        isCloseAngle50 = detector.angleCheck(myAngle=angle,
                                             targetAngle=50,
                                             offset=10)

        # Print the result of the angle comparison
        print(isCloseAngle50)

    # Display the processed frame
    cv2.imshow("Image", img)

    # Introduce a brief pause of 1 millisecond between frames
    cv2.waitKey(1)
