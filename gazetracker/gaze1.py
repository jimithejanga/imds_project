import cv2
import numpy as np

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize camera
cap = cv2.VideoCapture(0)

def detect_pupil(eye_roi):
    # Convert to grayscale and apply adaptive threshold
    gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    thresh_eye = cv2.adaptiveThreshold(gray_eye, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pupil = None

    # Identify the largest contour as the pupil
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(max_contour)
        pupil = (x + w // 2, y + h // 2)  # Pupil center
    return pupil, thresh_eye

def classify_gaze(pupil, eye_width, eye_height):
    if pupil is None:
        return "No pupil detected"
    
    x_ratio = pupil[0] / eye_width
    y_ratio = pupil[1] / eye_height

    if x_ratio < 0.33:
        gaze = "Looking Left"
    elif x_ratio > 0.66:
        gaze = "Looking Right"
    else:
        gaze = "Looking Center"

    if y_ratio < 0.33:
        gaze += ", Looking Up"
    elif y_ratio > 0.66:
        gaze += ", Looking Down"

    return gaze

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Region of interest for eyes within the face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # Crop the eye region
            eye_roi = roi_color[ey:ey + eh, ex:ex + ew]

            # Detect pupil
            pupil, thresh_eye = detect_pupil(eye_roi)

            # Draw the eye region and pupil
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            if pupil:
                cv2.circle(eye_roi, pupil, 3, (0, 0, 255), -1)

                # Classify gaze direction
                gaze_direction = classify_gaze(pupil, ew, eh)
                print(gaze_direction)
                # Display gaze direction on the frame
                cv2.putText(frame, gaze_direction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display the thresholded eye for debugging
            cv2.imshow("Thresholded Eye", thresh_eye)

    cv2.imshow('Gaze Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# this used haskell  very poor needs a lot of lighting or maybe a good camera#
