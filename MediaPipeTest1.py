import cv2
import mediapipe as mp
import os
import time

# Create a directory to store hand frames
output_directory = "Hand_Frames"
os.makedirs(output_directory, exist_ok=True)

# Initialize MediaPipe hand tracking
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize video capture (replace 0 with your camera index or video file path)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Process the frame with MediaPipe hand tracking
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # If hands are detected, you can get the landmarks and draw them on the frame
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        # Save the frame as an image with a unique filename
        timestamp = int(time.time())  # Use timestamp to make filenames unique
        filename = f"{output_directory}/hand_frame_{timestamp}.png"
        cv2.imwrite(filename, frame)

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
