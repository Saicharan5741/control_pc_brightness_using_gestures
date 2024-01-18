import cv2
import mediapipe as mp
import math

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Open the webcam
cap = cv2.VideoCapture(0)

# Initialize brightness level
brightness = 50

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark

        # Get the coordinates of thumb and index finger
        thumb = (int(landmarks[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1]),
                 int(landmarks[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0]))
        index_finger = (int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]),
                        int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0]))

        # Calculate distance between thumb and index finger
        distance = calculate_distance(thumb, index_finger)

        # Adjust brightness based on distance
        brightness = int((distance / frame.shape[1]) * 100)

        # Draw skeleton lines
        for connection in mp_hands.HAND_CONNECTIONS:
            connection_start = tuple(
                [int(landmarks[connection[0]].x * frame.shape[1]), int(landmarks[connection[0]].y * frame.shape[0])])
            connection_end = tuple(
                [int(landmarks[connection[1]].x * frame.shape[1]), int(landmarks[connection[1]].y * frame.shape[0])])

            cv2.line(frame, connection_start, connection_end, (0, 255, 0), 2)

    # Display the frame with adjusted brightness and skeleton
    frame = cv2.convertScaleAbs(frame, alpha=brightness / 100.0, beta=0)
    cv2.putText(frame, f'Brightness: {brightness}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Brightness Control', frame)

    # Break the loop when 'ESC' is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
