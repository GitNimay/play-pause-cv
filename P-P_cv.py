import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
)

# Open the webcam
cap = cv2.VideoCapture(0)

paused = False  # Keeps track of the current video state (playing/paused)
last_action_time = 0  # Tracks the time of the last gesture action
cooldown_seconds = 1.5  # Cooldown time in seconds

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror-like view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (required for Mediapipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    result = hands.process(rgb_frame)

    # If a hand is detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of thumb and index finger landmarks
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

            # Check for thumbs-up gesture
            if (
                thumb_tip.y < index_mcp.y  # Thumb is above the index finger base
                and abs(thumb_tip.x - thumb_ip.x) < 0.05  # Thumb is straight
                and index_tip.y > index_mcp.y  # Index finger is relaxed (below MCP)
            ):
                current_time = time.time()  # Get the current time

                # Perform the action only if cooldown period has passed
                if current_time - last_action_time > cooldown_seconds:
                    if not paused:  # Pause video if not already paused
                        pyautogui.press("space")
                        paused = True
                        print("Video Paused")
                    else:  # Play video if already paused
                        pyautogui.press("space")
                        paused = False
                        print("Video Playing")

                    # Update the last action time
                    last_action_time = current_time
            break

    # Display the webcam feed
    cv2.imshow("Thumbs-Up Gesture Control", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
