import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

recording = False
current_label = None

# Connections for drawing hand landmarks
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index finger
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17)               # palm
]

# Open the default camera
cam = cv2.VideoCapture(0)

# Initialize the MediaPipe gesture recognizer
model_path = 'model_path/hand_landmarker.task'

# Set up the MediaPipe gesture recognizer
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='model_path/hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result, num_hands=2)

with HandLandmarker.create_from_options(options) as landmarker:
    latest_result = None
    file = open("dataStorage/landmarks.txt", "a")

    while True:
        key = cv2.waitKey(1)

        if key == ord('r'):
            recording = not recording
        elif key == ord('q'):
            break
        elif key != -1 and key < 128 and chr(key).isalpha():
            current_label = chr(key)

        timestamp = int(time.time() * 1000)
        ret, frame = cam.read()

        if not ret:
            continue

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, timestamp)

        if latest_result:
            # Landmark drawer
            for hand_landmarks in latest_result.hand_landmarks:
                coords = []
                # Draw circles for each landmark
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    if recording and current_label is not None:
                        coords.append((landmark.x, landmark.y))

                # Write all coords for this frame as one row
                if coords:
                    for i in range(len(coords)):
                        if i == len(coords) - 1:
                            file.write(str(coords[i][0]) + "," + str(coords[i][1]) + "," + current_label + "\n")
                        else:
                            file.write(str(coords[i][0]) + "," + str(coords[i][1]) + ",")

                # Draw lines between connected landmarks
                for start_index, end_index in HAND_CONNECTIONS:
                    start_landmark = hand_landmarks[start_index]
                    end_landmark = hand_landmarks[end_index]
                    x1 = int(start_landmark.x * frame.shape[1])
                    y1 = int(start_landmark.y * frame.shape[0])
                    x2 = int(end_landmark.x * frame.shape[1])
                    y2 = int(end_landmark.y * frame.shape[0])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Display label and recording status on screen
        cv2.putText(frame, f"Label: {current_label} | Recording: {recording}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the captured frame
        cv2.imshow('Camera', frame)

    # Release the capture
    cam.release()
    file.close()