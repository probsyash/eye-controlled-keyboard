import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import joblib
from pynput.keyboard import Controller

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH      = "model_path/hand_landmarker.task"
CLASSIFIER_PATH = "dataStorage/hand_sign_classifier.pkl"
LANDMARKS_PATH  = "dataStorage/landmarks.txt"
STABLE_THRESHOLD = 100   # frames a sign must be held before typing

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index finger
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17),              # palm
]

# ── State ─────────────────────────────────────────────────────────────────────
latest_result   = None
recording       = False
current_label   = None
predicted_letter = None

stable_letter   = None   # letter currently being tracked
stable_count    = 0      # consecutive frames it's been seen
last_typed      = None   # last letter sent to keyboard

# ── Setup ─────────────────────────────────────────────────────────────────────
classifier = joblib.load(CLASSIFIER_PATH)
keyboard   = Controller()
cam        = cv2.VideoCapture(0)

BaseOptions         = mp.tasks.BaseOptions
HandLandmarker      = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult  = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode   = mp.tasks.vision.RunningMode


def on_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=on_result,
    num_hands=2,
)

# ── Main loop ─────────────────────────────────────────────────────────────────
with HandLandmarker.create_from_options(options) as landmarker:
    with open(LANDMARKS_PATH, "a") as file:

        while True:
            # ── Key handling ──────────────────────────────────────────────
            key = cv2.waitKey(1)
            if key == ord('1'):
                recording = not recording
            elif key == ord('2'):
                break
            elif key != -1 and key < 128 and chr(key).isalpha():
                current_label = chr(key)

            # ── Capture frame ─────────────────────────────────────────────
            ret, frame = cam.read()
            if not ret:
                continue

            timestamp = int(time.time() * 1000)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            landmarker.detect_async(mp_image, timestamp)

            # ── Process landmarks ─────────────────────────────────────────
            if latest_result:
                for hand_landmarks in latest_result.hand_landmarks:
                    row = []

                    # Draw landmark dots and build feature row
                    for landmark in hand_landmarks:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                        row.append(landmark.x)
                        row.append(landmark.y)

                    # Record to file if active
                    if recording and current_label is not None:
                        file.write(",".join(str(v) for v in row) + "," + current_label + "\n")

                    # Draw skeleton lines
                    for start_idx, end_idx in HAND_CONNECTIONS:
                        s = hand_landmarks[start_idx]
                        e = hand_landmarks[end_idx]
                        x1 = int(s.x * frame.shape[1]);  y1 = int(s.y * frame.shape[0])
                        x2 = int(e.x * frame.shape[1]);  y2 = int(e.y * frame.shape[0])
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # ── Predict ───────────────────────────────────────────
                    predicted_letter = classifier.predict([row])[0]

                    # ── Debounce / type ───────────────────────────────────
                    if predicted_letter == stable_letter:
                        stable_count += 1
                    else:
                        stable_letter = predicted_letter
                        stable_count  = 1
                        last_typed    = None  # reset last typed when letter changes

                    if stable_count == STABLE_THRESHOLD and stable_letter != last_typed:
                        keyboard.type(stable_letter)
                        last_typed = stable_letter

            # ── HUD ───────────────────────────────────────────────────────
            cv2.putText(frame, f"Label: {current_label} | Recording: {recording}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Prediction: {predicted_letter}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Stable: {stable_letter} ({stable_count}/{STABLE_THRESHOLD})",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.imshow("Hand Sign Recognizer", frame)

# ── Cleanup ───────────────────────────────────────────────────────────────────
cam.release()
cv2.destroyAllWindows()