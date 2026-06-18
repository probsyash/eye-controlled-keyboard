# ASL Hand Sign Recogniser

Real-time American Sign Language (ASL) finger-spelling recognition using MediaPipe hand tracking and a Random Forest classifier. Detected signs are typed directly into any active application via simulated keypresses.

## Demo
![Demo](/gif_IGNORE/demo.gif)

## How it works
1. MediaPipe detects 21 hand landmarks per frame via webcam
2. The (x, y) coordinates of each landmark are fed into a trained Random Forest classifier
3. Once a sign is held consistently for a set number of frames, the predicted letter is typed into the OS via `pynput`

## Results
- **99.8%** accuracy on held-out test split from training session
- **83%** accuracy on an independent held-out recording session, confirming generalisation beyond training data
- Lowest performing signs: v, r, k - likely due to visual similarity with neighbouring signs (v/u, r/u)

## Stack
- Python, OpenCV, MediaPipe, scikit-learn, pynput

## Files
| File | Purpose |
|------|---------|
| `openWebcam.py` | Live webcam inference + keyboard typing |
| `train.py` | Trains and evaluates the Random Forest classifier |
| `landmarksTester.py` | Evaluates classifier on independent held-out session |
| `dataStorage/landmarks.txt` | Training data (hand landmark coordinates + labels) |
| `dataStorage/landmarks_test.txt` | Independent held-out session data |
| `dataStorage/hand_sign_classifier.pkl` | Trained Random Forest model |
| `dataStorage/metrics.txt` | Held-out session evaluation metrics |

## Usage

**Recording training data:**

Run `main.py`, press a letter key to set the label, press `1` to toggle recording on/off, press `2` to quit.

**Training:**
```bash
python train.py
```

**Running:**
```bash
python openWebcam.py
```

Click into any text field, then sign letters — hold a sign steady for ~3 seconds to type it.

## Limitations & future work
- Trained on a single person's hands — performance may vary across users
- No support for J and Z (require motion)
- Planned: word/sentence builder with space and backspace gestures