# Video Classification – Cash Lifting vs Normal Activity

## Overview

This project is designed to classify video clips into two categories:

1. **Cash Lifting** – Suspicious or theft-related activity.
2. **Normal Activity** – Regular, non-suspicious actions.

The workflow handles **video preprocessing, model training, and prediction** from test videos. It supports `.dav` file conversion to `.mp4` for processing and uses a lightweight deep learning approach to achieve acceptable accuracy even with limited data.

---

## Folder Structure

```
/content/cashlifting              # Folder containing Cash Lifting videos (.dav or .mp4)
/content/drive/MyDrive/clips/train/normal  # Folder containing Normal Activity videos (.mp4)
```

---

## Steps

### 1. Convert `.dav` Files to `.mp4`

If your dataset contains `.dav` files (common from CCTV systems), convert them to `.mp4`:

```python
!pip install ffmpeg-python
import ffmpeg, os

input_folder = "/content/cashlifting"
for file in os.listdir(input_folder):
    if file.lower().endswith(".dav"):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(input_folder, file.replace(".dav", ".mp4"))
        ffmpeg.input(input_path).output(output_path).run()
```

---

### 2. Preprocess Videos

Videos are resized to `224x224` and set to `15 fps` for uniformity:

```python
!pip install moviepy
from moviepy.editor import VideoFileClip
import os

def preprocess_video(input_path, output_path, target_size=(224, 224), fps=15):
    clip = VideoFileClip(input_path).resize(newsize=target_size).set_fps(fps)
    clip.write_videofile(output_path, codec="libx264", audio=False)

for folder in ["/content/cashlifting", "/content/drive/MyDrive/clips/train/normal"]:
    for file in os.listdir(folder):
        if file.endswith(".mp4"):
            preprocess_video(os.path.join(folder, file), os.path.join(folder, f"processed_{file}"))
```

---

### 3. Extract Frames and Prepare Dataset

Each video is split into frames, and frames are saved for training:

```python
import cv2
import numpy as np

def extract_frames(video_path, label, frame_limit=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while count < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
        count += 1
    cap.release()
    return np.array(frames), [label] * len(frames)
```

---

### 4. Train the Model

A simple CNN model is used for training:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

X_train, y_train = [], []
# Append extracted frames and labels here from both classes
# X_train.append(frames), y_train.append(labels)

X_train = np.array(X_train) / 255.0
y_train = to_categorical(y_train, num_classes=2)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)
model.save("video_classifier.h5")
```

---

### 5. Predict from Test Video (First Frame)

This step:

1. Reads the first frame of a test video.
2. Preprocesses it the same way as the training frames.
3. Loads the trained model.
4. Predicts the class and confidence.

```python
import cv2
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("video_classifier.h5")

def predict_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to read video.")
        return
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype("float32") / 255.0
    frame = np.expand_dims(frame, axis=0)
    prediction = model.predict(frame)
    class_id = np.argmax(prediction)
    confidence = prediction[0][class_id]
    classes = ["Normal", "Cash Lifting"]
    print(f"Predicted Class: {classes[class_id]}")
    print(f"Confidence: {confidence * 100:.2f}%")

predict_first_frame("/content/test_video.mp4")
```

---

## How It Works

1. **Data Preparation** – Convert videos to `.mp4`, resize, and set a uniform frame rate.
2. **Frame Extraction** – Capture a limited number of frames per video to reduce computational load.
3. **Model Training** – Train a CNN model on extracted frames from both classes.
4. **Prediction** – Use the first frame from a test video to make a quick classification.

---

## Advantages

* Works with `.dav` and `.mp4` formats.
* Lightweight and quick to train.
* Requires minimal preprocessing after conversion.
* Can achieve reasonable accuracy (\~75% or more) with enough balanced training data.

---

## Limitations

* **Small dataset** (especially for Cash Lifting) can cause overfitting and bias towards the majority class.
* Using **only the first frame** for prediction ignores temporal information and can lead to incorrect results if the activity is not visible in that frame.
* Performance is heavily dependent on video quality and camera angle.
* The CNN is basic and may not generalize well to new environments.
* Not robust to lighting changes, occlusion, or unusual motion patterns.
* For higher accuracy, a 3D CNN or sequence model (like LSTM) should be used.
