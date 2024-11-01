import cv2
import mediapipe as mp
from fer import FER
import matplotlib.pyplot as plt
from collections import Counter

# Initialize MediaPipe Face Mesh and FER Emotion Detector
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
emotion_detector = FER()

# Initialize emotion counter
emotion_counts = Counter()

# Path to the uploaded video file
video_path = 'Men.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Parameters
frame_interval = 10  # Process every 10th frame to save time
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process only every nth frame (adjust to control processing speed)
    if frame_count % frame_interval == 0:
        # Convert the frame to RGB (required for MediaPipe and FER)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Emotion detection
        emotion_analysis = emotion_detector.top_emotion(rgb_frame)
        if emotion_analysis is not None and emotion_analysis[0] is not None:
            emotion, score = emotion_analysis
            emotion_counts[emotion] += 1
            print(f"Detected emotion: {emotion} with score: {score}")
        else:
            print(f"No detectable emotion in frame {frame_count}")

    frame_count += 1

cap.release()

# Check if any emotions were detected
if emotion_counts:
    # Generate Statistical Report
    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(emotions, counts, color="skyblue")
    plt.xlabel("Emotions")
    plt.ylabel("Frequency")
    plt.title("Emotion Frequency in Video")
    plt.show()
else:
    print("No emotions were detected in the video.")
