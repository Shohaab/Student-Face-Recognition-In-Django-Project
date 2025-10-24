# student/face_utilis.py

import cv2
import numpy as np
import os
from ultralytics import YOLO
from keras_facenet import FaceNet
from numpy.linalg import norm

# Initialize models once (load on startup)
yolo_model = YOLO(r"E:\Python\AI_face_recognition\models\yolov8n-face.pt")
embedder = FaceNet()

# Load existing embeddings
EMB_PATH = r"E:\Python\AI_face_recognition\embeddings\embeddings.npy"
NAMES_PATH = r"E:\Python\AI_face_recognition\embeddings\names.npy"

if os.path.exists(EMB_PATH) and os.path.exists(NAMES_PATH):
    known_embeddings = np.load(EMB_PATH, allow_pickle=True)
    known_names = np.load(NAMES_PATH, allow_pickle=True)
    print(f"✅ Loaded {len(known_embeddings)} embeddings from disk.")
else:
    known_embeddings, known_names = [], []
    print("⚠️ No embeddings found initially.")


def preprocess_face(face):
    """Preprocess face image before embedding."""
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (160, 160))
    face_resized = face_resized.astype("float32") / 255.0
    face_array = np.expand_dims(face_resized, axis=0)
    return face_array

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))







def detect_and_embed(frame):
    results = yolo_model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        return ("No face detected", "unknown")

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (160, 160))
        face_array = np.expand_dims(face_resized, axis=0)

        # 👉 Get embedding and normalize it
        embedding = embedder.embeddings(face_array)[0]
        embedding = l2_normalize(embedding)

        if len(known_embeddings) == 0:
            return ("Database empty", "unknown")

        # 👉 Normalize all known embeddings
        normalized_known = [l2_normalize(e) for e in known_embeddings]

        # 👉 Compute cosine distance (better than Euclidean)
        distances = [1 - np.dot(embedding, e) for e in normalized_known]
        min_dist = min(distances)
        idx = np.argmin(distances)

        threshold = 0.7  # ✅ tuned for FaceNet cosine distance
        if min_dist < threshold:
            matched_name = known_names[idx]
            print(f"[MATCH] {matched_name} (distance={min_dist:.3f})")
            return (matched_name, "known")
        else:
            print(f"[NO MATCH] Closest: {known_names[idx]} (distance={min_dist:.3f})")
            return ("Unknown face", "unknown")



def reload_embedddings():
    """Reload embeddings after new student added."""
    global known_embeddings, known_names
    if os.path.exists(EMB_PATH) and os.path.exists(NAMES_PATH):
        known_embeddings = np.load(EMB_PATH, allow_pickle=True)
        known_names = np.load(NAMES_PATH, allow_pickle=True)
        print(f"🔄 Reloaded {len(known_embeddings)} embeddings.")
    else:
        known_embeddings, known_names = [], []
        print("⚠️ No embeddings found on reload.")





"""import cv2, numpy as np, os
from ultralytics import YOLO
from keras_facenet import FaceNet
from numpy.linalg import norm

# Initialize models once (load on startup)
yolo_model = YOLO(r"E:\Python\AI_face_recognition\models\yolov8n-face.pt")
embedder = FaceNet()

# Load existing embeddings
EMB_PATH = r"E:\Python\AI_face_recognition\embeddings\embeddings.npy"
NAMES_PATH = r"E:\Python\AI_face_recognition\embeddings\names.npy"

if os.path.exists(EMB_PATH):
    known_embeddings = np.load(EMB_PATH)
    known_names = np.load(NAMES_PATH)
else:
    known_embeddings = []
    known_names = []
def detect_and_embed(frame):
    results = yolo_model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        return ("No face detected", "unknown")

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (160, 160))
        face_array = np.expand_dims(face_resized, axis=0)

        embedding = embedder.embeddings(face_array)[0]

        if len(known_embeddings) == 0:
            return ("Database empty", "unknown")

        distances = [norm(embedding - e) for e in known_embeddings]
        min_dist = min(distances)
        idx = np.argmin(distances)
        threshold = 1.0
        if min_dist < threshold:
            return (known_names[idx], "known")
        else:
            return ("Unknown face", "unknown")



def reload_embedddings():
    global known_embeddings,known_names
    if os.path.exists(EMB_PATH):
        known_embeddings=np.load(EMB_PATH)
        known_names=np.load(NAMES_PATH)
        print(f"Reload Embeddings: {len(known_embeddings)}known_names.")
    else:
        known_embeddings,known_names=[],[]
        print("No EMbeddings Found")"""
