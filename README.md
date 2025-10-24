🧠 AI-Based Real-Time Student Face Recognition & Attendance System

A Django + Deep Learning-based intelligent system for real-time student identification, attendance tracking, and automatic embedding generation using YOLO and FaceNet.

📘 Overview

This project is an AI-powered face recognition and attendance management system designed for schools and universities.
It automatically detects student faces in real time, compares them with stored embeddings, and recognizes each student with high accuracy.

The system is integrated with a Django web dashboard for administrators to manage student data — including adding, editing, deleting, and re-capturing images.
Whenever a new student is registered or an image is updated, the system automatically regenerates embeddings (.npy files) to keep recognition data up to date.

🚀 Key Features

✅ Real-Time Face Recognition – Detect and identify students using webcam video feed.
✅ YOLOv8 + FaceNet Integration – YOLO detects faces, FaceNet generates 128-D embeddings for recognition.
✅ Auto Embedding Update – Automatically updates .npy embedding files whenever student data changes.
✅ Accurate Recognition – Uses Euclidean distance between embeddings with an adaptive threshold for precision.
✅ Admin Dashboard (Django) – Manage students, update records, delete data, and view registered students.
✅ Recapture Option – Admin can re-capture student images for better accuracy.
✅ Search and Pagination – Search students by name, father name, or surname with pagination for large data.
✅ Media File Management – Automatically stores student images in MEDIA_ROOT and updates recognition data.
✅ Optimized Warm-Up – FaceNet and YOLO are warmed up at server start for faster and more accurate recognition.

🧩 Tech Stack
Component	Technology
Frontend	HTML5, CSS3, Bootstrap
Backend	Django (Python 3.13)
Database	SQLite / PostgreSQL
AI Models	YOLOv8 (Face Detection), FaceNet (Feature Extraction)
Libraries	OpenCV, NumPy, TensorFlow / Keras, Ultralytics YOLO, keras-facenet
Environment	Virtualenv
Hardware (Optional)	NVIDIA GPU for faster processing
⚙️ Project Structure
AI_face_recognition/
│
├── face_recognize/                # Django Project Folder
│   ├── student/                   # App handling student operations
│   │   ├── templates/student/     # HTML Templates
│   │   ├── static/                # CSS, JS, images
│   │   ├── models.py              # Student model
│   │   ├── views.py               # View logic (CRUD + recognition)
│   │   ├── urls.py                # URL patterns
│   │   ├── face_utils.py          # YOLO + FaceNet logic
│   │   ├── manage_embeddings.py   # Embedding generation & update
│   │   ├── signals.py             # Auto-embedding update on save/delete
│   ├── settings.py                # Configs (MEDIA, STATIC paths)
│   ├── urls.py                    # Root URL config
│
├── embeddings/                    # Stores .npy files (embeddings & names)
│   ├── embeddings.npy
│   ├── names.npy
│
├── models/                        # YOLO & FaceNet models
│   ├── yolov8n-face.pt
│
├── media/                         # Uploaded student images
│   ├── student_images/
│
└── requirements.txt               # All dependencies

🧠 How It Works

Face Detection:
YOLOv8 detects the face region from the input frame or webcam feed.

Feature Extraction (Embeddings):
FaceNet converts the detected face into a 128-dimensional embedding vector.

Comparison:
The embedding is compared with all stored embeddings using Euclidean distance.
If the distance is below a threshold, the face is recognized.

Database Update:
When new students are added or updated, a signal automatically triggers embedding regeneration.

Recognition Output:
The system identifies known faces in real-time and can log attendance or trigger actions (like notifications).

⚡ Installation Guide
1️⃣ Clone the Repository
git clone https://github.com/yourusername/AI-FaceRecognition-Django.git
cd AI-FaceRecognition-Django

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Setup Database
python manage.py makemigrations
python manage.py migrate

5️⃣ Create Superuser
python manage.py createsuperuser

6️⃣ Run Server
python manage.py runserver


Then visit 👉 http://127.0.0.1:8000/

🧮 Example Embedding Logic
from keras_facenet import FaceNet
from numpy.linalg import norm
import numpy as np, cv2

embedder = FaceNet()
face = cv2.imread('student_images/shohaab.jpg')
face = cv2.resize(face, (160, 160))
embedding = embedder.embeddings(np.expand_dims(face, axis=0))[0]

# Compare
dist = norm(embedding - known_embedding)
if dist < 1.0:
    print("Face recognized!")
else:
    print("Unknown face")

🧰 Optimization Features

Model warm-up at server start (improves first-time accuracy)

Cached embeddings for faster lookup

Configurable threshold for accuracy tuning

Lightweight YOLOv8n model for fast inference

🌟 Future Improvements

🔹 Real-time attendance logging
🔹 Parent notification system via Twilio SMS
🔹 Multi-camera support
🔹 Dockerized deployment for production
🔹 Integration with cloud database (Firebase / AWS RDS)

🧑‍💻 Author

👨‍💻 Shohaab Aslam
🎓 Computer Science Student
💡 Project: AI-Based Real-Time Student Face Recognition System
🤝 Collaboration: Developed with assistance and guidance on model optimization, Django integration, and embedding automation.

📄 License

This project is open-source under the MIT License.
