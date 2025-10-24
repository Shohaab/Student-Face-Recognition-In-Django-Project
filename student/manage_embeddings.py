from keras_facenet import FaceNet
import numpy as np, os, cv2




# Path to images uploaded from Django
dataset_dir = r"E:\Python\AI_face_recognition\face_recognize\media\student_images"
embedder = FaceNet()

embeddings, names = [], []

# Generate embeddings for each image in the folder
for file in os.listdir(dataset_dir):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(dataset_dir, file)
        img = cv2.imread(path)
        if img is None:
            print(f" Could not read {file}")
            continue
        img = cv2.resize(img, (160, 160))
        emb = embedder.embeddings(np.expand_dims(img, axis=0))[0]
        embeddings.append(emb)
        names.append(os.path.splitext(file)[0])

# Save the new embeddings and names
os.makedirs(r"E:\Python\AI_face_recognition\embeddings", exist_ok=True)
np.save(r"E:\Python\AI_face_recognition\embeddings\embeddings.npy", np.array(embeddings))
np.save(r"E:\Python\AI_face_recognition\embeddings\names.npy", np.array(names))

print(f" Saved {len(names)} embeddings from {dataset_dir}")
print("Names:", names)
print(f" Embeddings generated and saved in {os.getcwd()}")
