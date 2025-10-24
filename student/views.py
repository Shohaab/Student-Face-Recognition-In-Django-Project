import base64, cv2, numpy as np, os, json,subprocess,sys
from django.http import JsonResponse
from django.shortcuts import render,get_object_or_404, redirect
from django.conf import settings
from .models import Student
from .face_utilis import detect_and_embed,reload_embedddings # custom helper


from django.shortcuts import render, redirect, get_object_or_404
from .models import Student
from .forms import StudentForm
import os
import numpy as np
from keras_facenet import FaceNet
from PIL import Image
from mtcnn import MTCNN
import cv2



from django.core.paginator import Paginator
from django.db.models import Q
from django.shortcuts import render
from .models import Student




embedder = FaceNet()
detector = MTCNN()
EMBEDDINGS_DIR = r"E:\Python\AI_face_recognition\embeddings"







# 🏠 Home page
def home(request):
    return render(request, "student/home.html")


# 🔍 Face recognition check
"""def check_face(request):
    if request.method == "POST":
        image_data = request.POST.get('image')
        if not image_data:
            return JsonResponse({'error': 'No image provided'}, status=400)
        
        img_bytes = base64.b64decode(image_data.split(',')[1])
        np_img = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        name, status = detect_and_embed(frame)
        return JsonResponse({'status': status, 'name': name})

    # If someone opens /check_face/ directly
    return JsonResponse({'error': 'Invalid request'}, status=400)
"""
def check_face(request):
    if request.method == "POST":
        image_data = request.POST.get('image')

        if not image_data:
            return JsonResponse({'status': 'error', 'message': 'No image received'})

        # Remove data:image/jpeg;base64, header
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        name, status = detect_and_embed(frame)
        return JsonResponse({'status': status, 'name': name})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'})






#  Student registration
"""def register_student(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        father_name = request.POST.get('father_name')
        surname = request.POST.get('surname')
        phone_no = request.POST.get('phone_no')
        image_data = request.POST.get('image')

        if not image_data:
            return JsonResponse({'status': 'error', 'message': 'No image received.'})

        # Decode and save image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)

        folder = os.path.join(settings.MEDIA_ROOT, 'student_images')
        os.makedirs(folder, exist_ok=True)
        image_filename = f"{name}_{surname}.jpg"
        image_path = os.path.join(folder, image_filename)

        with open(image_path, 'wb') as f:
            f.write(image_bytes)

        # Save student data
        Student.objects.create(
            name=name,
            father_name=father_name,
            surname=surname,
            phone_no=phone_no,
            image=f'student_images/{image_filename}'
        )
        subprocess.run(['python', 'face_recognize/student/manage_embeddings.py'])
        reload_embedddings()
        # Optional: regenerate embeddings
        # os.system(r"python face_recognize/student/manage_embeddings.py")
        
        return JsonResponse({'status': 'success', 'message': 'Student data saved successfully.'})

    # 👇 This line should be OUTSIDE the if-block
    # so GET request will show the HTML page
    return render(request, "student/form.html")
"""


def register_student(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        father_name = request.POST.get('father_name')
        surname = request.POST.get('surname')
        phone_no = request.POST.get('phone_no')
        image_data = request.POST.get('image')

        if not image_data:
            return JsonResponse({'status': 'error', 'message': 'No image received.'})

        # Decode Base64 image and save
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)

        folder = os.path.join(settings.MEDIA_ROOT, 'student_images')
        os.makedirs(folder, exist_ok=True)
        image_filename = f"{name}_{surname}.jpg"
        image_path = os.path.join(folder, image_filename)

        with open(image_path, 'wb') as f:
            f.write(image_bytes)

        # Save student in DB
        Student.objects.create(
            name=name,
            father_name=father_name,
            surname=surname,
            phone_no=phone_no,
            image=f'student_images/{image_filename}'
        )

        # --- Run manage_embeddings.py with the same Python environment ---
        manage_script = os.path.join(settings.BASE_DIR, 'student', 'manage_embeddings.py')

        try:
            subprocess.run(
                [sys.executable, manage_script],
                check=True,
                capture_output=True,
                text=True
            )
            print("✅ manage_embeddings.py executed successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Embedding update failed:\n{e.stderr}")
            return JsonResponse({'status': 'error', 'message': f'Embedding update failed: {e.stderr}'})

        # Reload embeddings into memory
        reload_embedddings()
        print("✅ Embeddings reloaded in memory.")

        return JsonResponse({'status': 'success', 'message': 'Student added and embeddings updated!'})

    return render(request, "student/form.html")

def view_students(request):
    query = request.GET.get('q', '')

    students_list = Student.objects.all()
    if query:
        students_list = students_list.filter(
            Q(name__icontains=query) |
            Q(father_name__icontains=query) |
            Q(surname__icontains=query)
        )

    paginator = Paginator(students_list, 10)  # 10 per page
    page_number = request.GET.get('page')
    students = paginator.get_page(page_number)

    return render(request, 'student/view_students.html', {'students': students})

def edit_student(request, pk):
    student = get_object_or_404(Student, pk=pk)
    if request.method == 'POST':
        form = StudentForm(request.POST, request.FILES, instance=student)
        if form.is_valid():
            form.save()
            return redirect('view_students')
    else:
        form = StudentForm(instance=student)
    return render(request, 'student/edit_student.html', {'form': form})


def delete_student(request, pk):
    student = get_object_or_404(Student, pk=pk)
    student.delete()
    return redirect('view_students')


def recapture_image(request, pk):
    student = get_object_or_404(Student, pk=pk)

    # Open webcam
    cap = cv2.VideoCapture(0)
    print("Press 's' to capture new image for", student.name)
    while True:
        ret, frame = cap.read()
        cv2.imshow("Recapture Image - Press 's' to Save or 'q' to Quit", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            filename = f"{student.name}.jpg"
            save_path = os.path.join("media/student_images", filename)
            cv2.imwrite(save_path, frame)
            student.image = f"student_images/{filename}"
            student.save()
            print("✅ New image saved successfully.")
            break
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    # Generate new embedding
    image = Image.open(student.image.path)
    image = np.asarray(image)
    faces = detector.detect_faces(image)
    if faces:
        x, y, w, h = faces[0]['box']
        face = image[y:y+h, x:x+w]
        face = Image.fromarray(face).resize((160, 160))
        embedding = embedder.embeddings([np.asarray(face)])[0]

        # Load existing data
        emb_path = os.path.join(EMBEDDINGS_DIR, "embeddings/embeddings.npy")
        names_path = os.path.join(EMBEDDINGS_DIR, "embeddings/names.npy")

        if os.path.exists(emb_path) and os.path.exists(names_path):
            embeddings = np.load(emb_path, allow_pickle=True).tolist()
            names = np.load(names_path, allow_pickle=True).tolist()
        else:
            embeddings, names = [], []

        # Replace or add new embedding
        if student.name in names:
            idx = names.index(student.name)
            embeddings[idx] = embedding
        else:
            embeddings.append(embedding)
            names.append(student.name)

        np.save(emb_path, embeddings)
        np.save(names_path, names)
        print("✅ Embedding updated successfully.")

    return redirect('view_students')









