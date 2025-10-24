from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('add/', views.register_student, name='add_student'),
    path('check_face/', views.check_face, name='check_face'),
    path('register/', views.register_student, name='register_student'),
    path('view-students/', views.view_students, name='view_students'),
    path('delete/<int:student_id>/', views.delete_student, name='delete_student'),
    path('edit/<int:student_id>/', views.edit_student, name='edit_student'),
    path('recapture_image/<int:pk>/', views.recapture_image, name='recapture_image'),
]
