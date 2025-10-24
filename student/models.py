from django.db import models

# Create your models here.

class Student(models.Model):
    name=models.CharField(max_length=100)
    father_name=models.CharField(max_length=100)
    surname=models.CharField(max_length=100)
    phone_no=models.CharField(max_length=15)
    image=models.ImageField(upload_to='student_images')
    embedding_file=models.CharField(max_length=255,blank=True,null=True)
    
    def __str__(self):
        return self.name