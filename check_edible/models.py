from django.db import models
from django.core.files.storage import FileSystemStorage

fs = FileSystemStorage(location='/media/images')


# Create your models here.
class UserInput(models.Model):
    CHOICES=[('dog','dog'), ('cat','cat')]

    pet_type = models.CharField(max_length=3, choices=CHOICES)
    food_image = models.ImageField(upload_to='images/')

    