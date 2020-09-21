from django.db import models
from django import forms
from .models import UserInput

class UploadForm(forms.ModelForm):
    class Meta: 
        model = UserInput 
        fields = ['pet_type', 'food_image'] 


#     pet_type = forms.CharField(
#          max_length=100, label='pet type',
#          widget=forms.TextInput(attrs={}))
        
#     food_image = forms.ImageField(label="Food image")