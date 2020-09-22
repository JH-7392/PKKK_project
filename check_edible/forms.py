from django.db import models
from django import forms

class UploadForm(forms.Form):
    CHOICES=[('dog','dog'), ('cat','cat')]
    attrs_radio_button = {'class': 'radio-control', }
    attrs_image_button = {'class': 'image-control', 'accept': 'image/*;capture=camera' }

    pet_type = forms.ChoiceField(
        widget=forms.RadioSelect(attrs=attrs_radio_button), 
        choices=CHOICES, 
        label="Pet type")
        
    food_image = forms.ImageField(label="Food for check")