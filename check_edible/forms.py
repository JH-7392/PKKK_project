from django.db import models
from django import forms

class UploadForm(forms.Form):
    CHOICES=[('dog','dog'), ('cat','cat')]
    attrs_radio_button = {'class': 'radio-control', }
    attrs_image_button = {'class': 'image-control', }

    pet_type = forms.ChoiceField(
        widget=forms.RadioSelect(attrs=attrs_radio_button), 
        choices=CHOICES, 
        label="What kind of pet do you live with?")
        
    food_image = forms.ImageField(label="Upload food pic to check edible.")