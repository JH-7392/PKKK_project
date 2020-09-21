from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from .forms import UploadForm

import os
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Create your views here.
def index(request):
    return render(request, 'check_edible/base.html', {})


def upload(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            pet_type = form.cleaned_data['pet_type']
            food_image = form.cleaned_data['food_image']

            form.save() 

            # UserInput.objects.create(pet_type=pet_type, food_image=food_image)
            return redirect('result', {'pet_type': pet_type, 'food_image': food_image})
    else:
        form = UploadForm()

    return render(request, "check_edible/upload.html", {'form': form}) 


def result(request):
    # table data
    fruits = ['almond', 'apple', 'broccoli', 'carrot', 'grape', 'mandarin', 'melon', 'onion', 'welshOnion' ]
    cat = [False, True, False, True, True, False, True, True, False]
    dog = [False, True, False, True, True, False, True, True, False]
    table = pd.DataFrame(index=fruits, columns = ['dog', 'cat'])
    table['dog'] = dog
    table['cat'] = cat

    pet_type = request.POST['pet_type']
    food_image = request.FILES['food_image']
    
    # model inference part
    np.set_printoptions(suppress=True)

    modelpath = './check_edible/saved_model'
    model = tf.keras.models.load_model(modelpath)

    # Image preprocess
    image_size = 100
    # img_path = os.path.join(food_image)
    img = Image.open(food_image)
    img = img.resize((image_size, image_size))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, 0)

    prediction = model.predict(img)
    score = tf.nn.softmax(prediction[0])
    prediction = fruits[np.argmax(score)]

    # data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # image = Image.open(food_image)
    # size = (224, 224)
    # image = ImageOps.fit(image, size, Image.ANTIALIAS)
    # image_array = np.asarray(image)
    # normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # data[0] = normalized_image_array

    # prediction = model.predict(data)


    # # look for the mapping table whether it is edible
    is_edible = table[pet_type][prediction]

    
    # Gather variables for rendering and throw it
    context = {
        'pet_type': pet_type,
        'food_image': food_image,
        'food_type': prediction,
        'is_edible': is_edible,
    }
    return render(request, "check_edible/result.html", context)


def search(request):
    keyword = request.GET.get('keyword')
    engine = request.GET.get('engine')

    if engine == 'google':
        search_url ='https://www.google.co.kr/search?q='+ keyword + '&tbm=isch'
    elif engine == 'naver':
        search_url = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query=' + keyword

    
    return HttpResponseRedirect(search_url)
