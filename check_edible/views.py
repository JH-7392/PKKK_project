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
    # Edible data
    fruits = ['almond', 'apple', 'broccoli', 'carrot', 'grape', 'mandarin', 'melon', 'onion', 'welshOnion' ]
    fruits_kor = {
        'almond': '아몬드', 
        'apple': '사과', 
        'broccoli': '브로콜리', 
        'carrot': '당근', 
        'grape': '포도',
        'mandarin': '귤', 
        'melon': '참외', 
        'onion': '양파',
        'welshOnion': '대파' }
    cat = [False, True, True, True, False, True, True, False, False]
    dog = [False, True, True, True, False, True, True, False, False]
    edible_table = pd.DataFrame(index=fruits, columns = ['dog', 'cat'])
    edible_table['dog'] = dog
    edible_table['cat'] = cat

    # Precaution edible_table
    dog = [
        '아몬드는 고지방 식품이기 때문에 췌장염 위험이 있습니다. 아몬드를 통째로 삼킬 경우에는 위장관 폐쇄의 위험이 있으니 급여하지 않는 것이 좋습니다.', 
        '사과의 씨앗, 껍질, 심 부분을 제거 한 후 급여해주세요. 알러지나 부작용이 있을 수 있으니, 소량 급여를 통해 부작용을 확인하세요. 과량 섭취시에는 설사나 복통을 유발할 수 있습니다.', 
        '브로콜리는 먹어도 되지만, 소량만 주어야 합니다. 양념이나 기름으로 구운상태로 주면 안 됩니다. 간이 안된 상태로 준다면 쪄서 주거나 생으로 줘도 됩니다. 줄기 부분이 식도에 걸릴 수 있기 때문에 잘게 잘라서 주는 것이 좋습니다.', 
        '당근은 눈 건강에 좋으며, 당근에 함유되어있는 비타민 A는 자궁건강에 좋습니다. 생당근을 잘게 썰어서 급여하거나 찐 당근도 급여 가능합니다.', 
        '중독을 일으켜 급성 신부전이 올 수 있습니다. 구토, 설사 등을 동반하며 사망에 이를 수 있어 위험합니다.', 
        '귤 껍질과 알맹이에 있는 흰색 부분을 제거하시고, 급여해주세요. 당이 높기 때문에 과도한 섭취는 금물입니다.', 
        '껍질과 씨 부분을 제거한 후 급여해주세요.', 
        '양파 내의 N-propyl disulfide라는 독성 성분이 적혈구를 파괴하여 빈혈을 야기할 수 있습니다. ', 
        '양파와 마찬가지로 독성 성분이 적혈구를 파괴하고, 용혈성 빈혈 증상이 일어날 수 있습니다. 심하면 사망에 이를 수 있으니 주의하세요.']
    cat = [
        '아몬드를 통째로 삼킬 경우에는 위장관 폐쇄의 위험이 있으니 급여하지 않는 것이 좋습니다.', 
        '사과의 씨앗은 \'시안배당체\'라는 물질이 포함되어 있어 고양이에게 독이 될 수 있으니 꼭 제거하고 급여하세요. 또한, 고양이에게 사과를 너무 많이 주면 소화 불량을 일으킬 수 있으므로 적당량 급여해 주세요.', 
        '브로콜리는 항암 효과가 뛰어납니다. 비타민 C가 레몬의 두 배 이상 함유되어 있어 감기 예방과 면역력 증가에도 도움이 됩니다. 또한, 칼슘도 풍부해 골다공증 및 심장병 예방에도 도움이 되는 것으로 알려져 있습니다. ', 
        '고양이는 당근을 섭취해도 괜찮지만, 비타민 A를 공급받을 수 없기 때문에 당근이 몸에 좋다고 하기는 힘듭니다. 다만 당근에는 식이 섬유가 많기 때문에 변비에 효과가 있을 수 있습니다. 하지만 과잉 섭취 했을 경우 간에 부담을 주거나 요로결석이 생길 위험이 있으니 주의가 필요합니다. ', 
        '강아지와 마찬가지로 고양이도 포도에 중독 증상을 나타낸 보고가 있으므로 주의해야 합니다. 특히 위험한 것이 급성 신부정 증상입니다. 급성 신부전은 신장이 제대로 작동하지 않아 사망까지 다다를 수 있는 무서운 질병이므로 포도를 급여하는 것은 매우 위험합니다.', 
        '고양이는 귤 껍질에 함유된 리모넨이라는 물질을 분해하지 못합니다. 과육, 과즙만 주는 것은 괜찮지만 주의가 필요합니다. 또한, 껍질의 즙이 닿으면 피부 염증을 일으키는 경우도 있으므로 주의가 필요하다.', 
        '참외에는 비타민 C가 풍부하게 함유되어 있어 피로회복에 도움이 되며, 과육의 수분 함량도 90% 이상이어서 섭취시 수분 보충에 많은 도움이 됩니다. 단, 참외 씨는 소화가 잘 되지 않거나 배탈을 유발할 수 있으므로 반드시 제거하고 과육만 적당한 크기로 썰어서 줍시다. 고양이 전용 우유와 함께 믹서기에 넣고 갈아서 주어도 괜찮습니다.', 
        '고양이에게 양파는 양파의 allyl propyl disulfide라는 성분이 헤모글로빈을 산화시켜 적혈구를 파괴합니다. 적혈구가 너무 많이 파괴되면 용혈성 빈혈이 생길 수 있습니다. 또한 이 성분은 가열해도 성질이 변하지 않기 때문에 요리시에 주의하셔야 합니다.', 
        '양파와 마찬가지로 독성 성분이 적혈구를 파괴하고, 용혈성 빈혈 증상이 일어날 수 있습니다. 심하면 사망에 이를 수 있으니 주의하세요.']
    precaution_table = pd.DataFrame(index=fruits, columns = ['dog', 'cat'])
    precaution_table['dog'] = dog
    precaution_table['cat'] = cat

    
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


    # # look for the mapping edible_table whether it is edible
    is_edible = edible_table[pet_type][prediction]
    precaution = precaution_table[pet_type][prediction]

    kor_name = fruits_kor[prediction]
    # Gather variables for rendering and throw it
    context = {
        'pet_type': pet_type,
        'food_image': food_image,
        'food_type': prediction,
        'is_edible': is_edible,
        'precaution': precaution,
        'kor_name': kor_name,
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
