from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect

# Create your views here.
def mainpage(request):
    return render(request, 'mainpage/index.html', {})

def search(request):
    keyword = request.GET.get('keyword')
    engine = request.GET.get('engine')

    if engine == 'google':
        search_url ='https://www.google.co.kr/search?q='+ keyword + '&tbm=isch'
    elif engine == 'naver':
        search_url = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query=' + keyword

    
    return HttpResponseRedirect(search_url)
