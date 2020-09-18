conset Searching = Search.prototype;

function Search(){
    this.keyword = document.querySelector('input[name="search"]');
    this.engine = document.querySelector('.SelectSearch');
    this.button = document.querySelector('.img-button');
    this.form = document.querySelector('.search');
}

Searching.Engine = function(){
     this.form.addEventListener('submit', e=> {
        
         e.preventDefault();
        
        let engine = this.engine.value;
        let keyword = this.keyword.value;

        if(engine === 'google'){
            location.herf='https://www.google.co.kr/search?q='+keyword;
         }
         else if(engine === 'naver'){
             location.herf = 'https://search.naver.com/search.naver?query=' + keyword;
         }
     });
 }

new Search();
