import nltk
from nltk.util import ngrams
import re

#здесь мы открываем файл и готовим его к работе с nltk
text = open('testfortest.txt').read()
text = text.lower
text = re.sub(r'[^а-я ]', '', text)

#здесь токенизируем текст и удаляем стопслова
word_token = nltk.word_tokenize(text)
stop_words = nltk.corpus.stopwords.words('russian')
filtered = [word for word in word_token if word not in stop_words]

#самый простой код для создания ngram с помощью встроенной в nltk функции
ngrams = nltk.ngrams(filtered, 3)
for gram in ngrams:
    print(gram)
