import nltk
from nltk.util import ngrams
import re
import sklearn
from sklearn import neighbors
import numpy as np

#подготовка файла, токенизация, фильтрация
text = open('testfortest.txt')
text = text.read()
text = text.lower()
text = re.sub(r'[^а-я ]', '', text)
word_token = nltk.word_tokenize(text)
stop_words = nltk.corpus.stopwords.words('russian')
filtered = [word for word in word_token if word not in stop_words]

#подготовка для работы с функциями sklearn
from sklearn.feature_extraction.text import CountVectorizer 
vectorizer = CountVectorizer(max_features=1500) 
dox = vectorizer.fit_transform(filtered).toarray()

#великий костыль — введи слово, которое будет сохранено в документе для подготовки работы в sklearn
request = input('слово: ')
if request:
    doc = open('doctest.txt', 'w')
    doc.write(request)
    doc.close()

    text2 = open('doctest.txt')
    text2 = text2.read()
    text2 = text2.lower()
    word_token2 = nltk.word_tokenize(text2)

    word = vectorizer.fit_transform(word_token2).toarray()

    from sklearn.neighbors import KNeighborsClassifier
    knn = neighbors.KNeighborsClassifier(n_neighbors = 3)
    knn.fit(dox, word)
    # тут проводится обучение по основному доку и введёному слову
   
    words = knn.predict(1)
    print(words)
    #в плане было, что обучившись система начнёт выдавать наиболее близких соседей введённому слову.
    #я до сих пор думаю, что идея хорошая. но моих нынешних знаний не хватает для её воплощения.
else:
    pass
