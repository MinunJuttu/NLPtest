import nltk
from nltk.util import ngrams
import re
import sklearn
from sklearn import neighbors
import numpy as np


text = open('testfortest.txt')
text = text.read()
text = text.lower()
text = re.sub(r'[^а-я ]', '', text)
word_token = nltk.word_tokenize(text)
stop_words = nltk.corpus.stopwords.words('russian')
filtered = [word for word in word_token if word not in stop_words]


from sklearn.feature_extraction.text import CountVectorizer 
vectorizer = CountVectorizer(max_features=1500) 
dox = vectorizer.fit_transform(filtered).toarray()
doy = vectorizer.fit_transform(filtered).toarray()

from sklearn.neighbors import KNeighborsClassifier
knn = neighbors.KNeighborsClassifier(n_neighbors = 3)
knn.fit(dox, doy)

request = input('слово: ')
if request:
	word = vectorizer.fit_transform(request).toarray()
	words = knn.predict(word)
	print(words)
else:
    pass
