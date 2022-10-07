import nltk
from nltk.util import ngrams
import re

text = open('testfortest.txt').read()
text = re.sub(r'[^а-я ]', '', text)
word_token = nltk.word_tokenize(text)
stop_words = nltk.corpus.stopwords.words('russian')
filtered = [word for word in word_token if word not in stop_words]

ngrams = nltk.ngrams(filtered, 3)
for gram in ngrams:
    print(gram)