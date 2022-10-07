import nltk
import re
import string


def wordtoken(text):
    text = text.read()
    text = text.lower() 
    text = re.sub(r'[^а-я ]', '', text)
    word_token = nltk.word_tokenize(text)
    return word_token

def filterednsave(word_token):
    stop_words = nltk.corpus.stopwords.words('russian')
    filtered = [word for word in word_token if word not in stop_words]

    dictionary = open('dictionary.txt', 'w')
    for word in filtered:
        dictionary.write(word + '\n')
    dictionary.close()

text = open('testfortest.txt') 
word_token = wordtoken(text)
word_token = filterednsave(word_token)
text.close()







