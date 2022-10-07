import nltk
import re
import pickle
import sklearn
import numpy as np

word_data = sklearn.datasets.load_files('text')


from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer(max_features=2000, stop_words=nltk.corpus.stopwords.words('russian'))
vector2 = CountVectorizer(max_features=2000)
dox = vector.fit_transform(word_data).toarray()
doy = vector2.fit_transform(word_data).toarray()


from sklearn.model_selection import train_test_split
dox_train, dox_test, doy_train, doy_test  = train_test_split(dox, doy, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000, random_state=0) 
classifier.fit(dox_train, doy_train)

with open('text_classifier', 'wb') as picklefile: 
    pickle.dump(classifier, picklefile)