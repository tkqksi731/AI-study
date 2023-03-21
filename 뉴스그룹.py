from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
newsgroups = fetch_20newsgroups(subset='all')
# print(newsgroups["DESCR"])
import numpy as np

newsgroups_train = fetch_20newsgroups(subset='train')

print(newsgroups_train.target.shape)
print(newsgroups_train.target[:10])

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
print(vectors.shape)
# print( vectors[:10])
