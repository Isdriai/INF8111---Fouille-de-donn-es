import nltk
import numpy as np
import pandas as pd
from scipy import stats as st

filename = "tweets.csv"

dataset = pd.read_csv(filename, sep=";")

size_sample = 200
sample = dataset["text"].sample(n=size_sample)

# https://medium.com/data-science-journal/how-to-correctly-select-a-sample-from-a-huge-dataset-in-machine-learning-24327650372c
# https://towardsdatascience.com/inferential-statistics-series-t-test-using-numpy-2718f8f9bf2f

tweets = sample.copy()

import re

for index, tweet in tweets.items():
    tweets[index] = re.sub(r"(http|@|#)\S*", "", tweet).lower() # URL/account/hashtag remove + lower

print(tweets)

from googletrans import Translator

list_text = list(map(lambda t: t[1], tweets.items()))

lt_chunks = []
chunk = []
down = size_sample
for lt in list_text:
    chunk.append(lt)
    down -= 1
    if (len(chunk) == 30 or down == 0):
        lt_chunks.append(chunk)
        chunk = []

"""
translations = []
for c in lt_chunks:
    translations += Translator().translate(list_text)

i = 0
for index, tweet in tweets.items():
    tweets[index] = translations[i]
    i += 1
"""

for index, tweet in tweets.items():
    tokens = nltk.word_tokenize(tweet)
    tags = nltk.pos_tag(tokens)
    tweets[index] = " ".join([word for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')])


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(1,2))
tweets_tfidf = tfidf.fit_transform(tweets)

from sklearn.cluster import KMeans

Kmeans = []
for k in range(4, 10):
    Kmeans.append(KMeans(n_clusters=k).fit(tweets_tfidf))
# correction de bruit

# regarder a la main quelques k qui semblent etre le point d'inflexion
# TSNE

# regarder les mots les plus présents dans les clusters et voitr si ils font sens

# les hashtags communs d'un topic doivent etre concentré sur ce topic 

# https://stats.stackexchange.com/questions/79028/performance-metrics-to-evaluate-unsupervised-learning

# differentes langues
# https://pypi.org/project/googletrans/

# word to vect
