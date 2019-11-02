import nltk
import numpy as np
import pandas as pd
from scipy import stats as st

filename = "tweets_lang.csv"


dataset = pd.read_csv(filename, sep=";")
print("fin dataset")

size_sample = 100000

labels = ["text", "lang", "truncated"]

sample = dataset.sample(n=size_sample)[labels]
sample = sample[(sample["lang"] == "en") & (sample["truncated"] == False)]

print("fin sample")

# https://medium.com/data-science-journal/how-to-correctly-select-a-sample-from-a-huge-dataset-in-machine-learning-24327650372c
# https://towardsdatascience.com/inferential-statistics-series-t-test-using-numpy-2718f8f9bf2f

tweets = sample["text"].copy()

import re

for index, tweet in tweets.items():
    tweets[index] = re.sub(r"(http|@|#)\S*", "", tweet).lower() # URL/account/hashtag remove + lower

print("fin clean")


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)

for index, tweet in tweets.items():
    tokens = nltk.word_tokenize(tweet)
    tags = nltk.pos_tag(tokens)
    tweets[index] = " ".join([stemmer.stem(word) for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')])

print("fin noms + stem")

percent = 0.50
chunk = int(len(tweets)*percent)
train = tweets.iloc[:chunk]
test = tweets.iloc[chunk:]

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(1,2))
tweets_train_tfidf = tfidf.fit_transform(train)

print("fin td idf")

from sklearn.cluster import KMeans

ran = range(1, 21)
ks = list(ran)

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

def kgo(k):
    print ("kmeans " + str(k))
    res = KMeans(n_clusters=k).fit(tweets_train_tfidf)
    print ("kmeans " + str(k) + " fini")
    return res

Kmeans = Parallel(n_jobs=num_cores)(delayed(kgo)(i) for i in ran)

"""def group(kmean, data):
    pred = kmean.predict(data)
    res = {}
    index = 0
    for p in pred:
        if p in res:
            res[p].append(index)
        else:
            res[p] = [index]
        index += 1
    return res"""

import matplotlib.pyplot as plt

inerties = []

for k in ks:
    inerties.append(Kmeans[k-ks[0]].inertia_)

plt.plot([ k - 4 for k in ks], inerties, 'o-')
plt.show()


print("fin")
# correction de bruit

# regarder a la main quelques k qui semblent etre le point d'inflexion
# TSNE

# regarder les mots les plus présents dans les clusters et voitr si ils font sens

# les hashtags communs d'un topic doivent etre concentré sur ce topic 

# https://stats.stackexchange.com/questions/79028/performance-metrics-to-evaluate-unsupervised-learning

# differentes langues
# https://pypi.org/project/googletrans/

# word to vect

