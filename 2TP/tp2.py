import nltk
import numpy as np
import pandas as pd
from scipy import stats as st
import re
from nltk.stem import WordNetLemmatizer
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD


filename = "tweets_lang.csv"

dataset = pd.read_csv(filename, sep=";")
# On ne garde que les tweets en anglais et ceux qui n'ont pas été coupés
dataset = dataset[(dataset["lang"] == "en") & (dataset["truncated"] == False)]
print("fin dataset")

size_sample = 100000

lemmatizer = WordNetLemmatizer()
def clean_tweet(tweet):
    clean = re.sub(r"(http|@|#)\S*", "", tweet.lower())
    tokens = nltk.word_tokenize(re.sub(r"rt", "", clean))
    tags = nltk.pos_tag(tokens)
    nouns = [word for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
    return " ".join([lemmatizer.lemmatize(noun) for noun in nouns])

sample = dataset.sample(n=size_sample)["text"]
sample = sample.map(lambda tweet: clean_tweet(tweet))

print("fin clean")

tfidf = TfidfVectorizer()
tweets_tfidf = tfidf.fit_transform(sample)

print("fin tfidf")

svd = TruncatedSVD(n_components=20)
tweets_svd = svd.fit_transform(tweets_tfidf)

print("fin svd")

ran = range(1, 162, 20)
ks = list(ran)

def kmeans_go(k):
    print("kmeans " + str(k))
    res = KMeans(n_clusters=k).fit(tweets_svd)
    print("kmeans " + str(k) + " fini")
    return res

num_cores = multiprocessing.cpu_count()
Kmeans = Parallel(n_jobs=num_cores-1)(delayed(kmeans_go)(i) for i in ran)

inerties = []

for k in Kmeans:
    inerties.append(k.inertia_)

plt.plot(ks, inerties, 'o-')
plt.show()
