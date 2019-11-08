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
from collections import Counter
import csv
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from sklearn.decomposition import SparsePCA
import seaborn as sns
from sklearn.datasets import load_digits
from tsne import bh_sne




filename = "tweets_lang.csv"

dataset = pd.read_csv(filename, sep=";")
# On ne garde que les tweets en anglais et ceux qui n'ont pas été coupés
dataset = dataset[(dataset["lang"] == "en") & (dataset["truncated"] == False)]
print("fin dataset")

size_sample = 100000

lemmatizer = WordNetLemmatizer()
def clean_tweet(tweet):
    clean = re.sub(r"(http|@|#)\S*", "", tweet.lower())
    tokens = nltk.word_tokenize(re.sub(r"rt|”|'s|—|'|\.|,|\"|%|`|’|‘|–|\||“|▸", "", clean))
    tags = nltk.pos_tag(tokens)
    nouns = [word for word,pos in tags if ((pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS' ) and word != "i" and word != "s")]
    return " ".join([lemmatizer.lemmatize(noun) for noun in nouns])

sample = dataset.sample(n=size_sample)["text"]
sample = sample.map(clean_tweet)

del dataset

tfidf = TfidfVectorizer(
    ngram_range=(1,2),
    stop_words = 'english'
)
tweets_tfidf = tfidf.fit_transform(sample)

print("fin tfidf")


components = 9
SVD = TruncatedSVD(n_components = components)
tweets_svd = SVD.fit_transform(tweets_tfidf)


plt.bar(range(components), SVD.explained_variance_ratio_)
plt.show() # si on prend large, 10

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

nbr_clusters = 20
clusters = MiniBatchKMeans(n_clusters=nbr_clusters, init_size=1024, batch_size=2048, random_state=20).fit_predict(tweets_svd)

max_items = np.random.choice(range(tweets_svd.shape[0]), size=2000)

tsne = bh_sne(tweets_svd[max_items])
vis_x = tsne[:, 0]
vis_y = tsne[:, 1]

plt.scatter(vis_x, vis_y, c=clusters[max_items], cmap=plt.cm.get_cmap("jet", nbr_clusters))
plt.colorbar(ticks=range(nbr_clusters))
plt.show()

predict = clusters
tweets = sample
labels = list(range(nbr_clusters))
n_words = 10

group = {}
for l in labels:
    group[l] = []
for i in range(len(predict)):
    group[predict[i]].append(i)
most_commons = {}
for l in labels:
    counter = Counter([word for tweet in tweets.iloc[group[l]] for word in tweet.split()])
    most_commons[l] = counter.most_common(n_words)

n = 0
for i in range(nbr_clusters):
    n += len(group[i])
    print("cluster " + str(i))
    print(most_commons[i])