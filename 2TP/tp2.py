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


tfidf = TfidfVectorizer(
    min_df = 5,
    max_df = 0.95,
    max_features = 9000,
    stop_words = 'english'
)
tweets_tfidf = tfidf.fit_transform(sample)

print("fin tfidf")


def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1)
    
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
        
    return sse

s = find_optimal_clusters(tweets_tfidf, 100)
nbr_clusters = index_min = min(range(len(s)), key=s.__getitem__)
clusters = MiniBatchKMeans(n_clusters=nbr_clusters, init_size=1024, batch_size=2048, random_state=20).fit_predict(tweets_tfidf)

def plot_tsne_pca(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=3000, replace=False)
    
    pca = PCA(n_components=2).fit_transform(data[max_items,:].todense())
    pp = PCA(n_components=50)
    tsne = TSNE().fit_transform(pp.fit_transform(data[max_items,:].todense()))
    
    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
    
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot')
    
    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('TSNE Cluster Plot')

    return pp.explained_variance_
    
variance = plot_tsne_pca(tweets_tfidf, clusters)
plt.show()

plt.plot(range(len(variance)), variance)
plt.show()

def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
            
get_top_keywords(tweets_tfidf, clusters, tfidf.get_feature_names(), 10)

