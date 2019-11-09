import nltk
import numpy as np
import pandas as pd
from scipy import stats as st
import re
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
from collections import Counter
import csv
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from tsne import bh_sne

# Seulement les mots les plus fréquents, c'est à dire les résultats finaux seront affichés, 
# pour afficher les étapes intermédiaires, décommenter les "plt.*"


########################### RECUPERATION DES DONNEES ###########################


filename = "tweets_lang.csv"

dataset = pd.read_csv(filename, sep=";") ##### Attention, le séparateur des données dans le dataset doit etre ";"

# On ne garde que les tweets en anglais et ceux qui n'ont pas été coupés
################################################################################################################################
#                                                                                                                              #
# ATTENTION !!! Le dataset doit contenir non seulement le texte des tweets mais aussi les attributs "lang" et "truncated" !!!! #
#                                                                                                                              #
################################################################################################################################
dataset = dataset[(dataset["lang"] == "en") & (dataset["truncated"] == False)]
print("fin dataset")


########################### PREPROCESSING et ECHANTILLONNAGE ###########################

lemmatizer = WordNetLemmatizer()
def clean_tweet(tweet):
    clean = re.sub(r"(http|@|#)\S*", "", tweet.lower())  # On enlebe les URLs, les hastags et les références aux autres comptes
    tokens = nltk.word_tokenize(re.sub(r"rt|”|'s|—|'|\.|,|\"|%|`|’|‘|–|\||“|▸", "", clean))   # on supprime les caractère spéciaux
    tags = nltk.pos_tag(tokens)
    nouns = [word for word,pos in tags if ((pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS' ) # on ne garde que les noms/noms propres
    and word != "i" and word != "s" and len(word) > 3)]
    return " ".join([lemmatizer.lemmatize(noun) for noun in nouns]) # On lemmatize

size_sample = 100000

sample = dataset.sample(n=size_sample)["text"]
sample = sample.map(clean_tweet)

del dataset # nous n'avons plus besoin du dataset en entier, nous ne gardons que notre echantillon
print("fin clean")

tfidf = TfidfVectorizer(
    ngram_range=(1,2),
    stop_words = 'english'
)
tweets_tfidf = tfidf.fit_transform(sample)

print("fin tfidf")

########################### SVD ###########################

components = 5
SVD = TruncatedSVD(n_components = components)
tweets_svd = SVD.fit_transform(tweets_tfidf)

print("fin svd")

#plt.bar(range(components), SVD.explained_variance_ratio_)
#plt.show() # On prendra les 5 première composantes

########################### K-MEANS ###########################


ran = range(1, 21, 1)
ks = list(ran)

def kmeans_go(k):
    print("kmeans " + str(k))
    res = KMeans(n_clusters=k).fit(tweets_svd)
    print("kmeans " + str(k) + " fini")
    return res

num_cores = multiprocessing.cpu_count()
Kmeans = Parallel(n_jobs=num_cores-1)(delayed(kmeans_go)(i) for i in ran) # parallélisation, cela laissera seulement un thread de disponible sur la machine

inerties = []

for k in Kmeans:
    inerties.append(k.inertia_)

print("fin des differents k-means")

#plt.plot(ks, inerties, 'o-')
#plt.show()

nbr_clusters = 12
clusters = MiniBatchKMeans(n_clusters=nbr_clusters, init_size=1024, batch_size=2048, random_state=20).fit_predict(tweets_svd)

print("fin k-means")

########################### T-SNE ###########################

max_items = np.random.choice(range(tweets_svd.shape[0]), size=2000) # cela ne sert à rien d'afficher 100000 tweets, un sous-echantillon de 2000 tweets suffira

tsne = bh_sne(tweets_svd[max_items])
axe_x = tsne[:, 0] # premier axe
axe_y = tsne[:, 1] # deuxième axe

plt.scatter(axe_x, axe_y, c=clusters[max_items], cmap=plt.cm.get_cmap("jet", nbr_clusters))

print("fin t-sne")

#plt.colorbar(ticks=range(nbr_clusters))
#plt.show()


########################### Calcul des mots les plus fréquents dans un cluster ###########################


labels = list(range(nbr_clusters))
n_words = 10

# On trie d'abord les tweets par leur cluster d'appartenance
group = {}
for l in labels:
    group[l] = []
for i in range(len(clusters)):
    group[clusters[i]].append(i)


# Puis pour chaque cluster, nous calculons les mots les plus fréquents
most_commons = {}
for l in labels:
    counter = Counter([word for tweet in sample.iloc[group[l]] for word in tweet.split()])
    most_commons[l] = counter.most_common(n_words)


# Affichage des mots les plus fréquents, bien sur selon l'echantillonnage, 
# les résultats finaux seront plus ou moins identiques à ceux indiqués dans le rapport
n = 0
for i in range(nbr_clusters):
    n += len(group[i])
    print("cluster " + str(i))
    print(most_commons[i])
