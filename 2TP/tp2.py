import nltk
import numpy as np
import pandas as pd
from scipy import stats as st

filename = "tweets.csv"

dataset = pd.read_csv(filename)

size_sample = 1000000

def get_sample():
    return dataset["text"].sample(n=size_sample) # Nous prenons seulement le texte de chaque tweet

sample = get_sample()

# https://medium.com/data-science-journal/how-to-correctly-select-a-sample-from-a-huge-dataset-in-machine-learning-24327650372c
# https://towardsdatascience.com/inferential-statistics-series-t-test-using-numpy-2718f8f9bf2f

# p value > 5% => ok

while st.ttest_1samp(sample, 0)[1] < 0.05:
    sample = get_sample()

# garder les noms
for index, row in df.iterrows():
    row["text"] = nltk.pos_tag(word_tokenize(row["text"])) 

# bigram nom nom

# TF IDF


# correction de bruit

# comparaison vect word

# k allant de 1 à 30
# k mean 
# regarder les k ou il y a une grande diff

# regarder a la main quelques k qui semblent etre le point d'inflexion

# test => pk pas regarder la distribution par rapport aux topics,
# les hashtags communs d'un topic doivent etre concentré sur ce topic 
# https://stats.stackexchange.com/questions/79028/performance-metrics-to-evaluate-unsupervised-learning




