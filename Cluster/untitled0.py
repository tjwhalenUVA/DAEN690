#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 09:47:14 2019

@author: davidwilbur
"""
import os
import re
import lda
import time
import logging
import numpy as np
import sqlite3
import pandas as pd
from sklearn.manifold import TSNE
from Functions_Wilbur_Zimmermann import tokenize_only
from sklearn.feature_extraction.text import CountVectorizer
# Plotly plotting packages
import plotly
import plotly.graph_objs as go
from nltk.corpus import stopwords
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn import metrics 
import gensim.models as g
import codecs
from nltk.cluster import KMeansClusterer
import nltk


from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
import nltk
import numpy as np 
from sklearn import cluster
from sklearn import metrics
#%%  
# training data
 # %%
# Set global path variables
directory = os.path.dirname('__file__')
input_file = os.path.join(directory,'..','Article Collection')
_dbFile = os.path.join(input_file,'articles_zenodo.db')

#%%
# Set up our connection to the database
#
_db = sqlite3.connect(_dbFile)
_cursor = _db.cursor()

# Pull the relevant bits of data out of the database and
# store them in a pandas dataframe.
#
print('Pulling article IDs, leanings, and text from database')
_cursor.execute("SELECT ln.id, ln.bias_final, cn.text " +
                "FROM train_lean ln, train_content cn " +
                "WHERE cn.`published-at` >= '2018-01-01' AND ln.id == cn.id AND ln.url_keep='1'")
_df = pd.DataFrame(_cursor.fetchall(), columns=('id', 'lean', 'text'))
_db.close()
#%%
_df.text = _df.text.str.lower()
nonAlpha = re.compile('[^a-zA-Z]+')
_df.text = _df.text.replace(nonAlpha, ' ')

cachedStopWords = stopwords.words('english')
_df.text = _df.text.apply(
                          lambda x: (' '.join([word for word in x.split() 
                          if word not in cachedStopWords])))

#%%
print('Tokenizing Articles')
import gensim
from nltk.tokenize import word_tokenize
raw_documents = _df.text
gen_docs = [[w.lower() for w in word_tokenize(text)] for text in raw_documents] 

 #%% 
  
model = Word2Vec(gen_docs, min_count=1)
 
def sent_vectorizer(sent, model):
    sent_vec =[]
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass
     
    return np.asarray(sent_vec) / numw
  
  
X=[]
for sentence in gen_docs:
    X.append(sent_vectorizer(sentence, model))   
 
 
NUM_CLUSTERS=5
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
print (assigned_clusters)
  
  
for index, sentence in enumerate(sentences):    
    print (str(assigned_clusters[index]) + ":" + str(sentence))
 
     
kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X)
  
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
  
print ("Cluster id labels for inputted data")
print (labels)
print ("Centroids data")
print (centroids)
  
print ("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
print (kmeans.score(X))
  
silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
  
print ("Silhouette_score: ")
print (silhouette_score)
 

import matplotlib.pyplot as plt
 
from sklearn.manifold import TSNE
 
model = TSNE(n_components=3, random_state=0)
np.set_printoptions(suppress=True)
 
Y=model.fit_transform(X)
_lean = _df['lean'][_idx]

def SetColor(x):
    if(x == 'left'): return "blue"
    elif(x == 'left-center'): return "lightblue"
    elif(x == 'least'): return "lightgray"
    elif(x == 'right-center'): return "orangered"
    elif(x == 'right'): return "red"

tsne_lda_df = pd.DataFrame(Y, columns=['x', 'y', 'z'])
# tsne_lda_df['Resumes'] = tsne_lda_df['Resumes']
x = tsne_lda_df.x
y = tsne_lda_df.y
z = tsne_lda_df.z

trace1 =\
    go.Scatter3d(x=x, y=y, z=z,
                 mode='markers',
                 marker = dict(color=list(map(SetColor, _lean))))
    
data = [trace1]
layout = go.Layout(
                   scene=dict(
                    xaxis=dict(
                        nticks=4, range=[-20, 20], ),
                    yaxis=dict(
                        nticks=4, range=[-20, 20], ),
                    zaxis=dict(
                        nticks=4, range=[-20, 20], ), ),
                    width=700,
                    margin=dict(
                    r=20, l=10,
                    b=10, t=10)
                  )
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig)
    
# plt.scatter(Y[:, 0], Y[:, 1], c=assigned_clusters, s=290,alpha=.5)
# 
# 
#for j in range(len(sentences)):    
#   plt.annotate(assigned_clusters[j],xy=(Y[j][0], Y[j][1]),xytext=(0,0),textcoords='offset points')
#   print ("%s %s" % (assigned_clusters[j],  sentences[j]))
# 
# 
#plt.show()