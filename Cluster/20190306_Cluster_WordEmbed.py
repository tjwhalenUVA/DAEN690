#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 18:05:25 2019

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

#%%

#%%#from gensim.models import Word2Vec
#model = Word2Vec(test1, min_count=1)
    
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)


from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


>>>
>>> glove_file = datapath('glove.6B.50d.txt')
>>> tmp_file = get_tmpfile("test_word2vec.txt")
>>>
>>> _ = glove2word2vec(glove_file, tmp_file)
>>>
>>> model = KeyedVectors.load_word2vec_format(tmp_file)

#%%
X = model.wv



NUM_CLUSTERS=3
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=2)
assigned_clusters = kclusterer.cluster(X.vectors, assign_clusters=True)
print (assigned_clusters)
# output: [0, 2, 1, 2, 2, 1, 2, 2, 0, 1, 0, 1, 2, 1, 2]