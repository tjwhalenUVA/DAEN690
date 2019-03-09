# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 17:33:25 2019

@author: Timothy.Whalen
"""
import os
import sqlite3
import pandas as pd
import numpy as np

from nltk.corpus import stopwords

#%% Set global path variables
directory = os.path.dirname('__file__')
input_file = os.path.join(directory,'..','Article Collection')
_dbFile = os.path.join(input_file,'articles_zenodo.db')

#%% Set up our connection to the database
_db = sqlite3.connect(_dbFile)
_cursor = _db.cursor()

# Pull the relevant bits of data out of the database and
# store them in a pandas dataframe.
print('Pulling article IDs, leanings, and text from database')
_cursor.execute("SELECT ln.id, ln.bias_final, cn.text " +
                "FROM train_lean ln, train_content cn " +
                "WHERE cn.`published-at` >= '2018-01-01' AND ln.id == cn.id AND ln.url_keep='1'")
_df = pd.DataFrame(_cursor.fetchall(), columns=('id', 'lean', 'text'))

#%%Create matrix of TFIDF values
print('Tokenizing Articles')
import gensim
from nltk.tokenize import word_tokenize
raw_documents = _df.text
gen_docs = [[w.lower() for w in word_tokenize(text)] for text in raw_documents]
#%%
print('Creating Dictionary of Wrods')
dictionary = gensim.corpora.Dictionary(gen_docs)
print("Number of words in dictionary:",len(dictionary))
#%%
print('Creating Corpus')
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
#%%
print('TF-IDF')
tf_idf = gensim.models.TfidfModel(corpus)
#%%
print('Calculate Similarity')
sims = gensim.similarities.Similarity(directory,tf_idf[corpus], num_features=len(dictionary))
print('Clean Up Memory')
import gc
del corpus, raw_documents, gen_docs
gc.collect()
#%%
s = 5
doc_ids = list()
doc_sims = list()
sim_dis = list()
for d in _df.index:
#    d = 0
    print("Working doc " + str(d))
    new_doc = _df.text[d]
    query_doc = [w.lower() for w in word_tokenize(new_doc)]
    query_doc_bow = dictionary.doc2bow(query_doc)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    n_d_s = sims[query_doc_tf_idf]
    #Get top 5 similar articles
    top_s_idx = np.argsort(n_d_s)[-s-1:][::-1][1:]
    top_s_values = [n_d_s[i] for i in top_s_idx]
    doc_ids.append(_df.iloc[d].id)
    doc_sims.append('_'.join(str(did) for did in _df.iloc[top_s_idx].id.tolist()))
    sim_dis.append('_'.join(str(did) for did in top_s_values))


_sim_df = pd.DataFrame({'id': doc_ids, 'similar': doc_sims, 'distance': sim_dis})
_sim_df.to_sql(name='similarity', con=_db, if_exists='replace')