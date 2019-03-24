# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 07:59:10 2019

@author: timothy.whalen
"""

import os
import sqlite3
import pandas as pd
import numpy as np

from nltk.corpus import stopwords

#%% Set global path variables
directory = os.path.dirname(os.path.abspath('__file__'))
#input_file = os.path.join(directory,'Article Collection')
#_dbFile = os.path.join(input_file,'articles_zenodo.db')
_dbFile = r'%s/articles_zenodo.db' % directory.replace("Similar Documents", "Article Collection")

#%% Set up our connection to the database
_db = sqlite3.connect(_dbFile)

# Pull the relevant bits of data out of the database and
# store them in a pandas dataframe.
print('Pulling article IDs, leanings, and text from database')
q1 = """SELECT ln.id, ln.bias_final, cn.text
        FROM test_lean ln, test_content cn
        WHERE cn.`published-at` >= '2009-01-01' AND ln.id == cn.id AND ln.url_keep='1'"""
_df = pd.read_sql(q1, _db, columns=('id', 'lean', 'text'))
_lean = pd.read_sql('select * from test_lean;', _db)

#%% Source to article ID mapping
print('Adding source column to leaning')
_lean['source'] = _lean.url.str.split('//', expand=True, n=1)[1].str.split('.', expand=True, n=1)[0]

#%%Create matrix of TFIDF values
print('Tokenizing Articles')
from nltk.tokenize import word_tokenize
raw_documents = _df.text
gen_docs = [[w.lower() for w in word_tokenize(text)] for text in raw_documents]

#%% Get all documents into a single list by publisher
print('Combine all articles by publisher')
_pubs_dic = {}
_words = []
gi = 0
for i1 in _df.id:
    s = _lean.loc[_lean.id == i1, 'source']
    s = s[s.index[0]]
    print('%s: %s' % (str(gi), s))
    _words.extend(gen_docs[gi])
    if s in _pubs_dic.keys():
        _pubs_dic[s].extend(gen_docs[gi])
    else:
        _pubs_dic[s] = gen_docs[gi]
    gi += 1

#%%
print('Word counts (using collections)')
_pubs_wc = {}
from collections import Counter

for p, ws in _pubs_dic.items():
    print(p)
    _pubs_wc[p] = dict(Counter(ws))

print('create dataframe of WCs')
_all_words = dict(Counter(_words))
_df_out = pd.DataFrame(data={'word': list(_all_words.keys()), 'all': list(_all_words.values())})
for p, wc in _pubs_wc.items():
    print(p)
    _tmp_df = pd.DataFrame(data={'word': list(wc.keys()), p: list(wc.values())})
    _df_out = _df_out.join(_tmp_df.set_index('word'), on='word')
print('writing to database')
_df_out.to_sql(name='test_publisher_WC', con=_db, if_exists='replace', index=False)
