#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:42:18 2019

@author: davidwilbur
"""
import os
import sqlite3
import numpy as np
import pandas as pd
from time import time
from operator import itemgetter
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#%%
# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}\n".format(score.parameters))


#%% Set global path variables
directory = os.path.dirname(os.path.abspath('__file__'))
input_file = os.path.join(directory,'Article Collection')
_dbFile = os.path.join(input_file,'articles_zenodo.db')

#%% Set up our connection to the database
_db = sqlite3.connect(_dbFile)

# Pull the relevant bits of data out of the database and
# store them in a pandas dataframe.
print('Pulling article IDs, leanings, and text from database')
q1 = """SELECT ln.id, ln.bias_final, cn.text, ln.url, ln.pubs_100, ln.source
        FROM train_lean ln, train_content cn
        WHERE cn.`published-at` >= '2009-01-01' AND ln.id == cn.id AND ln.url_keep='1' AND ln.pubs_100='1.0'"""
_df = pd.read_sql(q1, _db, columns=('id', 'lean', 'text'))
_lean = pd.read_sql('select * from train_lean;', _db)


# Pull the relevant bits of data out of the database and
# store them in a pandas dataframe.
print('Pulling article IDs, leanings, and text from database')
q2 = """SELECT ln.id, ln.bias_final, cn.text, ln.url
        FROM test_lean ln, test_content cn
        WHERE cn.`published-at` >= '2009-01-01' AND ln.id == cn.id """
test_df = pd.read_sql(q2, _db, columns=('id', 'lean', 'text'))
test_lean = pd.read_sql('select * from test_lean;', _db)

#%%
#print('Excluding specific publishers due to strongly biasing the model')
#_excludePublishers = ['NULL']
#_excludePublishers = ['apnews', 'foxbusiness']
#_excludeString = '|'.join(_excludePublishers)
#_df = _df[~_df['url'].str.contains(_excludeString)]

#%%
#frames = [_df, test_df]
#new_df = pd.concat(frames, axis=0)

#%%
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(new_df.text,new_df.bias_final, test_size=0.1, random_state=42)

#%%

_df1 = _df.groupby('source').apply(lambda x: x.sample(n=100))

#_excludePublishers = ['NULL']
#_excludePublishers = ['abqjournal', 'washingtontimes']
#_excludeString = '|'.join(_excludePublishers)
#_df1 = _df1[~_df1['source'].str.contains(_excludeString)]

#%%

#text_clf_svm = Pipeline([('vect', CountVectorizer()),
#                         ('tfidf', TfidfTransformer()),
#                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
#                                                   alpha=1e-3, n_iter=10, random_state=42)),])
#text_clf_svm = text_clf_svm.fit(_df.text, _df.bias_final)
#predicted_svm = text_clf_svm.predict(test_df.text)
#np.mean(predicted_svm == test_df.bias_final )


pipe_svm = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', LinearSVC()),])


# build a classifier
#clf = SGDClassifier()

# use a full grid over all parameters
param_grid = {'vect': [CountVectorizer()],
                        "vect__ngram_range": [(1,5)],
                        "vect__max_features": [10000],
              'tfidf': [TfidfTransformer()],
                           "tfidf__sublinear_tf": [False]}
 
# run grid search
grid_search = GridSearchCV(pipe_svm, param_grid=param_grid, scoring='accuracy')
start = time()
grid_search.fit(_df1.text, _df1.bias_final)
 
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.grid_scores_)))
report(grid_search.grid_scores_)

text_clf_svm = Pipeline([('vect', CountVectorizer(analyzer='word',
                                                  binary=False,
                                                  decode_error='strict',
                                                  encoding='utf-8',
                                                  input='content',
                                                  lowercase=True,
                                                  max_df=1.0,
                                                  max_features=10000,
                                                  min_df=1,
                                                  ngram_range=(1, 5),
                                                  preprocessor=None,
                                                  stop_words=None,
                                                  strip_accents=None,
                                                  token_pattern='(?u)\\b\\w\\w+\\b',
                                                  tokenizer=None,
                                                  vocabulary=None)),
                         ('tfidf', TfidfTransformer(norm='l2',
                                                    smooth_idf=True,
                                                    sublinear_tf=False,
                                                    use_idf=True)),
                         ('clf-svm', LinearSVC()),])
    
text_clf_svm = text_clf_svm.fit(_df1.text, _df1.bias_final)
predicted_svm = text_clf_svm.predict(test_df.text)
np.mean(predicted_svm == test_df.bias_final )