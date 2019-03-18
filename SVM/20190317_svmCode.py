#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:42:18 2019

@author: davidwilbur
"""


import os
import sqlite3
import pandas as pd
import numpy as np

from nltk.corpus import stopwords

#%% Set global path variables
directory = os.path.dirname(os.path.abspath('__file__'))
input_file = os.path.join(directory,'Article Collection')
_dbFile = os.path.join(input_file,'articles_zenodo.db')

#%% Set up our connection to the database
_db = sqlite3.connect(_dbFile)

# Pull the relevant bits of data out of the database and
# store them in a pandas dataframe.
print('Pulling article IDs, leanings, and text from database')
q1 = """SELECT ln.id, ln.bias_final, cn.text
        FROM train_lean ln, train_content cn
        WHERE cn.`published-at` >= '2009-01-01' AND ln.id == cn.id AND ln.url_keep='1'"""
_df = pd.read_sql(q1, _db, columns=('id', 'lean', 'text'))
_lean = pd.read_sql('select * from train_lean;', _db)


# Pull the relevant bits of data out of the database and
# store them in a pandas dataframe.
print('Pulling article IDs, leanings, and text from database')
q2 = """SELECT ln.id, ln.bias_final, cn.text
        FROM test_lean ln, test_content cn
        WHERE cn.`published-at` >= '2009-01-01' AND ln.id == cn.id """
test_df = pd.read_sql(q2, _db, columns=('id', 'lean', 'text'))
test_lean = pd.read_sql('select * from test_lean;', _db)

#%% Source to article ID mapping

_lean['source'] = _lean.url.str.split('//', expand=True, n=1)[1].str.split('.', expand=True, n=1)[0]
test_lean['source'] = test_lean.url.str.split('//', expand=True, n=1)[1].str.split('.', expand=True, n=1)[0]

#%%

#from sklearn.model_selection import train_test_split
#xTrain, xTest, yTrain, yTest = train_test_split(_df.text, _df.bias_final, test_size = 0.2, random_state = 0)

#%%

#from sklearn.feature_extraction.text import CountVectorizer
#count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform(_df.text)
#X_train_counts.shape
# %%

#from sklearn.feature_extraction.text import TfidfTransformer
#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#X_train_tfidf.shape

#%%

#from sklearn.naive_bayes import MultinomialNB
#clf = MultinomialNB().fit(X_train_tfidf, yTrain)

#%%

from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),])
text_clf = text_clf.fit(_df.text, _df.bias_final)

#%%

import numpy as np
predicted = text_clf.predict(test_df.text)
np.mean(predicted == test_df.bias_final)

#%%
import numpy as np
from time import time
from operator import itemgetter
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}\n".format(score.parameters))



#text_clf_svm = Pipeline([('vect', CountVectorizer()),
#                         ('tfidf', TfidfTransformer()),
#                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
#                                                   alpha=1e-3, n_iter=10, random_state=42)),])
#text_clf_svm = text_clf_svm.fit(_df.text, _df.bias_final)
#predicted_svm = text_clf_svm.predict(test_df.text)
#np.mean(predicted_svm == test_df.bias_final )


pipe_svm = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier()),])


# build a classifier
#clf = SGDClassifier()

# use a full grid over all parameters
param_grid = {'clf-svm': [SGDClassifier()],
                            "clf-svm__n_iter": [1, 5, 10],
                            "clf-svm__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                            "clf-svm__penalty": ["none", "l1", "l2"]}
 
# run grid search
grid_search = GridSearchCV(pipe_svm, param_grid=param_grid, scoring='accuracy')
start = time()
grid_search.fit(_df.text, _df.bias_final)
 
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.grid_scores_)))
report(grid_search.grid_scores_)