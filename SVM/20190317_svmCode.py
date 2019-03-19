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
q1 = """SELECT ln.id, ln.bias_final, cn.text, ln.url
        FROM train_lean ln, train_content cn
        WHERE cn.`published-at` >= '2009-01-01' AND ln.id == cn.id AND ln.url_keep='1'"""
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

frames = [_df, test_df]
new_df = pd.concat(frames, axis=0)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_df.text,new_df.bias_final, test_size=0.2, random_state=42)

#%%

print('Excluding specific publishers due to strongly biasing the model')
_excludePublishers = ['NULL']
_excludePublishers = ['apnews', 'foxbusiness']
_excludeString = '|'.join(_excludePublishers)
_df = _df[~_df['url'].str.contains(_excludeString)]

#%%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

pipe_mnb = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf-mnb', MultinomialNB()),])

# use a full grid over all parameters
param_grid_mnb = {'clf-mnb': [MultinomialNB()],
                            "clf-mnb__alpha": [0, 0.25, 0.50, 1]}
 

grid_search_mnb = GridSearchCV(pipe_mnb, param_grid=param_grid_mnb, scoring='accuracy')
start = time()
grid_search_mnb.fit(_df.text, _df.bias_final)

 
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search_mnb.grid_scores_)))
report(grid_search_mnb.grid_scores_)


text_clf_mnb = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf-mnb', MultinomialNB()),])
text_clf_mnb = text_clf_mnb.fit(X_train, y_train)
predicted_mnb = text_clf_mnb.predict(X_test)
np.mean(predicted_mnb == y_test)

#text_clf = text_clf.fit(_df.text, _df.bias_final)

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
                            "clf-svm__max_iter": [1, 5, 10],
                            "clf-svm__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                            "clf-svm__penalty": ["none", "l1", "l2"],
                            "clf-svm__n_jobs": [-1]}
 
# run grid search
grid_search = GridSearchCV(pipe_svm, param_grid=param_grid, scoring='accuracy')
start = time()
grid_search.fit(_df.text, _df.bias_final)
 
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.grid_scores_)))
report(grid_search.grid_scores_)

text_clf_svm = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(alpha=0.0001, 
                           average=False, 
                           class_weight=None, 
                           epsilon=0.1,
                           eta0=0.0,
                           fit_intercept=True,
                           l1_ratio=0.15,
                           learning_rate='optimal',
                           loss='hinge',
                           max_iter=None,
                           n_iter=100,
                           n_jobs=1,
                           penalty='none',
                           power_t=0.5,
                           random_state=None,
                           shuffle=True,
                           tol=None,
                           verbose=0,
                           warm_start=False)),])
text_clf_svm = text_clf_svm.fit(_df.text, _df.bias_final)
predicted_svm = text_clf_svm.predict(test_df.text)
np.mean(predicted_svm == test_df.bias_final )