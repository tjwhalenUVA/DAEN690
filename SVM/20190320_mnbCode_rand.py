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
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
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

# %% Set global path variables

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

#%%

_df = _df.sample(n=516)
test_df = test_df.sample(n=129)


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

#_df1 = _df.groupby('source').apply(lambda x: x.sample(n=100))

_excludePublishers = ['NULL']
#_excludePublishers = ['abqjournal', 'foxbusiness', 'apnews', 'motherjones', 'counterpunch', 'newsline', 'pri']
_excludePublishers = ['foxbusiness']
_excludeString = '|'.join(_excludePublishers)
_df = _df[~_df['source'].str.contains(_excludeString)]

_excludePublishersTest = ['NULL']
#_excludePublishersTest = ['themoderatevoice', 'vox', 'grist', 'cfr', 'texastribune']
_excludePublishersTest = ['cfr']
_excludeStringTest = '|'.join(_excludePublishersTest)
test_df = test_df[~test_df['url'].str.contains(_excludeStringTest)]


#%%
_excludePublishers = ['NULL']
_excludePublishers = ['left-center', 'right-center', 'least']
_excludeString = '|'.join(_excludePublishers)
_df = _df[~_df['bias_final'].str.contains(_excludeString)]

_excludePublishersTest = ['NULL']
_excludePublishersTest = ['left-center', 'right-center', 'least']
_excludeStringTest = '|'.join(_excludePublishersTest)
test_df = test_df[~test_df['bias_final'].str.contains(_excludeStringTest)]
#%%

pipe_mnb = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf-mnb', MultinomialNB()),])

# use a full grid over all parameters
param_grid_mnb = {'clf-mnb': [MultinomialNB()],
                            "clf-mnb__alpha": [.001, 0.25, 0.50, 1],
                  'vect': [CountVectorizer()],
                          "vect__ngram_range": [(0,1),(1,2),(1,3),(1,5),(1,10)],
                          "vect__max_features": [100,500,1000,5000,10000,100000],
                  'tfidf': [TfidfTransformer()],
                           "tfidf__norm": ['l1','l2',None],
                           "tfidf__sublinear_tf": [True,False]}
 

grid_search_mnb = GridSearchCV(pipe_mnb, param_grid=param_grid_mnb, scoring='accuracy')
start = time()
grid_search_mnb.fit(_df1.text, _df1.bias_final)

 
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search_mnb.grid_scores_)))
report(grid_search_mnb.grid_scores_)


text_clf_mnb = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf-mnb', MultinomialNB()),])
text_clf_mnb = text_clf_mnb.fit(_df.text, _df.bias_final)
predicted_mnb = text_clf_mnb.predict(test_df.text)
np.mean(predicted_mnb == test_df.bias_final)

from sklearn.metrics import confusion_matrix
y_true = test_df.bias_final
y_pred = predicted_mnb
confusion_matrix(y_true, y_pred)

class_names =['left','right']

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = ['left', 'right']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_true, y_pred, classes=class_names,
                      title='Confusion matrix')

# Plot normalized confusion matrix
#plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
#                     title='Normalized confusion matrix')

plt.show()
#text_clf = text_clf.fit(_df.text, _df.bias_final)

