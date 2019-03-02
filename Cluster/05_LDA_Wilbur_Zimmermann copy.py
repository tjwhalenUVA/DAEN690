#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:49:39 2017

@author: davidwilbur
"""

# %%###########################################################################
###############################################################################
#
# Cluster Based Analysis of Corporate Resumes (Resume Topic Modeling)
# Submitted for Successful Completion of CS584
# Authored By: David Wilbur and Russell Zimmermann
#
# CS584 Theory and Applications of Data Mining
# Daniel Barbará, Ph.D.
# George Mason University
# Volgenau School of Engineering
# Data Analytics Engineering
#
###############################################################################
#
# LDA: This script reads in the conditioned set of resumes from the previous
#         script, transforms each feature from token space to Term Frequency
#         Inverse Document Frequency (TFIDF) space.  The TFIDF matrix is fed to
#         the KMeans algorithm and finally a 3D plot of the clusters and a
#         partitioning of the resumes into their respective groups is displayed
#
#   Output:
#     *** 3D Plot of the original dataset (SVD-tSNE)
#     *** 3D Plot of the data after KMeans processing (colored by cluster)
#     *** A bar chart showing the number of the total resumes in each cluster
#
#   This script takes about XX minutes to run.
#
###############################################################################
import os
import re
import lda
import time
import logging
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from Functions_Wilbur_Zimmermann import tokenize_only
from sklearn.feature_extraction.text import CountVectorizer
# Plotly plotting packages
import plotly
import plotly.graph_objs as go

# %%
directory = os.path.dirname('__file__')
input_file = os.path.join(directory, '..', 'script_output')
output = os.path.join(directory, '..', 'script_output')
# %%
print('\nLDA Step 1: Read in conditioned resumes from file ',
      os.path.realpath(input_file))

r = re.compile(r'\d{8}-\d{6}\_conditioned_resumes_FINAL\.csv$')
latest_file = max(filter(r.search, os.listdir(input_file)))
print('\nLDA Step 1: file name is {}'.format(latest_file))
resumes_df = pd.read_csv(os.path.join(input_file, latest_file))
# %%

logging.getLogger("lda").setLevel(logging.WARNING)
# %%

cvectorizer = CountVectorizer(min_df=5, max_features=10000,
                              tokenizer=tokenize_only, ngram_range=(1, 2))
cvz = cvectorizer.fit_transform(resumes_df['Resumes'])

# %%
# This takes a little bit of time to run.
begin_time = time.time()
start_time = time.strftime('%H:%M:%S')
print('\nStart time: {}  (should take about 5 minutes to run)'
      .format(start_time))

n_topics = 25
n_iter = 500
lda_model = lda.LDA(n_topics=n_topics, n_iter=n_iter, random_state=1)
X_topics = lda_model.fit_transform(cvz)

n_top_words = 10
topic_summaries = []

topic_word = lda_model.topic_word_  # get the topic words
vocab = cvectorizer.get_feature_names()
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

duration = round((time.time() - begin_time)/60, 2)
print('The tsne modeler took {} minutes to run.'.format(duration))

# %%

threshold = 0.3  # Consider altering this to understand its impact on the plot
_idx = np.amax(X_topics, axis=1) > threshold  # idx of news that > threshold
_topics = X_topics[_idx]
_FileNames = resumes_df.FileNames[_idx]
_lda_doc_topic = lda_model.doc_topic_[_idx]

num_example = len(_topics)

tsne_model = TSNE(n_components=3, verbose=1, random_state=1, angle=.99,
                  init='pca')
tsne_lda = tsne_model.fit_transform(_topics[:num_example])

# find the most probable topic for each news
_lda_keys = []
for i in range(_topics.shape[0]):
    _lda_keys += _topics[i].argmax(),

# %%

colormap = np.array(['#6d8dca', '#69de53', '#723bca', '#c3e14c', '#c84dc9',
                     '#68af4e', '#6e6cd5', '#e3be38', '#4e2d7c', '#5fdfa8',
                     '#d34690', '#3f6d31', '#d44427', '#7fcdd8', '#cb4053',
                     '#5e9981', '#803a62', '#9b9e39', '#c88cca', '#e1c37b',
                     '#34223b', '#bdd8a3', '#6e3326', '#cfbdce', '#d07d3c',
                     '#52697d', '#7d6d33', '#d27c88', '#36422b', '#b68f79'])

# %%
# !!! We need to get the mouse over working with this.
# Need to plot the code above.
tsne_lda_df = pd.DataFrame(tsne_lda, columns=['x', 'y', 'z'])
# tsne_lda_df['Resumes'] = tsne_lda_df['Resumes']
x = tsne_lda_df.x
y = tsne_lda_df.y
z = tsne_lda_df.z

tsne_lda_df['topic'] = _lda_keys
tsne_lda_df['topic'] = tsne_lda_df['topic'].map(int)
topic_string = []
for q in range(len(tsne_lda_df)):
    topic_string.append(str(tsne_lda_df.topic[q]))

tsne_lda_df['str_topic'] = topic_string

tsne_lda_df['FileNames'] = _FileNames
tsne_lda_df['labels_file'] = tsne_lda_df.str_topic + ' | ' +\
                              tsne_lda_df.FileNames


#resumesPlus_df['labels'] = kmeans_clusters
#resumesPlus_df['labels'] = resumesPlus_df['labels'].apply(str)


hoverTxt_ar = tsne_lda_df['labels_file']

# hoverTxt_ar = something  # !!! Need to find hover text

trace1 =\
    go.Scatter3d(x=x, y=y, z=z,
                 mode='markers',
                 marker=dict(size=3, color=colormap[_lda_keys],
                             line=dict(
                                       color='rgb(0,0,0)',
                                       width=0.2),
                             opacity=1,),
                 text=hoverTxt_ar)

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

# %%
# This code is from Ahmed Besbe's website.
# It currently does not work.

resumes_df['tokens'] = resumes_df.Resumes.map(tokenize_only)
tsne_lda_df['len_docs'] = resumes_df['tokens'].map(len)


def prepareLDAData():
    data = {
        'vocab': vocab,
        'doc_topic_dists': _lda_doc_topic,
        'doc_lengths': list(tsne_lda_df['len_docs']),
        'term_frequency': cvectorizer.vocabulary_,
        'topic_term_dists': lda_model.components_
    }
    return data


ldadata = prepareLDAData()

import pyLDAvis

prepared_data = pyLDAvis.prepare(**ldadata)

pyLDAvis.save_html(prepared_data, './pyldadavis.html')
