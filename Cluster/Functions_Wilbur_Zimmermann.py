#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 19:12:31 2017

@author: davidwilbur
"""


def tokenize_only(text):
    import re
    from nltk.tokenize import sent_tokenize, word_tokenize
    # first tokenize by sentence, then by word to ensure that punctuation is
    # caught as it's own token
    tokens = [word.lower() for sent in sent_tokenize(text)
              for word in word_tokenize(sent)]
    return tokens


def kmeansCalc(df, k):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    """ Use the KMeans algorithm to label the samples into k clusters
    and return them along with the centroids and SSE (inertia). """
    kmeans = KMeans(n_clusters=k, n_jobs=-2, random_state=1).fit(df)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    sil = silhouette_score(df, labels, metric='euclidean')
    return labels, centers, inertia, sil
