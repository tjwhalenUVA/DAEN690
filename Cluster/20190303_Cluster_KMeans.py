#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:31:26 2017

@author: davidwilbur
"""
# %%
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
# KMeans: This script reads in the conditioned set of resumes from 03
#         Condition script, transforms each feature from token space to Term
#         Frequency Inverse Document Frequency (TFIDF) space.  The TFIDF
#         matrix is fed to the KMeans algorithm and finally a 3D plot of the
#         clusters and a partitioning of the resumes into their respective
#         groups is displayed
#
#   Output:
#     *** An elbow plot showing SSE as a function of K
#     *** 3D Plot of the data after KMeans processing (colored by cluster)
#     *** A bar chart showing the number of the total resumes in each cluster
#
#   This script takes about XX minutes to run
#   The code to create the KMeans elbow curve is commented out.  If you
#   want to generate a new elbow curve it will have to be uncommeted and
#   re-run.
#
###############################################################################
# %%
# Remove variables, ensures we are not using old values.
print('\n KMeans CLustering: Cleaning up variable names')
from IPython import get_ipython
get_ipython().magic('reset -sf')
# %%
# Required Python packages
print('\nKMeans Clustering: Importing required python packages')
import os
import re
import time
import plotly
import warnings
import collections
import sqlite3
import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from plotly.grid_objs import Grid, Column
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from Functions_Wilbur_Zimmermann import tokenize_only, kmeansCalc

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
# %%
# min_df is minimum number of documents that contain a term t.
#
# max_features is maximum number of unique tokens (across documents) that
# we'd consider TfidfVectorizer preprocesses the descriptions using the
# tokenizer we defined in the functions script.

print('\nKMeans Clustering: This vectorizer\'s run time is ~30 secs')
vectorizer = TfidfVectorizer(min_df=10, max_features=10000,
                             tokenizer=tokenize_only, ngram_range=(1, 2))
vz = vectorizer.fit_transform(list(_df['text']))

# %%
# This code helps us understand the tokens and their TFIDF values.

tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
tfidf = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf), orient='index')
tfidf.columns = ['tfidf']

tfidf.sort_values(by=['tfidf'], ascending=True).head(30)

tfidf.sort_values(by=['tfidf'], ascending=False).head(30)

# %%
# Plot the histogram and box plot of the TFIDF Score.
plt.rcParams['figure.figsize'] = (10, 10)
ax1 = plt.subplot(211)
plt.title('TFIDF Histogram')
plt.hist(tfidf.tfidf, bins=100, color='gray')
plt.xticks(np.arange(0, 9, 1.0), rotation='vertical')
plt.xlabel('TFIDF Score')
plt.ylabel('Number of Tokens')
plt.setp(ax1.get_xticklabels(), fontsize=10)

minTcount = tfidf.tfidf.min()
firstQTcount = tfidf.tfidf.quantile(0.25)
medTcount = tfidf.tfidf.median()
meanTcount = tfidf.tfidf.mean()
thirdQTcount = tfidf.tfidf.quantile(0.75)
maxTcount = tfidf.tfidf.max()

plt.axvline(minTcount, color='r', linestyle='--', linewidth=1)
plt.axvline(firstQTcount, color='c', linestyle='--', linewidth=1)
plt.axvline(medTcount, color='y', linestyle='--', linewidth=1)
plt.axvline(meanTcount, color='g', linestyle='--', linewidth=2)
plt.axvline(thirdQTcount, color='c', linestyle='--', linewidth=1)
plt.axvline(maxTcount, color='r', linestyle='--', linewidth=1)

r1_line = mlines.Line2D([], [], color='r', linestyle='--', linewidth=1,
                        label='min')
c1_line = mlines.Line2D([], [], color='c', linestyle='--', linewidth=1,
                        label='first quartile')
y_line = mlines.Line2D([], [], color='y', linestyle='--', linewidth=1,
                       label='median')
g_line = mlines.Line2D([], [], color='g', linestyle='--', linewidth=2,
                       label='mean')
c2_line = mlines.Line2D([], [], color='c', linestyle='--', linewidth=1,
                        label='third quartile')
r2_line = mlines.Line2D([], [], color='r', linestyle='--', linewidth=1,
                        label='max')
plt.legend(handles=[r1_line, c1_line, g_line, y_line, c2_line, r2_line], loc=1)
plt.tight_layout()

# Box plot of the TFIDF count.
ax2 = plt.subplot(212, sharex=ax1)
plt.title('TFIDF Boxplot')
plt.boxplot(tfidf.tfidf, 0, 'b+', 0)
plt.xticks(rotation='vertical')
plt.xlabel('TFIDF Score')
plt.ylabel('Number of Tokens')
plt.yticks([])
plt.setp(ax2.get_xticklabels(), fontsize=10)

plt.axvline(minTcount, color='r', linestyle='--', linewidth=1)
plt.axvline(firstQTcount, color='c', linestyle='--', linewidth=1)
plt.axvline(medTcount, color='y', linestyle='--', linewidth=1)
plt.axvline(meanTcount, color='g', linestyle='--', linewidth=2)
plt.axvline(thirdQTcount, color='c', linestyle='--', linewidth=1)
plt.axvline(maxTcount, color='r', linestyle='--', linewidth=1)

r1_line = mlines.Line2D([], [], color='r', linestyle='--', linewidth=1,
                        label='min')
c1_line = mlines.Line2D([], [], color='c', linestyle='--', linewidth=1,
                        label='first quartile')
y_line = mlines.Line2D([], [], color='y', linestyle='--', linewidth=1,
                       label='median')
g_line = mlines.Line2D([], [], color='g', linestyle='--', linewidth=2,
                       label='mean')
c2_line = mlines.Line2D([], [], color='c', linestyle='--', linewidth=1,
                        label='third quartile')
r2_line = mlines.Line2D([], [], color='r', linestyle='--', linewidth=1,
                        label='max')
plt.legend(handles=[r1_line, c1_line, g_line, y_line, c2_line, r2_line], loc=1)
plt.tight_layout()

timeStr = time.strftime('%Y%m%d-%H%M%S')
combined_token_count_png =\
   os.path.join(directory, '..', 'script_output',
                '%s_tfidf_hist_boxplot.png' % timeStr)
plt.savefig(combined_token_count_png, dpi=100)

# %%
# Use the kmeans function to create and capture the cluster information
# for k equal 3 through 47

begin_time = time.time()
start_time = time.strftime('%H:%M:%S')
print('\n\nStart time: {}  (should take about x mins )'.format(start_time))

k_first, k_last, step = 3, 49, 2

print('\nCalc clusters, centroids, SSE, & silhouette for '
      'k = {} through {} by {} ...\n'.format(k_first, k_last - step, step))
clusters_list = []
centroids_list = []
inertia_list = []
silhouette_list = []
topics_list = []
listIndex = 0
for i in range(k_first, k_last, step):
    print('  Calculating kmeans for {} clusters ...'.format(i))
    classes_ar, centroids_ar, inertia, sil = kmeansCalc(vz, i)
    clusters_list.append(classes_ar)
    centroids_list.append(centroids_ar)
    inertia_list.append(inertia)
    silhouette_list.append(sil)

    sorted_centroidsLOOP = centroids_ar.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for num_clusters in range(i):
        print("Cluster %d:" % num_clusters)
        aux = ''
        for j in sorted_centroidsLOOP[num_clusters, :10]:
            aux += terms[j] + ' | '
        print(aux)
        print()
        topics_row = (i, num_clusters, aux)
        print('\n', topics_row)
        topics_list.append(topics_row)

duration = round((time.time() - begin_time)/60, 2)
print('The KMeans modeler took {} minutes to run.'.format(duration))

# Save the lists of KMeans data collected
timeStr = time.strftime('%Y%m%d-%H%M%S')  # Set time string for filenames

clustAll_csv =\
    os.path.join(output_km, '{}_km_clustersAll.csv'.format(timeStr, i))
centAll_csv =\
    os.path.join(output_km, '{}_km_centroidsAll.csv'.format(timeStr, i))
inertiaAll_csv =\
    os.path.join(output_km, '{}_km_inertiaAll.csv'.format(timeStr, i))
silAll_csv =\
    os.path.join(output_km, '{}_km_silhouetteAll.csv'.format(timeStr, i))
topicsAll_csv =\
    os.path.join(output_km, '{}_km_topicsAll.csv'.format(timeStr, i))

np.savetxt(clustAll_csv, np.array(clusters_list), fmt='%s', delimiter=',')
np.savetxt(centAll_csv, np.array(centroids_list), fmt='%s', delimiter=',')
np.savetxt(inertiaAll_csv, np.array(inertia_list), fmt='%s', delimiter=',')
np.savetxt(silAll_csv, np.array(silhouette_list), fmt='%s', delimiter=',')
np.savetxt(topicsAll_csv, np.array(topics_list), fmt='%s', delimiter=',')

topics_df = pd.DataFrame(topics_list, columns=('k', 'cluster', 'topics'))
topics_pv = topics_df.pivot(index='cluster', columns='k', values='topics')
topicsAllPivot_csv =\
    os.path.join(output_km, '{}_km_topicsAll_pivot.csv'.format(timeStr, i))
topics_pv.to_csv(topicsAllPivot_csv)

# %%
# Plt SSE as a function of K

print('Plotting # of clusters (K) vs SSE to find the elbow ...')
x = [i for i in range(3, 51, 2)]
y = inertia_list

plt.title('Number of Clusters (K) versus Sum of Squared Errors (SSE)')
plt.xlabel('Values of K')
plt.ylabel('Values of SSE')
plt.scatter(x, y)

timeStr = time.strftime('%Y%m%d-%H%M%S')
sse_png =\
   os.path.join(directory, '..', 'script_output',
                '%s_SSE_versus_K.png' % timeStr)
plt.savefig(sse_png, dpi=100)

plt.show()

# %%
# Plt Silhouette coefficient as a function of K
# !!!!! Run this on Sunday
print('Plotting Silhouette coefficient vs # of clusters (K) ...')
x = [i for i in range(3, 49, 2)]
y = silhouette_list

plt.title('Silhouette coefficient versus Number of Clusters (K)')
plt.xlabel('Values of K')
plt.ylabel('Silhouette Coefficient')
plt.scatter(x, y)

timeStr = time.strftime('%Y%m%d-%H%M%S')
sil_png = os.path.join(directory, '..', 'script_output',
                       '%s_sil_versus_K.png' % timeStr)
plt.savefig(sil_png, dpi=100)

plt.show()

# %%
# KMeans Euclidean for the chose optimal number of clusters based on our
# topics analysis.
print('\nKMeans Clustering: Running KMeans for the optimal value of k.')

begin_time = time.time()
start_time = time.strftime('%H:%M:%S')
print('\n\nStart time: {}  (should take about 14 mins)'.format(start_time))

warnings.filterwarnings("ignore", category=DeprecationWarning)
# Look for the elbow in the curve, run the best and visually inspect.
num_clusters = 25  # 25 is the best value of k based on SSE and analysis
kmeans_model = KMeans(n_clusters=num_clusters, n_jobs=-2,
                      random_state=1).fit(vz)
kmeans = kmeans_model.fit(vz)
kmeans_clusters = kmeans.predict(vz)
kmeans_distances = kmeans.transform(vz)

sorted_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
# centroids = kmeans.cluster_centers_.argsort()
terms = vectorizer.get_feature_names()
for i in range(num_clusters):
    print("Cluster %d:" % i)
    aux = ''
    for j in sorted_centroids[i, :10]:
        print('j is ', j)
        aux += terms[j] + ' | '
    print(aux)
    print()

duration = round((time.time() - begin_time)/60, 2)
print('The KMeans modeler took {} minutes to run.'.format(duration))

# %%
# Convert the KMeans clusters to three dimensions for plotting.
print('\nKMeans Clustering: Decreasing the dimensionality to 3.  Reduced '
      'dimensionality will allow us to visualize clusters of resumes.')

begin_time = time.time()
start_time = time.strftime('%H:%M:%S')
print('\n\nStart time: {}  (should take about 8 mins)'.format(start_time))

tsne_model3D = TSNE(n_components=3, verbose=1, random_state=0)
tsne_kmeans = tsne_model3D.fit_transform(kmeans_distances)

duration = round((time.time() - begin_time)/60, 2)
print('The tsne modeler took {} minutes to run.'.format(duration))

# %%
# Sets colormap to visualize different clusters of the resume dataset.
# To have unique colors per cluster, number of colors should equal or
# be greater than the number of clusters.

colormap = np.array(['#6d8dca', '#69de53', '#723bca', '#c3e14c', '#c84dc9',
                     '#68af4e', '#6e6cd5', '#e3be38', '#4e2d7c', '#5fdfa8',
                     '#d34690', '#3f6d31', '#d44427', '#7fcdd8', '#cb4053',
                     '#5e9981', '#803a62', '#9b9e39', '#c88cca', '#e1c37b',
                     '#34223b', '#bdd8a3', '#6e3326', '#cfbdce', '#d07d3c',
                     '#52697d', '#7d6d33', '#d27c88', '#36422b', '#b68f79'])

# %%
# Code to plot the 3 dimensions of the KMeans output.
print('\nKMeans Clustering: Plotting clusters.')
tsne_kmeans_df = pd.DataFrame(tsne_kmeans, columns=['x', 'y', 'z'])
tsne_kmeans_df['Resumes'] = resumes_df.Resumes

x = tsne_kmeans_df.x
y = tsne_kmeans_df.y
z = tsne_kmeans_df.z
# hoverTxt_clust_ar = np.array(range(0, len(x), 1))  # Set to row numbers
# hoverTxt_ar = resumes_df.FileNames, '+'(kmeans.labels_)  # Set to file names

resumesPlus_df = resumes_df.copy()
resumesPlus_df['labels'] = kmeans_clusters
resumesPlus_df['labels'] = resumesPlus_df['labels'].apply(str)
resumesPlus_df['labels_file'] = resumesPlus_df['labels'] + ' | ' +\
                                resumesPlus_df.FileNames

hoverTxt_ar = resumesPlus_df['labels_file']

# we want
'5, Report....'

grid = Grid([x, y, z])

trace1 =\
    go.Scatter3d(x=x, y=y, z=z,
                 mode='markers',
                 marker=dict(size=3,
                             color=colormap[kmeans_clusters],
                             line=dict(color='rgb(0,0,0)', width=0.2),
                             opacity=1,),
                 text=hoverTxt_ar)

data = [trace1]
layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0),
                   updatemenus=[dict(type='buttons', showactive=False,
                                y=1, x=-0.05, xanchor='right',
                                yanchor='top', pad=dict(t=0, r=10),)])

fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig)

# !!! Manually save the html plot

# %%
# Create a histogram of the number of resumes per cluster for the optimal
# number of KMeans clusters.

clusters_df = pd.DataFrame(clusters_list)

# Isolate a list of clusters for a particular number of clusters
clusters_k25 = clusters_df.loc[5].values.tolist()
counter = collections.Counter(clusters_k25)

y = []
x = []
for key in counter.keys():
    y.append(key)
    x.append(counter[key])

clustHist_df = pd.DataFrame(x, y)
clustHist_df = clustHist_df.sort_index()
resCount = clustHist_df[0].tolist()

plt.figure(figsize=(10, 10))
fig, ax = plt.subplots()
plt.title('Number of Resumes per Cluster')
plt.xlabel('Number of Resumes')
plt.ylabel('Cluster Identifier')
plt.yticks(np.arange(0, len(resCount), 1.0))
for i, v in enumerate(resCount):
    ax.text(v, i, str(v))
plt.barh(range(len(resCount)), resCount)
