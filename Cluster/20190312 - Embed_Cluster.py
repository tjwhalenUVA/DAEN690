#! /Users/dpbrinegar/anaconda3/envs/gitHub/bin/python
# -*- coding: utf-8 -*-
#
# Author:  Paul M. Brinegar, II
#
# Date Created:  20190224
#
# CNN.py - Code for running a CNN (Convolutional Neural Net)
#
# This code should extract the article ID, leaning, and text from our SQLite database,
# convert the text for each article into a series of embedded word vectors (padding
# as necessary), and feed the results into a convolutional neural net.  We will be using
# the GloVe dataset as our source for word embedding vectors.
#
# Understanding of word embeddings and example code found at:
# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
#

# Import all the packages!
print('Importing packages')
import os
import sqlite3

from pandas import DataFrame
from tensorflow import keras
import numpy as np
    
# %%
# Set global path variables
directory = os.path.dirname('__file__')
input_file = os.path.join(directory,'..','Article Collection')
_dbFile = os.path.join(input_file,'articles_zenodo.db')

# %%
    # Import the relevant bits of data out of the database and store them in a pandas dataframe.
    #
    # Set up our connection to the database
_db = sqlite3.connect(_dbFile)
_cursor = _db.cursor()

    # Load the data from the database
print('Pulling article IDs, leanings, and text from database')
_cursor.execute("SELECT ln.id, ln.bias_final, cn.text " +
                "FROM train_lean ln, train_content cn " +
                "WHERE cn.`published-at` >= '2016-01-01' AND ln.id == cn.id AND ln.url_keep='1'")
_df = DataFrame(_cursor.fetchall(), columns=('id', 'lean', 'text'))
_db.close()

# %%
    # Import GloVe global word vectors from input file and build word vector dictionary
    # This creates a {word:vector} dictionary for every word in the GloVe input file.
    # The variable _dimensions can be 50, 100, 0r 300, and will read from the corresponding
    # GloVe input file.
    #
print('Loading word embedding vectors from GloVe 6B data')

# Initialize our word embedding dictionary
_embeddingDict = {}
_gloveFile = 'glove.6B.50d.txt'
# Compute the number of dimensions in our word vectors from the file name.
_dimensions = int(_gloveFile[-8:-5].strip('.'))

    # Load the word vector data
with open(_gloveFile) as _gf:
    for _row in _gf:
        _junk = _row.split()
        _embeddingDict[_junk[0]] = np.array(_junk[1:], dtype='float32')


    # Tokenize the articles to create a {word:index} dictionary.  The variable _vocabSize
    # can be any size we wish.  We set a token for out-of-vocabulary words.
    #
_vocabSize = 1000
_captureFraction = 0.95
print('Tokenizing corpus/vocabulary')
t = keras.preprocessing.text.Tokenizer(oov_token="<OOV>", num_words=_vocabSize)
t.fit_on_texts(_df.text)


    # The tokenizer in keras fails to create its {word:index} dictionary properly.  Words
    # with indices larger than the vocabulary size are retained, so we must rebuild the
    # dictionary to truncate to the specified vocabulary size.
    #
print('Correcting improper OOV handling in keras')
t.word_index = {word: index for word, index in t.word_index.items() if index <= _vocabSize}
t.index_word = {index: word for index, word in t.index_word.items() if index <= _vocabSize}
_vocabSize = max([index for index, word in t.index_word.items()]) + 1   # corrects for one-based indexing


    # Sequence the articles to convert them to a list of lists where each inner list
    # contains a serquence of word indices.
    #
print('Converting each article into a sequence of word indices')
_articleSequences = t.texts_to_sequences(_df.text.values)


    # Truncate/pad each article to a uniform length.  We wish to capture at least 90% of
    # the articles in their entirety.  Padding will be performed at the end of the article.
    #
print('Performing article padding/truncation to make all articles a uniform length')
_padLength = np.sort(np.array([len(x) for x in _articleSequences]))[int(np.ceil(
        len(_articleSequences) * _captureFraction))]
_articleSequencesPadded = keras.preprocessing.sequence.pad_sequences(_articleSequences,
                                                                         maxlen=_padLength,
                                                                         padding='post')
print('    Length of training set articles: %s' % _padLength)
print('    Number of training set articles: %s' % len(_articleSequencesPadded))


    # Build the embedding matrix for use in our neural net.  Initialize it to zeros.
    # The matrix has _vocabSize rows and _dimensions columns, and consists of a word
    # vector for each word in the GloVe dataset that exists in the article vocabulary.
    # The row position of each vector corresponds to the word index of that particular
    # word in the article vocabulary (e.g. the vector for the word "the" might occupy
    # row 2 in the matrix.
    #
print('Building the word embedding matrix')
_embeddingMatrix = np.zeros((_vocabSize, _dimensions))

    # Recompute the embedding dictionary with only the vocabulary contained in the
    # articles.  No sense having unused words in the matrix.
    #
print('    Limiting embedding dictionary to only words in our article vocabulary')
_embeddingDict = {word: vec for word, vec in _embeddingDict.items() if word in t.word_index}

    # Populate the matrix.  Words that aren't in the GloVe dataset result in a vector of
    # zeros.
    #
for k, v in t.word_index.items():
    try:
        _embeddingMatrix[v] = _embeddingDict[k]
    except KeyError:
        pass

#%%
        
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

A =  _embeddingMatrix
A_sparse = sparse.csr_matrix(A)

similarities = cosine_similarity(A_sparse)
print('pairwise dense output:\n {}\n'.format(similarities))

#also can output sparse matrices
similarities_sparse = cosine_similarity(A_sparse,dense_output=False)
print('pairwise sparse output:\n {}\n'.format(similarities_sparse))


#%%
import matplotlib.pyplot as plt
 
from sklearn.manifold import TSNE
 
model = TSNE(n_components=3, random_state=0)
np.set_printoptions(suppress=True)
 
Y=model.fit_transform(similarities)


#%%
import pandas as pd
tsne_lda_df = pd.DataFrame(Y, columns=['x', 'y', 'z'])
# tsne_lda_df['Resumes'] = tsne_lda_df['Resumes']
x = tsne_lda_df.x
y = tsne_lda_df.y
z = tsne_lda_df.z

import plotly
import plotly.graph_objs as go
_lean = _df['lean']

def SetColor(x):
    if(x == 'left'): return "blue"
    elif(x == 'left-center'): return "lightblue"
    elif(x == 'least'): return "lightgray"
    elif(x == 'right-center'): return "orangered"
    elif(x == 'right'): return "red"

trace1 =\
    go.Scatter3d(x=x, y=y, z=z,
                 mode='markers',
                 marker = dict(color=list(map(SetColor, _lean)),
                               line=dict(
                                       color='rgb(0,0,0)',
                                       width=0.2),
                                opacity=1))
    
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