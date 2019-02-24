# -*- coding: utf-8 -*-

# Test for running RNN - recurrent neural net
#
# This code should extract the article ID, leaning, and text from our SQLite database,
# convert the text for each article into a series of embedded word vectors (padding
# as necessary), and feed the results into a recurrent neural net.  We will be using
# the GloVe dataset as our source for word embedding vectors.
#
# Understanding of word embeddings and example code found at:
# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
#

# Import all the packages!
#
import os
import sqlite3

import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt

# Set some basic path and file variables
#
_rootPath = os.path.dirname(os.path.abspath('__file__'))
_dbFile = r'%s/articles_zenodo.db' % _rootPath

# Set up our connection to the database
#
_db = sqlite3.connect(_dbFile)
_cursor = _db.cursor()

# Pull the relevant bits of data out of the database and
# store them in a pandas dataframe.
#
print('Pulling article IDs, leanings, and text from database')
_cursor.execute("SELECT ln.id, ln.bias, cn.text " +
                "FROM lean ln, content cn " +
                "WHERE cn.`published-at` >= '2009-01-01' AND ln.id == cn.id")
_df = pd.DataFrame(_cursor.fetchall(), columns=('id', 'lean', 'text'))
_db.close()


# Pull GloVe global word vectors from input file and build word vector dictionary
# This creates a {word:vector} dictionary for every word in the GloVe input file.
# The variable _dimensions can be 50, 100, 0r 300, and will read from the corresponding
# GloVe input file.
#
print('Loading word embedding vectors from GloVe 6B data')
_embeddingDict = {}
_dimensions = 50
with open('../data/wordEmbeddings/glove.6B.%sd.txt' % _dimensions) as _gloveFile:
    for _row in _gloveFile:
        _junk = _row.split()
        _embeddingDict[_junk[0]] = np.array(_junk[1:], dtype='float32')


# Tokenize the articles to create a {word:index} dictionary.  The variable _vocabSize
# can be any size we wish.  We set a token for out-of-vocabulary words.
#
print('Tokenizing corpus/vocabulary')
_vocabSize = 25000
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
_captureFraction = 0.90
_padLength = np.sort(np.array([len(x) for x in _articleSequences]))[int(np.ceil(
    len(_articleSequences) * _captureFraction))]
_articleSequencesPadded = keras.preprocessing.sequence.pad_sequences(_articleSequences,
                                                                     maxlen=_padLength,
                                                                     padding='post')


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


# Construct the Tensorflow/keras model
#
model = keras.Sequential()
model.add(keras.layers.Embedding(input_dim=_vocabSize,
                                 output_dim=_dimensions,
                                 embeddings_initializer=keras.initializers.Constant(_embeddingMatrix),
                                 input_length=_padLength,
                                 trainable=False))
model.add(keras.layers.Conv1D(filters=_dimensions, kernel_size=5, activation='relu'))
model.add(keras.layers.MaxPool1D(pool_size=5))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=_dimensions, activation='relu'))
model.add(keras.layers.Dense(5, activation='sigmoid'))

model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

_leanDict = {'left': [1,0,0,0,0],
             'left-center': [0,1,0,0,0],
             'least': [0,0,1,0,0],
             'right-center': [0,0,0,1,0],
             'right': [0,0,0,0,1]}
_leanVals = np.array([_leanDict[k] for k in _df.lean])

model.fit(_articleSequencesPadded, _leanVals, epochs=10)









# _junk = np.where(np.array(_articleLength) >= np.sort(_articleLength)[-1000])
# _longest1000 = (_df.iloc[_junk].id).tolist()
# with open('longest1000.csv', 'w') as filename:
#     for theitem in _longest1000:
#         filename.write('%s\n' % theitem)
#
# _junk = np.sort(np.array(_articleLength))
# _percentages = np.arange(start=0, stop=len(_junk)) / float(len(_junk))
# _p90Length = _junk[np.argmax(_percentages >= 0.90)]
# _junk = np.where(np.array(_articleLength) >= _p90Length)
# _top10percent = (_df.iloc[_junk].id).tolist()
# with open('longest10percent.csv', 'w') as filename:
#     for theitem in _top10percent:
#         filename.write('%s\n' % theitem)
#
#
#
#
# _fig = plt.figure()
# plt.plot(_articleLength, _percentages, 'b')
# plt.title('Percentage of Articles with Length < X')
# plt.xlabel('Article Length (Tokens)')
# plt.ylabel('Percentage of Articles in Corpus')
# _fig.savefig('article_lengths.pdf', bbox_inches='tight')
#
# _longestArticle = len(max(_textIndices, key=len))










# Creating a reverse dictionary
reverse_word_map = dict(map(reversed, t.word_index.items()))

# Function takes a tokenized sentence and returns the words
def sequence_to_text(list_of_indices):
    # Looking up words in dictionary
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return(words)

sequence_to_text(max(_textIndices, key=len))




from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
