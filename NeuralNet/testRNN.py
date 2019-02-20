# -*- coding: utf-8 -*-

# Test for running RNN - recurrent neural net
#
# This code should extract the article ID, leaning, and text from our SQLite database,
# convert the text for each article into a series of embedded word vectors (padding
# as necessary), and feed the results into a recurrent neural net.
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
_cursor.execute('SELECT ln.id, ln.bias, cn.text ' +
                'from lean ln, content cn ' +
                'where ln.id == cn.id')
_rows = _cursor.fetchall()
_df = pd.DataFrame(_rows, columns=('id', 'lean', 'text'))
del _rows

# Convert each article's text into a series of word embeddings.
print('Tokenizing corpus/vocabulary')
_vocabSize = 25000
t = keras.preprocessing.text.Tokenizer(oov_token="OOV", num_words=_vocabSize)
t.fit_on_texts(_df.text)

# The tokenizer in keras fails to handle out of vocabulary words correctly.
# Correct for improper keras OOV behavior and the fact that keras tokenizer
# is 1-based instead of 0-based (0 is reserved).
print('Correcting improper OOV handling in keras')
t.word_index = {word:index for word, index in t.word_index.items() if index <= _vocabSize}
t.word_index[t.oov_token] = _vocabSize + 1

print('Computing vocabulary size')
_vocabularyCount = len(t.word_index) + 1   # add 1 for the null value

print('Assigning indices to each word')
_textIndices = t.texts_to_sequences(_df.text.values)

print('Computing article statistics')
_articleLength = []
for ti in _textIndices:
    _articleLength.append(len(ti))
_articleLength = np.sort(np.array(_articleLength))
_percentages = np.arange(start=0, stop=len(_articleLength)) / float(len(_articleLength))
_fig = plt.figure()
plt.plot(_articleLength, _percentages, 'b')
plt.title('Percentage of Articles with Length < X')
plt.xlabel('Article Length (Tokens)')
plt.ylabel('Percentage of Articles in Corpus')
_fig.savefig('article_lengths.pdf', bbox_inches='tight')

_longestArticle = len(max(_textIndices, key=len))

# Pad and truncate articles to a uniform length (want to capture at least 90% of the
# articles in their entirety -- works out to a length of 1296
#
print('Padding shorter articles')
_padLength = _articleLength[np.argmax(_percentages > 0.90)]
_textPadded = keras.preprocessing.sequence.pad_sequences(_textIndices, maxlen=_padLength,
                                                         padding='post')

# Build our word embedding dictionary from GloVe
#
_embeddingDict = {}
_dimensions = 50
with open('../data/wordEmbeddings/glove.6B.%sd.txt' % _dimensions) as _gloveFile:
    for row in _gloveFile:
        _junk = row.split()
        assert len(_junk) == _dimensions + 1
        _embeddingDict[_junk[0]] = np.array(_junk[1:], dtype='float32')








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
