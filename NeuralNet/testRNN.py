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
t = keras.preprocessing.text.Tokenizer(num_words=_vocabSize)
t.fit_on_texts(_df.text)

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




print('Padding shorter articles')
_textPadded = keras.preprocessing.sequence.pad_sequences(_textIndices, maxlen=_longestArticle,
                                                         padding='post')
_embeddingsDict = {}
_gloveFile = open('../data/wordEmbeddings/glove.6B.100d.txt')



from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
