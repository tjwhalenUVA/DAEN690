#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 17:50:41 2019

@author: davidwilbur
Test code from https://nlpforhackers.io/keras-intro/
"""
import re
import pandas as pd
from sklearn.model_selection import train_test_split
 
def clean_review(text):
    # Strip HTML tags
    text = re.sub('<[^<]+?>', ' ', text)
 
    # Strip escaped quotes
    text = text.replace('\\"', '')
 
    # Strip quotes
    text = text.replace('"', '')
 
    return text
 
df = pd.read_csv('labeledTrainData.tsv', sep='\t', quoting=3)
df['cleaned_review'] = df['review'].apply(clean_review)
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['sentiment'], test_size=0.2)
 
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
 
vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'), 
                             lowercase=True, min_df=3, max_df=0.9, max_features=5000)
X_train_onehot = vectorizer.fit_transform(X_train)

from keras.models import Sequential
from keras.layers import Dense
 
model = Sequential()
 
model.add(Dense(units=500, activation='relu', input_dim=len(vectorizer.get_feature_names())))
model.add(Dense(units=1, activation='sigmoid'))
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train_onehot[:-100], y_train[:-100], 
          epochs=2, batch_size=128, verbose=1, 
          validation_data=(X_train_onehot[-100:], y_train[-100:]))

scores = model.evaluate(vectorizer.transform(X_test), y_test, verbose=1)
print("Accuracy:", scores[1])  # Accuracy: 0.875

word2idx = {word: idx for idx, word in enumerate(vectorizer.get_feature_names())}
tokenize = vectorizer.build_tokenizer()
preprocess = vectorizer.build_preprocessor()
 
def to_sequence(tokenizer, preprocessor, index, text):
    words = tokenizer(preprocessor(text))
    indexes = [index[word] for word in words if word in index]
    return indexes
 
print(to_sequence(tokenize, preprocess, word2idx, "This is an important test!"))  # [2269, 4453]
X_train_sequences = [to_sequence(tokenize, preprocess, word2idx, x) for x in X_train]
print(X_train_sequences[0])

# Compute the max lenght of a text
MAX_SEQ_LENGHT = len(max(X_train_sequences, key=len))
print("MAX_SEQ_LENGHT=", MAX_SEQ_LENGHT)
 
from keras.preprocessing.sequence import pad_sequences
N_FEATURES = len(vectorizer.get_feature_names())
X_train_sequences = pad_sequences(X_train_sequences, maxlen=MAX_SEQ_LENGHT, value=N_FEATURES)
print(X_train_sequences[0])


from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Embedding
 
model = Sequential()
model.add(Embedding(len(vectorizer.get_feature_names()) + 1,
                    64,  # Embedding size
                    input_length=MAX_SEQ_LENGHT))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train_sequences[:-100], y_train[:-100], 
          epochs=3, batch_size=512, verbose=1,
          validation_data=(X_train_sequences[-100:], y_train[-100:]))

X_test_sequences = [to_sequence(tokenize, preprocess, word2idx, x) for x in X_test]
X_test_sequences = pad_sequences(X_test_sequences, maxlen=MAX_SEQ_LENGHT, value=N_FEATURES)

scores = model.evaluate(X_test_sequences, y_test, verbose=1)
print("Accuracy:", scores[1]) # 0.8766

# LSTM

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
 
model = Sequential()
model.add(Embedding(len(vectorizer.get_feature_names()) + 1,
                    64,  # Embedding size
                    input_length=MAX_SEQ_LENGHT))
model.add(LSTM(64))
model.add(Dense(units=1, activation='sigmoid'))
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
 
model.fit(X_train_sequences[:-100], y_train[:-100], 
          epochs=2, batch_size=128, verbose=1, 
          validation_data=(X_train_sequences[-100:], y_train[-100:]))
 
scores = model.evaluate(X_test_sequences, y_test, verbose=1)
print("Accuracy:", scores[1]) # 0.875
 
# Transfer Learning with spaCy embeddings

import spacy
import numpy as np
nlp = spacy.load('en_core_web_md')
 
EMBEDDINGS_LEN = len(nlp.vocab['apple'].vector)
print("EMBEDDINGS_LEN=", EMBEDDINGS_LEN)  # 300
 
embeddings_index = np.zeros((len(vectorizer.get_feature_names()) + 1, EMBEDDINGS_LEN))
for word, idx in word2idx.items():
    try:
        embedding = nlp.vocab[word].vector
        embeddings_index[idx] = embedding
    except:
        pass

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
 
model = Sequential()
model.add(Embedding(len(vectorizer.get_feature_names()) + 1,
                    EMBEDDINGS_LEN,  # Embedding size
                    weights=[embeddings_index],
                    input_length=MAX_SEQ_LENGHT,
                    trainable=False))
model.add(LSTM(300))
model.add(Dense(units=1, activation='sigmoid'))
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
 
model.fit(X_train_sequences[:-100], y_train[:-100], 
          epochs=1, batch_size=128, verbose=1, 
          validation_data=(X_train_sequences[-100:], y_train[-100:]))
 
scores = model.evaluate(X_test_sequences, y_test, verbose=1)
print("Accuracy:", scores[1])  # 0.8508

# Transfer learning with GloVe embeddings

import numpy as np
 
GLOVE_PATH = './glove.6B/glove.6B.50d.txt'
GLOVE_VECTOR_LENGHT = 50
 
def read_glove_vectors(path, lenght):
    embeddings = {}
    with open(path) as glove_f:
        for line in glove_f:
            chunks = line.split()
            assert len(chunks) == lenght + 1
            embeddings[chunks[0]] = np.array(chunks[1:], dtype='float32')
 
    return embeddings
 
GLOVE_INDEX = read_glove_vectors(GLOVE_PATH, GLOVE_VECTOR_LENGHT)
 
# Init the embeddings layer with GloVe embeddings
embeddings_index = np.zeros((len(vectorizer.get_feature_names()) + 1, GLOVE_VECTOR_LENGHT))
for word, idx in word2idx.items():
    try:
        embedding = GLOVE_INDEX[word]
        embeddings_index[idx] = embedding
    except:
        pass

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
 
model = Sequential()
model.add(Embedding(len(vectorizer.get_feature_names()) + 1,
                    GLOVE_VECTOR_LENGHT,  # Embedding size
                    weights=[embeddings_index],
                    input_length=MAX_SEQ_LENGHT,
                    trainable=False))
model.add(LSTM(128))
model.add(Dense(units=1, activation='sigmoid'))
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
 
model.fit(X_train_sequences[:-100], y_train[:-100], 
          epochs=3, batch_size=128, verbose=1, 
          validation_data=(X_train_sequences[-100:], y_train[-100:]))
 
scores = model.evaluate(X_test_sequences, y_test, verbose=1)
print("Accuracy:", scores[1])  # 0.8296

