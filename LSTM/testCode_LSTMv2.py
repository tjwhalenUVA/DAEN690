#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 20:32:31 2019

@author: davidwilbur
"""

# Deep Neural Network
# We’re going to use the same dataset we’ve used in the Introduction to 
# DeepLearning Tutorial. Let’s just quickly cover the data cleaning part:

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
 
# Let’s now build a CountVectorizer how we usually do:

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
 
vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'), 
                             lowercase=True, min_df=3, max_df=0.9, max_features=5000)
X_train_onehot = vectorizer.fit_transform(X_train)
 
# Here’s how to create a simple, 2 layer network. The first layer 
# (which actually comes after an input layer) is called the hidden layer, 
# and the second one is called the output layer. Notice how we had to specify 
# the input dimension (input_dim) and how we only have 1 unit in the output 
# layer because we’re dealing with a binary classification problem. Because 
# we’re dealing with a binary classification problem we chose the output 
# layer’s activation function to be the sigmoid. For the same reason, we 
# chose the binary_crossentropy as the loss function:

from keras.models import Sequential
from keras.layers import Dense
 
model = Sequential()
 
model.add(Dense(units=500, activation='relu', input_dim=len(vectorizer.get_feature_names())))
model.add(Dense(units=1, activation='sigmoid'))
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# dense_22 (Dense)             (None, 500)               2500500   
# _________________________________________________________________
# dense_23 (Dense)             (None, 1)                 501       
# =================================================================
# Total params: 2,501,001
# Trainable params: 2,501,001
# Non-trainable params: 0
# _________________________________________________________________
 
# Here’s how the training is done:

model.fit(X_train_onehot[:-100], y_train[:-100], 
          epochs=2, batch_size=128, verbose=1, 
          validation_data=(X_train_onehot[-100:], y_train[-100:]))
 
# Notice how we set aside some samples for doing validation while training. 
# We still need to do the evaluation on test data:

scores = model.evaluate(vectorizer.transform(X_test), y_test, verbose=1)
print("Accuracy:", scores[1])  # Accuracy: 0.875
 
# We got an 87.5% accuracy, which is pretty good. Let’s check out the 
# other models.

# Convolutional Network

# For working with conv nets and recurrent nets we need to transform the 
# texts into sequences of word ids. We will train an embeddings layer, and 
# using the word ids we can fetch the corresponding word vector.

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
 
# We have a problem though. The sequences are of different lengths. 
# We solve this problem by padding the sequence to the left with 5000.

# Compute the max lenght of a text
MAX_SEQ_LENGHT = len(max(X_train_sequences, key=len))
print("MAX_SEQ_LENGHT=", MAX_SEQ_LENGHT)
 
from keras.preprocessing.sequence import pad_sequences
N_FEATURES = len(vectorizer.get_feature_names())
X_train_sequences = pad_sequences(X_train_sequences, maxlen=MAX_SEQ_LENGHT, value=N_FEATURES)
print(X_train_sequences[0])
 
# Let’s now define a simple CNN for text classification:

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
 
# Training the model looks the same as before:


model.fit(X_train_sequences[:-100], y_train[:-100], 
          epochs=3, batch_size=512, verbose=1,
          validation_data=(X_train_sequences[-100:], y_train[-100:]))
 
# Let’s now transform the test data to sequences and pad them:

X_test_sequences = [to_sequence(tokenize, preprocess, word2idx, x) for x in X_test]
X_test_sequences = pad_sequences(X_test_sequences, maxlen=MAX_SEQ_LENGHT, value=N_FEATURES)
 
# Here’s how to evaluate the model:


scores = model.evaluate(X_test_sequences, y_test, verbose=1)
print("Accuracy:", scores[1]) # 0.8766
 
# LSTM Network

# Let’s build what’s probably the most popular type of model in NLP at the 
# moment: Long Short Term Memory network. This architecture is specially 
# designed to work on sequence data. It fits perfectly for many NLP tasks like 
# tagging and text classification. It treats the text as a sequence rather 
# than a bag of words or as ngrams.

# Here’s a possible model definition:

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
 
# Training is similar:

model.fit(X_train_sequences[:-100], y_train[:-100], 
          epochs=2, batch_size=128, verbose=1, 
          validation_data=(X_train_sequences[-100:], y_train[-100:]))
 
# Here’s the evaluation phase and results:

scores = model.evaluate(X_test_sequences, y_test, verbose=1)
print("Accuracy:", scores[1]) # 0.875
 
# In the next 2 sections, we’re going to explore transfer learning, a method 
# for reducing the number of parameters we need to train for a network.

# Transfer Learning with spaCy embeddings

# Notice how in the previous two examples, we used an Embedding layer. 
# In the previous cases, that layer had to be trained, adding to the number 
# of parameters that need to be trained. What if we used some precomputed 
# embeddings? We can certainly do this. Say we trained a Word2Vec model on our 
# corpus and then we use those embeddings for the various other models we need 
# to train. In this tutorial, we’ll first use the spaCy embeddings. Here’s how 
# to do that:

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
 
# Next, we’ll define the same network, just like before, but using a 
# pretrained Embedding layer:

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
 
# Here’s how that performs:

model.fit(X_train_sequences[:-100], y_train[:-100], 
          epochs=1, batch_size=128, verbose=1, 
          validation_data=(X_train_sequences[-100:], y_train[-100:]))
 
scores = model.evaluate(X_test_sequences, y_test, verbose=1)
print("Accuracy:", scores[1])  # 0.8508
 
# Transfer learning with GloVe embeddings

# In this section we’re going to do the same, but with smaller, 
# GloVe embeddings.

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
 
# Let’s try this model out:

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
