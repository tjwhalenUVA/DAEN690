#! /Users/dpbrinegar/anaconda3/envs/gitHub/bin/python
# -*- coding: utf-8 -*-
#
# Author:  Paul M. Brinegar, II
#
# Date Created:  20190301
#
# CNNgrid.py - Code for running a CNN (Convolutional Neural Net) with grid search
#
# This code should extract the article ID, leaning, and text from our SQLite database,
# convert the text for each article into a series of embedded word vectors (padding
# as necessary), and feed the results into a convolutional neural net.  We will be using
# the GloVe dataset as our source for word embedding vectors.
#
# We use a grid search with various parameters.  Apparently it is possible to wrap
# our neural net model inside a sklearn grid search wrapper, but attempts to accomplish
# this resulted in multiple, multiple, multiple tensorflow errors being thrown, to the
# point where it was simply easier to do a grid search the old-fashioned way.
#
# Rather than using sklearn's wrapper, we instead built an input file with all of the
# hyperparameter combinations we wish to test.  We then loop through the file row by row
# and store the results from each training/validation run.  We also track the row whose
# results are the "best" by whatever metric we select, and that's our winner.
#
# Understanding of word embeddings and example code found at:
# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
#

import argparse                         # package for handling command line arguments
import tensorflow as tf
from tensorflow import keras


# Routine for building our model.  This routine constructs our neural net model using the arguments
# passed to it.
#
# By moving the model construction into a separate function, we also can allow the neural net model
# to be called easily inside of a loop
#
def constructModel(_vocabSize, _embeddingMatrix, _padLength, _dimensions=50,
                   _cnnFilters=50, _cnnKernel=5, _convActivation='relu',
                   _cnnPool=5,
                   _cnnFlatten=True,
                   _cnnDense=50, _denseActivation='relu',
                   _cnnDropout=0.0,
                   _outputActivation='softmax',
                   _lossFunction='categorical_crossentropy',
                   _summarize=True,
                   _optimizer='adam'):

    # Initialize the model
    theModel = keras.Sequential()

    # Add our word embedding layer.  This layer converts the word indices sent to the model into
    # vectors of length N, where N is the length of the GloVe word vectors.  This conversion is
    # done for each word in an article, resulting in a matrix of size [_padLength, N] (... maybe
    # transposed from that?)
    theModel.add(keras.layers.Embedding(input_dim=_vocabSize,
                                        output_dim=_dimensions,
                                        embeddings_initializer=keras.initializers.Constant(_embeddingMatrix),
                                        input_length=_padLength,
                                        trainable=False))

    # Add a 1-dimensional convolution layer.  This layer moves a window of size _cnnKernel across
    # the input and creates an output of length _cnnFilters for each window.
    theModel.add(keras.layers.Conv1D(filters=_cnnFilters, kernel_size=_cnnKernel, activation=_convActivation))

    # Add a max pooling layer.  This layer looks at the vectors contained in a window of size _cnnPool
    # and outputs the vector with the greatest L2 norm.
    theModel.add(keras.layers.MaxPool1D(pool_size=_cnnPool))

    # Add a flatten layer.  This layer removes reduces the output to a one-dimensional vector
    if _cnnFlatten:
        theModel.add(keras.layers.Flatten())

    # Add a fully connected dense layer.  This layer adds a lot of nodes to the model to allow
    # for different features in the article to activate different groups of nodes.
    theModel.add(keras.layers.Dense(units=_cnnDense, activation=_denseActivation))

    # Add a dropout layer.  This layer reduces overfitting by randomly "turning off" nodes
    # during each training epoch.  Doing this prevents a small set of nodes doing all the
    # work while a bunch of other nodes sit around playing poker.
    if _cnnDropout > 0.0:
        theModel.add(keras.layers.Dropout(_cnnDropout))

    # Add our output layer.  We have 5 classes of output "left", "left-center", "least",
    # "right-center", and "right".  This layer converts the inputs from the dense/dropout
    # layer into outputs for these 5 classes, essentially predicting the article leaning.
    theModel.add(keras.layers.Dense(5, activation=_outputActivation))

    # Display a summary of our model
    if _summarize:
        theModel.summary()

    # Compile our model.  We use categorical crossentropy for the training loss function
    # since our predictions are multiclass.  We use categorical accuracy as our
    # performance metric, though we can report others as well.  The optimizer can
    # be any of the allowed optimizers (default is 'adam', a form of stochastic
    # gradient descent).
    theModel.compile(optimizer=_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return theModel


def main(_gridFile, _numFolds, _epochs, _verbose, _GPUid):
    #
    # Import all the packages!
    print('Importing packages')
    import os
    if _GPUid != None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '%s' % _GPUid

    import gc

    import sqlite3

    from pandas import DataFrame
    from pandas import read_csv

    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    from sys import platform as sys_pf
    if sys_pf == 'darwin':
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
    else:
        import matplotlib.pyplot as plt

    # Initialize some variables that affect when to re-load and re-process the input files
    oldDbFile = None
    oldGloveFile = None
    oldVocabSize = None
    oldCaptureFraction = None


    # Load our grid search CSV file into a pandas dataframe
    _dfGrid = read_csv(_gridFile, index_col=False,
                       dtype={'id': int,
                              'dbFile': str,
                              'gloveFile': str,
                              'vocabSize': int,
                              'captureFraction': float,
                              'convolutionFilters': int,
                              'convolutionKernel': int,
                              'convolutionActivation': str,
                              'poolSize': int,
                              'flattenLayer': bool,
                              'denseUnits': int,
                              'denseActivation': str,
                              'dropoutFraction': float,
                              'outputActivation': str,
                              'lossFunction': str})

    # Start the grid search
    _idBase = min(_dfGrid.id)
    _maxRow = None
    _maxAcc = 0.0
    _minRow = None
    _minLoss = np.finfo(np.float32).max
    _acc = []
    _val_acc = []
    _loss = []
    _val_loss = []
    for _row in _dfGrid.itertuples():
        print(_row)

        # Set a "reprocess the input data" flag
        _flag = False

        # If we are reading in a new database, here's where we do it.
        if _row.dbFile != oldDbFile:
            _flag = True

            oldDbFile = _row.dbFile
            print('Loading article IDs, leanings, and text from database: %s' % _row.dbFile)

            # Create connection to the database file
            _db = sqlite3.connect(_row.dbFile)
            _cur = _db.cursor()

            # Load the data from the database
            _command = "SELECT cn.id, ln.bias_final, cn.text " + \
                       "FROM train_content cn, train_lean ln " + \
                       "WHERE (cn.id < 9999999999) AND " + \
                       "(cn.`published-at` >= '2009-01-01') AND " + \
                             "(cn.id == ln.id) AND " + \
                             "(ln.url_keep == 1) AND " + \
                             "(cn.id NOT IN (SELECT a.id " + \
                                            "FROM train_content a, train_content b " + \
                                            "WHERE (a.id < b.id) AND " + \
                                                  "(a.text == b.text)));"

            _cur.execute(_command)
            _df = DataFrame(_cur.fetchall(), columns=('id', 'lean', 'text'))
            _db.close()
            print('%s records read from database' % len(_df))

        # If we are reading in a new GloVe global word vector file, here's where we do it.  This creates a
        # {word: vector} dictionary for every word in the GloVe input file.
        if _row.gloveFile != oldGloveFile:
            _flag = True

            oldGloveFile = _row.gloveFile
            print('Loading global word vectors from file: %s' % _row.gloveFile)

            # Initialize our word embedding dictionary
            _embeddingDict = {}

            # Compute the number of dimensions in our word vectors from the file name.
            _dimensions = int(_row.gloveFile[-8:-5].strip('.'))

            # Load the word vector data
            with open(_row.gloveFile) as _gf:
                for _gfRow in _gf:
                    _junk = _gfRow.split()
                    _embeddingDict[_junk[0]] = np.array(_junk[1:], dtype='float32')

        if _row.vocabSize != oldVocabSize:
            _flag = True
            oldVocabSize = _row.vocabSize

        if _row.captureFraction != oldCaptureFraction:
            _flag = True
            oldCaptureFraction = _row.captureFraction

        # Repprocess the input data (if necessary)
        if _flag:
            _vocabSize = _row.vocabSize
            # Tokenize the articles to create a {word:index} dictionary.  The variable _vocabSize
            # can be any size we wish.  We set a token for out-of-vocabulary words.
            print('Tokenizing corpus of articles and building vocabulary')
            t = keras.preprocessing.text.Tokenizer(oov_token="<OOV>", num_words=_vocabSize)
            t.fit_on_texts(_df.text)

            # The tokenizer in keras fails to create its {word:index} dictionary properly.  Words
            # with indices larger than the vocabulary size are retained, so we must rebuild the
            # dictionary to truncate to the specified vocabulary size.
            print('Correcting improper OOV handling in keras')
            t.word_index = {word: index for word, index in t.word_index.items() if index <= _vocabSize}
            t.index_word = {index: word for index, word in t.index_word.items() if index <= _vocabSize}
            _vocabSize = max([index for index, word in t.index_word.items()]) + 1   # corrects for one-based indexing

            # Sequence the articles to convert them to a list of lists where each inner list
            # contains a serquence of word indices.  Store in temporary _junk variable.
            print('Converting each article into a sequence of word indices')
            _junk = t.texts_to_sequences(_df.text.values)
            _junklen = [len(x) for x in _junk]
            _df = _df.assign(length=_junklen)

            # Determine how much correlation there is between each word in the vocabulary
            # and article leaning.  The result should be an N by 5 matrix, where N is
            # the number of words/tokens in the vocabulary -- 5 columns, one for each
            # category of leaning.
            _leanValuesDict = {'left': 0,
                               'left-center': 1,
                               'least': 2,
                               'right-center': 3,
                               'right': 4}
            _leanVectorDict = {0: [1,0,0,0,0],
                               1: [0,1,0,0,0],
                               2: [0,0,1,0,0],
                               3: [0,0,0,1,0],
                               4: [0,0,0,0,1]}
            _leanArray = np.array([_leanVectorDict[_leanValuesDict[x]] for x in _df.lean])
            _corArray = np.zeros(shape=(max(t.index_word.keys())+1, len(_leanValuesDict)))
            _sy = np.sum(_leanArray, axis=0)
            ss_yy = _sy - np.square(_sy) / float(len(_leanArray))
            for _i in t.index_word.keys():
                print('Computing correlation for word %s: %s' % (_i, t.index_word[_i]))
                _presence = np.array([[int(_i in x) for x in _junk], ] * len(_leanValuesDict)).transpose()
                _sx = np.sum(_presence, axis=0)
                ss_xx = _sx - np.square(_sx) / float(len(_leanArray))

                ss_xy = np.sum(np.multiply(_leanArray, _presence), axis=0) - np.multiply(_sx, _sy) / float(len(_leanArray))

                _corArray[_i,] = np.square(ss_xy) / ss_xx / ss_yy

            # It is possible that articles contain publisher information or bylines that are
            # highly correlated with the leaning (since leaning is assigned by-publisher).
            # To combat this, we should remove byline information.  This information usually
            # occurs at the beginning or end of an article, so we are removing the first and
            # last N words from each article.  Of course, this means that we are throwing out
            # any article of length 2N or less.
            _N = 50
            _articleSequences = [x[_N:-_N] for x in _junk if len(x) > (2*_N)]

            # Truncate/pad each article to a uniform length.  We wish to capture at least 90% of
            # the articles in their entirety.  Padding will be performed at the end of the article.
            print('Performing article padding/truncation to make all articles a uniform length')
            _padLength = np.sort(np.array([len(x) for x in _articleSequences]))[int(np.ceil(
                len(_articleSequences) * _row.captureFraction))]
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
            print('Building the word embedding matrix')
            _embeddingMatrix = np.zeros((_vocabSize, _dimensions))

            # Recompute the embedding dictionary with only the vocabulary contained in the
            # articles.  No sense having unused words in the matrix.
            print('    Limiting embedding dictionary to only words in our article vocabulary')
            _embeddingDict = {word: vec for word, vec in _embeddingDict.items() if word in t.word_index}

            # Populate the matrix.  Words that aren't in the GloVe dataset result in a vector of
            # zeros.
            for k, v in t.word_index.items():
                try:
                    _embeddingMatrix[v] = _embeddingDict[k]
                except KeyError:
                    pass

            # Encode the leanings ultimately as one-hot categorical vectors
            #
            # Because we may be using sklearn's StratifiedKFold routine below, we
            # have to do a two-step process for creating our categorical vectors.
            # First, we convert our "left"/'right" leanings into integers.  These
            # get fed through the StratifiedKFold routine.  On the back side of that
            # routine, we convert those integers into one-hot categorical vectors results
            # ['left', 'left-center', 'least', 'right-center', 'right]
            # The neural net will output article probabilities for each of these categories
            print('Encoding leanings')
            _leanValuesDict = {'left': 0,
                               'left-center': 1,
                               'least': 2,
                               'right-center': 3,
                               'right': 4}
            _leanVectorDict = {0: [1,0,0,0,0],
                               1: [0,1,0,0,0],
                               2: [0,0,1,0,0],
                               3: [0,0,0,1,0],
                               4: [0,0,0,0,1]}
            _leanVals = np.array([_leanValuesDict[k] for k in _df.lean[_df.length > (2*_N)]])

        # Perform a K-fold cross validation of the training set.
        # We had to manually create this due to the way keras' model.fit
        # handles splitting.  Keras doesn't randomly or sequentially split
        # a dataset into train/validation sets; rather, it just always takes
        # the last N percent of the set as the validation set... can't
        # do cross validation like that unless we shuffled the data
        # each time through.
        #
        # By using sklearn's StratifiedKFold routine, we split the dataset into
        # K folds while attempting to preserve the relative percentages of each
        # class in both the training and test/validation set.  Thus we don't end
        # up with a horribly imbalanced set as part of our cross validation.
        #
        if _numFolds == 0:
            _foldFlag = False
            _numFolds = 10
            print('Performing 90/10 split training and validation, %s epochs' % _epochs)
        else:
            _foldFlag = True
            print('Performing %s fold cross validation, %s epochs per fold' % (_numFolds, _epochs))
        _kfold = StratifiedKFold(n_splits=_numFolds, shuffle=True)

        # Initialize some variables to hold our crossvalidation results
        _facc = []
        _fval_acc = []
        _floss = []
        _fval_loss = []

        # Perform the K-fold cross validation
        j = 1
        for _train, _val in _kfold.split(_articleSequencesPadded, _leanVals):
            if _foldFlag:
                print('\nFold %s' % j)

                # Construct the Tensorflow/keras model for a convolutional neural net
                model = constructModel(_vocabSize=_vocabSize, _dimensions=_dimensions, _embeddingMatrix=_embeddingMatrix,
                                       _padLength=_padLength,
                                       _cnnFilters=_row.convolutionFilters, _cnnKernel=_row.convolutionKernel,
                                       _convActivation=_row.convolutionActivation,
                                       _cnnPool=_row.poolSize,
                                       _cnnFlatten=_row.flattenLayer,
                                       _cnnDense=_row.denseUnits,
                                       _denseActivation=_row.denseActivation,
                                       _cnnDropout=_row.dropoutFraction,
                                       _outputActivation=_row.outputActivation,
                                       _lossFunction=_row.lossFunction)

                # Fit the model to the data
                _history = model.fit(_articleSequencesPadded[_train],
                                     np.array([_leanVectorDict[k] for k in _leanVals[_train]]),
                                     epochs=_epochs, batch_size=128,
                                     validation_data=(_articleSequencesPadded[_val],
                                                      np.array([_leanVectorDict[k] for k in _leanVals[_val]])),
                                     verbose=_verbose)
                _historyDict = _history.history
                _facc.append(_historyDict['categorical_accuracy'])
                _fval_acc.append(_historyDict['val_categorical_accuracy'])
                _floss.append(_historyDict['loss'])
                _fval_loss.append(_historyDict['val_loss'])

            else:
                if j == 1:
                    print('Training and Validating Model')
                    # Construct the Tensorflow/keras model for a convolutional neural net
                    model = constructModel(_vocabSize=_vocabSize, _dimensions=_dimensions, _embeddingMatrix=_embeddingMatrix,
                                           _padLength=_padLength,
                                           _cnnFilters=_row.convolutionFilters, _cnnKernel=_row.convolutionKernel,
                                           _convActivation=_row.convolutionActivation,
                                           _cnnPool=_row.poolSize,
                                           _cnnFlatten=_row.flattenLayer,
                                           _cnnDense=_row.denseUnits,
                                           _denseActivation=_row.denseActivation,
                                           _cnnDropout=_row.dropoutFraction,
                                           _outputActivation=_row.outputActivation,
                                           _lossFunction=_row.lossFunction)

                    # Fit the model to the data
                    _history = model.fit(_articleSequencesPadded[_train],
                                         np.array([_leanVectorDict[k] for k in _leanVals[_train]]),
                                         epochs=_epochs, batch_size=128,
                                         validation_data=(_articleSequencesPadded[_val],
                                                          np.array([_leanVectorDict[k] for k in _leanVals[_val]])),
                                         verbose=_verbose)
                    _historyDict = _history.history
                    _facc.append(_historyDict['categorical_accuracy'])
                    _fval_acc.append(_historyDict['val_categorical_accuracy'])
                    _floss.append(_historyDict['loss'])
                    _fval_loss.append(_historyDict['val_loss'])

            j += 1

        # Compute the mean results from all the crossvalidation folds
        _acc.append(np.mean(np.matrix(_facc), axis=0).tolist()[0])
        _val_acc.append(np.mean(np.matrix(_fval_acc), axis=0).tolist()[0])
        _loss.append(np.mean(np.matrix(_floss), axis=0).tolist()[0])
        _val_loss.append(np.mean(np.matrix(_fval_loss), axis=0).tolist()[0])

        # Compare results to the current best results using _val_acc as the measure, and keep the best one
        if np.max(np.mean(np.matrix(_fval_acc), axis=0).tolist()[0]) > _maxAcc:
            _maxAcc = np.max(np.mean(np.matrix(_fval_acc), axis=0).tolist()[0])
            _maxEpoch = np.argmax(np.mean(np.matrix(_fval_acc), axis=0).tolist()[0]) + 1
            _maxRow = _row
        if np.min(np.mean(np.matrix(_fval_loss), axis=0).tolist()[0]) < _minLoss:
            _minLoss = np.min(np.mean(np.matrix(_fval_loss), axis=0).tolist()[0])
            _minEpoch = np.argmin(np.mean(np.matrix(_fval_loss), axis=0).tolist()[0]) + 1
            _minRow = _row

        # Remove the current model from the tensorflow backend to prepare for the next model.
        # Also remove the model from keras, and perform some garbage cleanup.
        keras.backend.clear_session()
        del model
        junk = gc.collect()
        print('Garbage Collection: %s objects collected' % junk)

    print('\n\n')
    print('Parameter Combinations Processed:')
    for _row in _dfGrid.itertuples():
        print(_row)
    print('\n\nResults:')
    print('Greatest Accuracy:  %s' % _maxAcc)
    print('Number of Epochs:   %s' % _maxEpoch)
    print('Parameters:')
    print(_maxRow)
    print('\n')
    print('Lowest Loss:        %s' % _minLoss)
    print('Number of Epochs:   %s' % _minEpoch)
    print('Parameters:')
    print(_minRow)
    print('\n')
    print('Training Accuracies:')
    j = _idBase
    for junk in _acc:
        print(j, junk)
        j += 1
    print('\n')
    print('Validation Accuracies:')
    j = _idBase
    for junk in _val_acc:
        print(j, junk)
        j += 1
    print('\n')
    print('Training Loss:')
    j = _idBase
    for junk in _loss:
        print(j, junk)
        j += 1
    print('\n')
    print('Validation Loss:')
    j = _idBase
    for junk in _val_loss:
        print(j, junk)
        j += 1
    print('\n')
    print('Parameter Combinations Processed:')
    for _row in _dfGrid.itertuples():
        print(_row)


if __name__ == '__main__':
    _parser = argparse.ArgumentParser(description='Grid search for convolution neural net model.')
    _parser.add_argument('gridfile', type=str, default='gridSearch.csv',
                         help='Path/file to the grid search parameter CSV file to use.')
    _parser.add_argument('-f', '--folds', type=int, default=5, help='Number of cross-validation folds to perform')
    _parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of epochs to run in the neural net')
    _parser.add_argument('-v', '--verbose', type=int, default=1, help='Verbosity of model fitting ' +
                         '0: no output, 1: progress bar, 2: epoch only')
    _parser.add_argument('-Z', '--gpuid', type=int, default=None,
                         help='ID of the GPU to use (for none specified, skip this argument)')
    _args = _parser.parse_args()

    print(_args)

    main(_args.gridfile, _args.folds, _args.epochs, _args.verbose, _args.gpuid)

