#! /usr/bin/python
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

    # Initialize the model by creating an input layer.  This layer matches the shape of the
    # article's list of word IDs.
    inputLayer = keras.layers.Input(shape=(_padLength,),
                                    name='inputLayer')

    # Add our word embedding layer.  This layer converts the word indices sent to the model into
    # vectors of length N, where N is the length of the GloVe word vectors.  This conversion is
    # done for each word in an article, resulting in a matrix of size [_padLength, N] (... maybe
    # transposed from that?)
    wordEmbedding = keras.layers.Embedding(input_dim=_vocabSize,
                                           output_dim=_dimensions,
                                           embeddings_initializer=keras.initializers.Constant(_embeddingMatrix),
                                           trainable=False,
                                           name='embeddingLayer')(inputLayer)

    # Add a 1-dimensional convolution layer.  This layer moves a window of size _cnnKernel across
    # the input and creates an output of length _cnnFilters for each window.
    convLayer = keras.layers.Conv1D(filters=_cnnFilters,
                                    kernel_size=_cnnKernel,
                                    activation=_convActivation,
                                    name='convolutionLayer')(wordEmbedding)

    # Add a dropout layer.  This layer reduces overfitting by randomly "turning off" nodes
    # during each training epoch.  Doing this prevents a small set of nodes doing all the
    # work while a bunch of other nodes sit around playing poker.
    if _cnnDropout > 0.0:
        convLayer = keras.layers.Dropout(_cnnDropout,
                                         name='convolutionDropoutLayer')(convLayer)


    # Add a max pooling layer.  This layer looks at the vectors contained in a window of size _cnnPool
    # and outputs the vector with the greatest L2 norm.
    maxPoolLayer = keras.layers.MaxPool1D(pool_size=_cnnPool,
                                          name = 'maxPoolingLayer')(convLayer)

    # Add a flatten layer.  This layer removes reduces the output to a one-dimensional vector
    if _cnnFlatten:
        maxPoolLayer = keras.layers.Flatten(name='flatteLayer')(maxPoolLayer)


    # Add a fully connected dense layer.  This layer adds a lot of nodes to the model to allow
    # for different features in the article to activate different groups of nodes.
    denseLayer = keras.layers.Dense(units=_cnnDense,
                                    activation=_denseActivation,
                                    name='denseLayer')(maxPoolLayer)

    # Add a dropout layer.  This layer reduces overfitting by randomly "turning off" nodes
    # during each training epoch.  Doing this prevents a small set of nodes doing all the
    # work while a bunch of other nodes sit around playing poker.
    if _cnnDropout > 0.0:
        denseLayer = keras.layers.Dropout(_cnnDropout,
                                          name='denseDropoutLayer')(denseLayer)

    # Add our output layer.  We have 5 classes of output "left", "left-center", "least",
    # "right-center", and "right".  This layer converts the inputs from the dense/dropout
    # layer into outputs for these 5 classes, essentially predicting the article leaning.
    outputLayer = keras.layers.Dense(5, activation=_outputActivation,
                                     name='outputLayer')(denseLayer)

    # Create the actual model from the layers described above.
    theModel = keras.Model(inputs = [inputLayer], outputs=[outputLayer])

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


def main(_gridFile, _numFolds, _epochs, _verbose, _runtest, _GPUid):
    #
    # Import all the packages!
    print('Importing packages')
    import os
    if _GPUid != None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '%s' % _GPUid

    import time

    import gc

    import sqlite3

    from pandas import DataFrame
    from pandas import read_csv

    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import confusion_matrix

    from sys import platform as sys_pf
    if sys_pf == 'darwin':
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
    else:
        import matplotlib.pyplot as plt


    # Let's begin!
    _t0 = time.time()
    _ta = time.time()

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
    _tz = time.time()
    print('\nGrid search file load time: %0.3f\tTotal time elapsed: %0.3f seconds\n' % (_tz - _ta, _tz - _t0))

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
            ta = time.time()

            # Create connection to the database file
            _db = sqlite3.connect(_row.dbFile)
            _cur = _db.cursor()

            # Load the training set data from the database
            _command = "SELECT cn.id, ln.bias_final, cn.text, ln.url " + \
                       "FROM train_content cn, train_lean ln " + \
                       "WHERE (cn.id < 59999) AND " + \
                       "(cn.`published-at` >= '2009-01-01') AND " + \
                             "(cn.id == ln.id) AND " + \
                             "(ln.url_keep == 1) AND " + \
                             "(cn.id NOT IN (SELECT a.id " + \
                                            "FROM train_content a, train_content b " + \
                                            "WHERE (a.id < b.id) AND " + \
                                                  "(a.text == b.text)));"

            _cur.execute(_command)
            _df = DataFrame(_cur.fetchall(), columns=('id', 'lean', 'text', 'url'))

            # load the test set data from the database
            _command = "SELECT cn.id, ln.bias_final, cn.text, ln.url " + \
                       "FROM test_content cn, test_lean ln " + \
                       "WHERE (cn.id < 59999) AND " + \
                       "(cn.`published-at` >= '2009-01-01') AND " + \
                             "(cn.id == ln.id) AND " + \
                             "(ln.url_keep == 1) AND " + \
                             "(cn.id NOT IN (SELECT a.id " + \
                                            "FROM test_content a, test_content b " + \
                                            "WHERE (a.id < b.id) AND " + \
                                                  "(a.text == b.text)));"

            _cur.execute(_command)
            _dfx = DataFrame(_cur.fetchall(), columns=('id', 'lean', 'text', 'url'))
            _db.close()
            print('%s training records read from database' % len(_df))
            print('%s test records read from database' % len(_dfx))

            _tz = time.time()
            print('\nDatabase query time: %0.3f\tTotal time elapsed: %0.3f seconds' % (_tz - _ta, _tz - _t0))
            print('Database query time only valid for first time query is made\n')

        # If we are reading in a new GloVe global word vector file, here's where we do it.  This creates a
        # {word: vector} dictionary for every word in the GloVe input file.
        if _row.gloveFile != oldGloveFile:
            _flag = True

            oldGloveFile = _row.gloveFile
            print('Loading global word vectors from file: %s' % _row.gloveFile)
            ta = time.time()

            # Initialize our word embedding dictionary
            _embeddingDict = {}

            # Compute the number of dimensions in our word vectors from the file name.
            _dimensions = int(_row.gloveFile[-8:-5].strip('.'))

            # Load the word vector data
            with open(_row.gloveFile) as _gf:
                for _gfRow in _gf:
                    _junk = _gfRow.split()
                    _embeddingDict[_junk[0]] = np.array(_junk[1:], dtype='float32')

            _tz = time.time()
            print('\nGloVe file load time: %0.3f\tTotal time elapsed: %0.3f seconds\n' % (_tz - _ta, _tz - _t0))

        if _row.vocabSize != oldVocabSize:
            _flag = True
            oldVocabSize = _row.vocabSize

        if _row.captureFraction != oldCaptureFraction:
            _flag = True
            oldCaptureFraction = _row.captureFraction

        # Repprocess the input data (if necessary)
        if _flag:
            # If we wish to exclude certain publishers from the training set,
            # remove them here.  Publishers are exlucded by their primary URL
            # address (e.g. AP News would be 'apnews', Albuquerque Journal would
            # be 'abqjournal'.
            print('Excluding specific publishers due to strongly biasing the model')
            _excludePublishers = ['NULL']
            #            _excludePublishers = ['apnews', 'foxbusiness']
            _excludeString = '|'.join(_excludePublishers)
            _df = _df[~_df['url'].str.contains(_excludeString)]

            _vocabSize = _row.vocabSize

            _ta = time.time()

            # Tokenize the articles to create a {word:index} dictionary.  The variable _vocabSize
            # can be any size we wish.  We set a token for out-of-vocabulary words.  We start with
            # a vocabulary size much larger than we intend to actually use, to allow for removal
            # of various words/characters that we don't want to include.  NOTE:  keras does not
            # seem to do much with the num_words argument, as it does not truncate its dictionaries
            # during tokenizing.
            print('Tokenizing corpus of articles and building vocabulary')
            t = keras.preprocessing.text.Tokenizer(oov_token="<OOV>", num_words=_vocabSize)
            t.fit_on_texts(_df.text)

            _tz = time.time()
            print('\nTokenizing time: %0.3f\tTotal time elapsed: %0.3f seconds\n' % (_tz - _ta, _tz - _t0))

            # The tokenizer in keras fails to create its {word:index} dictionary properly.  Words
            # with indices larger than the vocabulary size are retained, so we must rebuild the
            # dictionary to truncate to the specified vocabulary size.
            print('Correcting improper OOV handling in keras')
            t.word_index = {word: index for word, index in t.word_index.items() if index <= _vocabSize}
            t.index_word = {index: word for index, word in t.index_word.items() if index <= _vocabSize}
            t.word_index['___'] = 0
            t.index_word[0] = '___'
            _vocabSize = max([index for index, word in t.index_word.items()]) + 1   # corrects for one-based indexing

            _ta = time.time()

            # Sequence the articles to convert them to a list of lists where each inner list
            # contains a serquence of word indices.  Store in temporary _junk variable.
            print('Converting each article into a sequence of word indices')
            _articleSequences = t.texts_to_sequences(_df.text.values)
            _sequenceLengths = [len(x) for x in _articleSequences]
            _df = _df.assign(length=_sequenceLengths)

            _testarticleSequences = t.texts_to_sequences(_dfx.text.values)

            _tz = time.time()
            print('\nSequence conversion time: %0.3f\tTotal time elapsed: %0.3f seconds\n' % (_tz - _ta, _tz - _t0))


            _ta = time.time()

            # Truncate/pad each article to a uniform length.  We wish to capture at least 90% of
            # the articles in their entirety.  Padding will be performed at the end of the article.
            print('Performing article padding/truncation to make all articles a uniform length')
            _padLength = np.sort(np.array([len(x) for x in _articleSequences]))[int(np.ceil(
                len(_articleSequences) * _row.captureFraction))]
            _articleSequencesPadded = keras.preprocessing.sequence.pad_sequences(_articleSequences,
                                                                                 maxlen=_padLength,
                                                                                 padding='post')
            _testarticleSequencesPadded = keras.preprocessing.sequence.pad_sequences(_testarticleSequences,
                                                                                     maxlen=_padLength,
                                                                                     padding='post')
            print('    Length of training set articles: %s' % _padLength)
            print('    Number of training set articles: %s' % len(_articleSequencesPadded))
            print('    Length of test set articles: %s' % _padLength)
            print('    Number of test set articles: %s' % len(_testarticleSequencesPadded))

            _tz = time.time()
            print('\nSequence padding time: %0.3f\tTotal time elapsed: %0.3f seconds\n' % (_tz - _ta, _tz - _t0))


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
            _leanVals = np.array([_leanValuesDict[k] for k in _df.lean])

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
        _ta = time.time()

        if _numFolds == 0:    # _numFolds = 0 means do a simple 90/10 split.
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

                # Get the results from the model
                _historyDict = _history.history
                _facc.append(_historyDict['categorical_accuracy'])
                _fval_acc.append(_historyDict['val_categorical_accuracy'])
                _floss.append(_historyDict['loss'])
                _fval_loss.append(_historyDict['val_loss'])

                # Remove the current model from the tensorflow backend to prepare for the next fold.
                # If we don't remove the existing model, it is retained in tensorflow and must be
                # reconciled with any subsequent models created.  Even though the older model is not
                # used when a new model is created, it adds overhead to the processing.
                #
                # Also remove the model from keras, and perform some garbage cleanup.
                if j < _numFolds:
                    keras.backend.clear_session()
                    del model
                junk = gc.collect()
                print('Garbage Collection: %s objects collected' % junk)


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

        _tz = time.time()
        print('\nTraining/validation time: %0.3f\tTotal time elapsed: %0.3f seconds\n' % (_tz - _ta, _tz - _t0))

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

    # Generate a confusion matrix for the last cross validation run
    _valPred = model.predict(_articleSequencesPadded[_val], batch_size=128, verbose=1)

    np.shape(_valPred)
    np.shape(_leanVals[_val])
    _trainConfusionMatrix = confusion_matrix(_leanVals[_val], np.argmax(_valPred, axis=1))
    _leans = ['left', 'left-center', 'least', 'right-center', 'right']
    with open('cnnTrainSetConfusionMatrix.txt', 'wt') as _outfile:
        _outfile.write('\tPredict Left\tPredict Left-Center\tPredict Least\tPredict Right Center\tPredict Right\n')
        _j = 0
        for _r in _trainConfusionMatrix:
            _outfile.write('%s\t%d\t%d\t%d\t%d\t%d\n' % (('Actual ' + _leans[_j]), _r[0], _r[1], _r[2], _r[3], _r[4]))
            _j += 1
    _trainOutput = list(zip(_df.iloc[_val].id, np.argmax(_valPred, axis=1), _leanVals[_val]))
    with open('cnnTrainSetPredictionResults.txt', 'wt') as _outfile:
        _outfile.write('ID\tPredicted Value\tActual Value\n')
        for _r in _trainOutput:
            _outfile.write('%d\t%d\t%d\n' % (_r[0], _r[1], _r[2]))

    # # Examine the outputs of the convolution layer
    # np.set_printoptions(threshold=sys.maxsize)
    # layerNames = ['embeddingLayer', 'convolutionLayer', 'maxPoolingLayer', 'denseLayer', 'outputLayer']
    #
    # layerID = 1
    # articleIdx = 0
    #
    # article = _articleSequencesPadded[_val[articleIdx]][np.newaxis, ...]
    # intermediateModel = keras.Model(inputs=model.input,
    #                                 outputs=model.get_layer(layerNames[layerID]).output)
    # intermediateOutput = intermediateModel.predict(article)
    #
    # for j in range(np.shape(intermediateOutput)[2]):
    #     pos = np.argmax(intermediateOutput[0,:,j])
    #     print('Filter %2d:  "%s"' % (j, ' '.join([t.index_word[x] for x in article[0][pos:pos+_row.convolutionKernel]])))


    # Apply the model to the test set data
    #
    if _runtest:
        _ta = time.time()
        print('Training on Full Training Set')
        keras.backend.clear_session()
        del model

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
        _history = model.fit(_articleSequencesPadded,
                             np.array([_leanVectorDict[k] for k in _leanVals]),
                             epochs=_maxEpoch, batch_size=128,
                             verbose=_verbose)

        # Predict results using the test data
        print('Applying Model to Full Test Set')
        print(_testarticleSequencesPadded)
        _predictions = model.predict(_testarticleSequencesPadded, batch_size = 128, verbose=1)
        print(_predictions)
        print(_testarticleSequencesPadded.shape)

        # Examine results from test set prediction
        _testLean = np.array([_leanValuesDict[x] for x in _dfx.lean])
        _testPred = np.argmax(np.array(_predictions), axis=1)
        print(_testLean)
        print(_testPred)
        print(float(sum(_testLean == _testPred)) / float(len(_testPred)))
        _testConfusionMatrix = confusion_matrix(_testLean, _testPred)
        _leans = ['left', 'left-center', 'least', 'right-center', 'right']
        with open('cnnTestSetConfusionMatrix.txt', 'wt') as _outfile:
            _outfile.write('\tPredict Left\tPredict Left-Center\tPredict Least\tPredict Right Center\tPredict Right\n')
            _j = 0
            for _r in _testConfusionMatrix:
                _outfile.write('%s\t%d\t%d\t%d\t%d\t%d\n' % (('Actual '+_leans[_j]), _r[0], _r[1], _r[2], _r[3], _r[4]))
                _j += 1
        _testOutput = list(zip(_dfx.id, _testPred, _testLean))
        with open('cnnTestSetPredictionResults.txt', 'wt') as _outfile:
            _outfile.write('ID\tPredicted Value\tActual Value\n')
            for _r in _testOutput:
                _outfile.write('%d\t%d\t%d\n' % (_r[0], _r[1], _r[2]))
        _tz = time.time()
        print('\nTest set evaluation time: %0.3f\tTotal time elapsed: %0.3f seconds\n' % (_tz - _ta, _tz - _t0))

    _t99 = time.time()
    print('\n\nRun completed.  Total time: %0.3f seconds' % (_t99 - _t0))


if __name__ == '__main__':
    _parser = argparse.ArgumentParser(description='Grid search for convolution neural net model.')
    _parser.add_argument('gridfile', type=str, default='gridSearch.csv',
                         help='Path/file to the grid search parameter CSV file to use.')
    _parser.add_argument('-f', '--folds', type=int, default=5, help='Number of cross-validation folds to perform')
    _parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of epochs to run in the neural net')
    _parser.add_argument('-v', '--verbose', type=int, default=1, help='Verbosity of model fitting ' +
                         '0: no output, 1: progress bar, 2: epoch only')
    _parser.add_argument('-T', '--runtest', action='store_true', default=False,
                         help='Run model against test data in database')
    _parser.add_argument('-Z', '--gpuid', type=int, default=None,
                         help='ID of the GPU to use (for none specified, skip this argument)')
    _args = _parser.parse_args()

    print(_args)

    main(_args.gridfile, _args.folds, _args.epochs, _args.verbose, _args.runtest, _args.gpuid)

