#! /usr/bin/python
from itertools import product
import pandas as pd
import sys

# Here is where we set all our grid search parameters.  Each parameter is a list of values (can be a single value, just
# enclose it in [brackets].
#
# *** Note ***  The number of test models to run is the product of the lengths of all of these parameters!  If you
#               have three possibilities (e.g. [a, b, c]) for each of ten different parameters here, you're looking
#               at running 3^10 (59,049) models!  Choose, but choose wisely!
#

# Files to use for each run
dbFile = ['./data/articles_zenodo.db']
gloveFile = ['./data/glove.6B.50d.txt']   #, './data/glove.6B.100d.txt']

# Number of words to include in our vocabulary (from 1 to who knows how large?)
vocabSize = [1000, 2500]

# Fraction of articles whose length we wish to capture completely.  Articles longer than
# this length will be truncated.  Shorter articles get post-padded with zeros. (0 to 1)
captureFraction = [0.95]

# Number of convolution filters to use.  Notionally set to match the number of dimensions
# in the GloVe word vector file being used, but it can be different.
convolutionFilters = [5, 10, 25, 50]

# Size of the convolution "window" to slide across the words in the article.
convolutionKernel = [10, 15]

# Type of activation to use for the output of the convolution operation.  'relu' is a
# commonly used activation for NLP.
convolutionActivation = ['relu']

# Size of the pooling window to pass across the convolved output.
poolSize = [5, 10]

# Whether to have a flattening layer or not (True, False)
flattenLayer = [True]    #, False]

# Number of nodes/units in the fully connected dense layer.  This is worth playing with,
# as it seriously affects the number of features/parameters in the neural net.
denseUnits = [25, 50]

# Type of activation to use for the output of the fully connected dense layer.
denseActivation = ['relu']

# Fraction of dense layer output nodes to drop out during each training epoch.  This
# helps with overfitting and forces nodes to "share the load" rather than having just
# a few nodes do all the work while the rest sit around playing poker.  (0 to 1, setting
# to zero will not include this layer in the model).
dropoutFraction = [0.00, 0.05]

# Activation for the output of the model.  Since we are dealing with categorical outputs,
# a 'softmax' activation is probably best to use.
outputActivation = ['softmax']

# Loss function to use for the model.  Conventionally we would use categorical crossentropy
# as the loss function, but others can be tried as well.
lossFunction = ['categorical_crossentropy']

# Build our dataframe.  Simply taking the itertools.product() of all the lists above creates
# a series of rows in a pandas dataframe.  Easy-peasy.
dfGrid = pd.DataFrame(list(product(dbFile, gloveFile, vocabSize, captureFraction,
                                   convolutionFilters, convolutionKernel, convolutionActivation,
                                   poolSize, flattenLayer, denseUnits, denseActivation, dropoutFraction,
                                   outputActivation, lossFunction)),
                      columns=['dbFile', 'gloveFile', 'vocabSize', 'captureFraction',
                               'convolutionFilters', 'convolutionKernel', 'convolutionActivation',
                               'poolSize', 'flattenLayer', 'denseUnits', 'denseActivation', 'dropoutFraction',
                               'outputActivation', 'lossFunction'])

# Add a unique ID column to the rows so that we can identify each one later.
dfGrid.insert(0, 'id', range(len(dfGrid)))

if len(sys.argv) > 1:
    # Dump our dataframe to a CSV file, specified by a command line argument for the CSV filename.
    dfGrid.to_csv(sys.argv[1], index=False, encoding='utf-8')
else:
    print('Usage:  Edit the lists of items in this python file, then run ' +
          '01_buildGridFile.py outputfile.csv')
