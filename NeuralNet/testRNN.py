# -*- coding: utf-8 -*-

# Test for running RNN - recurrent neural net
#
# This code should extract the article ID, leaning, and text from our SQLite database,
# convert the text for each article into a series of embedded word vectors (padding
# as necessary), and feed the results into a recurrent neural net.
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
_cursor.execute('SELECT ln.id, ln.bias, cn.text ' +
                'from lean ln, content cn ' +
                'where ln.id == cn.id')
_rows = _cursor.fetchall()
