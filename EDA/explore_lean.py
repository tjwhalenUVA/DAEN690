# -*- coding: utf-8 -*-
print('---importing packages---')
import sqlite3
import os

#Setting up paths
_thisFile = os.path.dirname(os.path.abspath('__file__'))
_dbfile = r'%s/articles_zenodo.db' % _thisFile.replace("EDA", "Article Collection")

#Connecting to database
_db = sqlite3.connect(_dbfile)
