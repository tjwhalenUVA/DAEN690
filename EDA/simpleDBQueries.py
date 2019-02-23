#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 16:36:58 2019

@author: davidwilbur
"""

import sqlite3
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime

#Setting up paths (redirect path to db)
_thisFile = os.path.dirname(os.path.abspath('__file__'))
_dbfile = r'%s/articles_zenodo.db' % _thisFile.replace("EDA","Article Collection")

#Connecting to database
_db = sqlite3.connect(_dbfile)
query = "SELECT `published-at` \
         FROM content;"
lean = pd.read_sql_query(query, _db)

_db = sqlite3.connect(_dbfile)
query = "SELECT `published-at` \
         FROM content \
         WHERE `published-at` IS NOT NULL AND \
               `published-at` >= '2009-01-01';"
lean = pd.read_sql_query(query, _db)


_db = sqlite3.connect(_dbfile)
query = "SELECT `published-at` \
         FROM content \
         WHERE `published-at` IS NULL;"
lean = pd.read_sql_query(query, _db)

_db = sqlite3.connect(_dbfile)
query = "SELECT content.id, lean.id, content.`published-at`, lean.bias \
         FROM content, lean \
         WHERE content.id=lean.id AND \
               content.`published-at` IS NOT NULL AND \
               content.`published-at` >= '2009-01-01' AND \
               content.`published-at` <= '2009-12-31' AND \
               lean.bias = 'right' ;"
lean = pd.read_sql_query(query, _db)

_db = sqlite3.connect(_dbfile)
query = "SELECT content.id, lean.id, content.`published-at`, lean.bias \
         FROM content, lean \
         WHERE content.id=lean.id AND \
               content.`published-at` IS NOT NULL AND \
               content.`published-at` >= '2009-01-01' AND \
               content.`published-at` <= '2009-12-31' AND \
               lean.bias = 'right-center' ;"
lean = pd.read_sql_query(query, _db)

####
####

