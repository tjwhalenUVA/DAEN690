#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 10:51:07 2019

@author: davidwilbur
"""

import sqlite3
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import csv

#Setting up paths (redirect path to db)
_thisFile = os.path.dirname(os.path.abspath('__file__'))
_dbfile = r'%s/articles_zenodo.db' % _thisFile.replace("EDA","Article Collection")
_deletefile = r'%s/toBeDeleted.csv' % _thisFile
_updateBias = r'%s/toBeUpdated.csv' % _thisFile




info = csv.reader(open(_deletefile))  
for row in info:
    row = row[0]
    sql = "DELETE FROM lean WHERE url LIKE '%{0}%';".format(row)
    _db = sqlite3.connect(_dbfile)
    _cursor = _db.cursor()
    _cursor.execute(sql)
    _db.commit()
    _cursor.close()

info = csv.reader(open(_updateBias)) 

query = "SELECT content.id, lean.id, content.`published-at`, lean.bias \
         FROM content, lean \
         WHERE content.id=lean.id AND \
               content.`published-at` IS NOT NULL AND \
               content.`published-at` >= '2009-01-01' ;"
lean = pd.read_sql_query(query, _db)




