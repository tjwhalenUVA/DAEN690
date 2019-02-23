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

#_db = sqlite3.connect(_dbfile)
#_cursor = _db.cursor()

info = csv.reader(open(_deletefile))  
for row in info :
    row = str(row[0]).strip(' ')
    print(row)
    print("DELETE FROM lean WHERE url LIKE '%{0}%';".format(row))
    
    _db = sqlite3.connect(_dbfile)
    _cursor = _db.cursor()
    _cursor.execute("DELETE FROM lean WHERE url LIKE '%row[0]%' ;")
    _db.commit()
    _cursor.close()



query = "SELECT content.id, lean.id, content.`published-at`, lean.bias \
         FROM content, lean \
         WHERE content.id=lean.id AND \
               content.`published-at` IS NOT NULL AND \
               content.`published-at` >= '2009-01-01' ;"
lean = pd.read_sql_query(query, _db)




