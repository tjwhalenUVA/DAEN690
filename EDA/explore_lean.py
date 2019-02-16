# -*- coding: utf-8 -*-
print('---importing packages---')
import sqlite3
import os
import pandas as pd

#Setting up paths (redirect path to db)
_thisFile = os.path.dirname(os.path.abspath('__file__'))
_dbfile = r'%s/articles_zenodo.db' % _thisFile.replace("EDA", "Article Collection")

#Connecting to database
_db = sqlite3.connect(_dbfile)
print('---read in [lean] table---')
query = "SELECT * FROM lean;"
lean = pd.read_sql_query(query, _db)

print(lean.head(10))