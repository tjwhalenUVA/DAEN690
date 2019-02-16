# -*- coding: utf-8 -*-
import sqlite3
import os
import pandas as pd
import matplotlib.pyplot as plt

#Setting up paths (redirect path to db)
_thisFile = os.path.dirname(os.path.abspath('__file__'))
_dbfile = r'%s/articles_zenodo.db' % _thisFile.replace("EDA", "Article Collection")

#Connecting to database
_db = sqlite3.connect(_dbfile)
query = "SELECT * FROM lean;"
lean = pd.read_sql_query(query, _db)

#Check counts of data leaning
plt.figure()
pd.value_counts(lean['hyperpartisan']).plot.bar()
plt.figure()
pd.value_counts(lean['bias']).plot.bar()

#Split out and group by just the site
lean['site'] = lean.url.str.split('//', expand=True)[1].str.split('.', expand=True)[0]

plt.figure()
pd.value_counts(lean['site'])[0:20].plot.bar()