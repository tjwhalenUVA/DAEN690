# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:58:28 2019

@author: timothy.whalen
"""
def remove_duplicates(df, training = True):
    import os
    import pandas as pd
    _thisFile = os.path.dirname(os.path.abspath('__file__'))
    if training:
        _dupFile = r'%s/duplicate_article_ids_train.csv' % _thisFile
    else:
        _dupFile = r'%s/duplicate_article_ids_test.csv' % _thisFile
    _dups = pd.read_csv(_dupFile, names=['first_id', 'dup_id'], skiprows=1)
    df = df[~df.id.isin(_dups.dup_id.tolist())]
    return(df)


#import sqlite3
#_dbfile = r'%s/articles_zenodo.db' % _thisFile.replace("EDA", "Article Collection")
#_db = sqlite3.connect(_dbfile)
#query = "SELECT * FROM test_content;"
#df = pd.read_sql_query(query, _db)
#df = remove_duplicates(df, training=False)
