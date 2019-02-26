# -*- coding: utf-8 -*-
print('packages')
import xml.etree.ElementTree as ET
import os

_thisFile = os.path.dirname(os.path.abspath('__file__'))

print('parse ground-truth-validation-bypublisher-20181122.xml')
_xmlfile = r'%s/ground-truth-validation-bypublisher-20181122.xml' % _thisFile
_tree = ET.parse(_xmlfile)
_root = _tree.getroot()

#Create list of articles in xml file
_articles = _root.getchildren()
#Dictionary for storing returned metadata for articles
_artDict = {}
#Loop through and extract article info and text
i = 0
for a in _articles:
    _artDict[i] = a.attrib
    i += 1

print('setting up db')
#Write data to SQLlite DB
import sqlite3
_dbfile = r'%s/articles_zenodo.db' % _thisFile

#Function to insert data into db table (make this a module avaiable between both)
def post_row(conn, tablename, rec):
    keys =  '[' + '],['.join(rec.keys()) + ']'
    question_marks = ','.join(list('?'*len(rec)))
    values = tuple(rec.values())
    conn.execute('INSERT INTO '+tablename+' ('+keys+') VALUES ('+question_marks+')', values)

#Check if DB exists
#if it does just print message; if it doesn't create it and fill it up
print('creating test_lean table')
_db = sqlite3.connect(_dbfile)
_cursor = _db.cursor()
_cursor.execute('''CREATE TABLE test_lean
([id] STRING PRIMARY KEY NOT NULL,
[hyperpartisan] STRING,
[bias] STRING,
[bias_final] STRING,
[url] STRING,
[url_keep] STRING,
[labeled-by] STRING);''')

print('insert articles to db')
for key, value in _artDict.items():
    print(key)
    post_row(_cursor, 'test_lean', value)
print('commit writes')
_db.commit()

# Update bias_final to reflect MBFC ratings
# Update bias_final to reflect MBFC ratings
import pandas as pd
_updateBias = r'%s/toBeUpdated.csv' % _thisFile
info = pd.read_csv(_updateBias, header = None )


for index,row in info.iterrows():
    row0 = row[0]
    row1 = row[1]
    sql = "UPDATE test_lean SET bias_final = '{1}' WHERE url LIKE '%{0}%';".format(row0, row1);
    sql2 = "UPDATE test_lean SET bias_final = bias WHERE bias_final IS NULL;"
    _db = sqlite3.connect(_dbfile)
    _cursor = _db.cursor()
    _cursor.execute(sql)
    _cursor.execute(sql2)
    _db.commit()
    _cursor.close()