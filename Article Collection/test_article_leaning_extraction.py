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
[url] STRING,
[labeled-by] STRING);''')

print('insert articles to db')
for key, value in _artDict.items():
    print(key)
    post_row(_cursor, 'test_lean', value)
print('commit writes')
_db.commit()