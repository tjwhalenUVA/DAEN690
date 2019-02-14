# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 20:34:49 2019

@author: timothy.whalen
"""
print('packages')
import xml.etree.ElementTree as ET
import os

_thisFile = os.path.dirname(os.path.abspath(__file__))

print('parse articles-training-bypublisher-20181122.xml')
_xmlfile = r'%s\articles-training-bypublisher-20181122.xml' % _thisFile
_tree = ET.parse(_xmlfile)
_root = _tree.getroot()

print('extract article content')
#Create function to return the text between the <article> tag
def gettext(elem):
    text = elem.text or ""
    for subelem in elem:
        text = text + gettext(subelem)
        if subelem.tail:
            text = text + subelem.tail
    return text

#Create list of articles in xml file
_articles = _root.getchildren()
#Dictionary for storing returned text from articles
_artDict = {}
#Loop through and extract article info and text
i = 1
for a in _articles:
    i += 1
    tmp = a.attrib
    tmp['text'] = gettext(a)
    _artDict[a.attrib['id']] = tmp

print('connect to db')
#Write data to SQLlite DB
import sqlite3
dbfile = r'C:\Users\e481340\Documents\GMU MASTERS\DAEN 690\DAEN690\Article Collection\articles_zenodo.db'
_db = sqlite3.connect(dbfile)
_cursor = _db.cursor()

print('insert articles to db')
def post_row(conn, tablename, rec):
    keys =  '[' + '],['.join(value.keys()) + ']'
    question_marks = ','.join(list('?'*len(rec)))
    values = tuple(rec.values())
    conn.execute('INSERT INTO '+tablename+' ('+keys+') VALUES ('+question_marks+')', values)

for key, value in _artDict.items():
    print(key)
    post_row(_cursor, 'content', value)

print('commit writes')
_db.commit()

