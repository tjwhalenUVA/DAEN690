# -*- coding: utf-8 -*-
print('packages')
import xml.etree.ElementTree as ET
import os
#example comment
_thisFile = os.path.dirname(os.path.abspath('__file__'))

print('parse articles-validation-bypublisher-20181122.xml')
_xmlfile = r'%s/articles-validation-bypublisher-20181122.xml' % _thisFile
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
for a in _articles:
    tmp = a.attrib
    tmp['text'] = gettext(a)
    _artDict[a.attrib['id']] = tmp

print('setting up db')
#Write data to SQLlite DB
import sqlite3
_dbfile = r'%s/articles_zenodo.db' % _thisFile

#Function to insert data into db table
def post_row(conn, tablename, rec):
    keys =  '[' + '],['.join(rec.keys()) + ']'
    question_marks = ','.join(list('?'*len(rec)))
    values = tuple(rec.values())
    conn.execute('INSERT INTO '+tablename+' ('+keys+') VALUES ('+question_marks+')', values)

#Check if DB exists
#if it does just print message; if it doesn't create it and fill it up
print('connect to db')
_db = sqlite3.connect(_dbfile)
_cursor = _db.cursor()
print('creating test_content table')
_cursor.execute('''CREATE TABLE test_content 
                ([id] STRING PRIMARY KEY NOT NULL,
                [published-at] DATE,
                [title] STRING,
                [text] STRING);''')

print('insert articles to db')
for key, value in _artDict.items():
    print(key)
    post_row(_cursor, 'test_content', value)
print('commit writes')
_db.commit()