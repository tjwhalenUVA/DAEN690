# -*- coding: utf-8 -*-
print('packages')
import xml.etree.ElementTree as ET
import os

_thisFile = os.path.dirname(os.path.abspath('__file__'))

print('parse ground-truth-training-bypublisher-20181122.xml')
_xmlfile = r'%s/ground-truth-training-bypublisher-20181122.xml' % _thisFile
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