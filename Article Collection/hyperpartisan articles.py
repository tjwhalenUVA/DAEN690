# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 20:34:49 2019

@author: timothy.whalen
"""

import xml.etree.ElementTree as ET

xmlfile = r'C:\Users\e481340\Documents\GMU MASTERS\DAEN 690\Data\zenodoData\articles-training-bypublisher-20181122\articles-training-bypublisher-20181122.xml'

tree = ET.parse(xmlfile)
root = tree.getroot()

#Root is the collection of all articles
root.tag
root.tail
root.text
root.attrib

#Create function to return the text between the <article> tag
def gettext(elem):
    text = elem.text or ""
    for subelem in elem:
        text = text + gettext(subelem)
        if subelem.tail:
            text = text + subelem.tail
    return text

#Create list of articles in xml file
articles = root.getchildren()
#Dictionary for storing returned text from articles
artDict = {}
#Loop through and extract article info and text
for a in articles:
    print(a)
    tmp = a.attrib
    tmp['text'] = gettext(a)
    artDict[a.attrib['id']] = tmp






