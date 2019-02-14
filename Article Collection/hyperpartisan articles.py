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

#Child 0 is the first article in the xml file
root.getchildren()[0]
root.getchildren()[0].tag
root.getchildren()[0].attrib

#Children of child 0 are the article content
root.getchildren()[0].getchildren()
#Exploring whats in the text
root.getchildren()[0].getchildren()[0].tag
root.getchildren()[0].getchildren()[0].attrib
root.getchildren()[0].getchildren()[0].text
#Sentence cuts off so next child is tag
root.getchildren()[0].getchildren()[0].getchildren()[0].tag
root.getchildren()[0].getchildren()[0].getchildren()[0].attrib
root.getchildren()[0].getchildren()[0].getchildren()[0].text
root.getchildren()[0].getchildren()[0].getchildren()[0].tail


#Print out everything for first article
print(ET.tostring(root.getchildren()[0], encoding='utf8').decode('utf8'))




gettext(root.getchildren()[0])
