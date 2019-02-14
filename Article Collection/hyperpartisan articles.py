# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 20:34:49 2019

@author: timothy.whalen
"""

import untangle

file = r'C:\Users\e481340\Documents\GMU MASTERS\DAEN 690\Data\zenodoData\articles-training-bypublisher-20181122\articles-training-bypublisher-20181122.xml'

o = untangle.parse(file)


o.articles.article[0]['id']
o.articles.article[0]['published-at']
o.articles.article[0]['title']
o.articles.article[0]
