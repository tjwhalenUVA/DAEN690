# -*- coding: utf-8 -*-
print('packages')
from bs4 import BeautifulSoup
import requests
import os

#Pages to scrape
_pages = {}
_pages['left'] = 'https://mediabiasfactcheck.com/left/'
_pages['leftcenter'] = 'https://mediabiasfactcheck.com/leftcenter/'
_pages['center'] = 'https://mediabiasfactcheck.com/center/'
_pages['rightcenter'] = 'https://mediabiasfactcheck.com/right-center/'
_pages['right'] = 'https://mediabiasfactcheck.com/right/'

#Lists of sources in bias groups
_sources = {}
_sources['left'] = []
_sources['leftcenter'] = []
_sources['center'] = []
_sources['rightcenter'] = []
_sources['right'] = []

#Loop through biases
for bias in _pages.keys():
    print('----------'+bias+'----------')
    #Get page conents
    _page = requests.get(_pages[bias])#replace with loop
    # Create a BeautifulSoup object
    _soup = BeautifulSoup(_page.text, 'html.parser')
    #Get DIV with links in it
    _link_div = _soup.find(class_='entry clearfix')
    # Get 2nd p section
    _link_p = _link_div.find_all('p')
    #Get all links which are the sources
    _link_a = _link_p[1].find_all('a')
    for _l in _link_a:
        print(_l.text)
        _src_page = requests.get(_l.get('href'))
        _src_soup = BeautifulSoup(_src_page.text, 'html.parser')
        for p in _src_soup.find_all('p'):
            if "Source:" in p.text:
                try:
                    _sources[bias].append(p.find_all('a')[0].get('href'))
                except:
                    pass

import pandas as pd
_mbfc_df = pd.DataFrame(columns=['bias', 'sourceURL'])

for bias in _sources.keys():
    _mbfc_df = pd.concat([_mbfc_df,
                          pd.DataFrame(data = {'bias': [bias] * len(_sources[bias]),
                                               'sourceURL': _sources[bias]})],
                         axis=0)


