# -*- coding: utf-8 -*-
print('packages')
from bs4 import BeautifulSoup
import requests
import os

#Pages to scrape
pages = {}
pages['left'] = 'https://mediabiasfactcheck.com/left/'
pages['leftcenter'] = 'https://mediabiasfactcheck.com/leftcenter/'
pages['center'] = 'https://mediabiasfactcheck.com/center/'
pages['rightcenter'] = 'https://mediabiasfactcheck.com/right-center/'
pages['right'] = 'https://mediabiasfactcheck.com/right/'

#Get page conents
page = requests.get(pages['left'])
# Create a BeautifulSoup object
soup = BeautifulSoup(page.text, 'html.parser')

#Get DIV with links in it
_link_div = soup.find(class_='entry clearfix')
# Pull text from all instances of <a> tag within BodyText div
_link_p = _link_div.find_all('p')
_link_a = _link_p.find_all('a')

print(_link_a)
print(len(_link_a))