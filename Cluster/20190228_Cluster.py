# %%###########################################################################
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
#
# Topic Modeling: An Analysis of Corporate Resumes
# Submitted for Successful Completion of CS584
# Authored By: David Wilbur and Russell Zimmermann
#
# CS584 Theory and Applications of Data Mining
# Daniel Barbará, Ph.D.
# George Mason University
# Volgenau School of Engineering
# Data Analytics Engineering
#
###############################################################################
#
# Data Ingest:
#         This script reads each resume into a Pandas dataframe row. The
#         resulting dataframe is 8,265 rows by 1 column. 
#         
#         Setup the following directory structure for this script to run 
#         properly: 
#            * 'Some Project Name'
#                 ** src
#                 ** data
#                 ** script_output
#          
#         The 'data' directory should contain individual resume files in .csv 
#         format. The 'src' directory should contain this .py file
#
#         Once the data and script are positioned, this script can be run in 
#         its entirety. Status messages are printed to standard out.
#
# Output:
#         (1.) The output of this .py file is a timestamped .csv file in the
#              'script_output' directory. The .csv file contains all of the 
#              individual resume files. The output file name is of the form 
#              YYYYMMDD-HHMMSS_compiled_resumes_FINAL.csv
#
# This script runs in approximately 1 second.
#
###############################################################################
#%%
# Remove variables, ensures we are not using old values.
print('\nIngest Data: Cleaning up variable names')
from IPython import get_ipython
get_ipython().magic('reset -sf') 
#%%
import time
begin_time = time.time()
#%%
# Required python packages

import os
print('Importing packages')
import re
import glob
import numpy as np
import pandas as pd
import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from Functions_Wilbur_Zimmermann import tokenize_only
#%%
# Set global path variables
directory = os.path.dirname('__file__')
input_file = os.path.join(directory,'..','Article Collection')
_dbFile = os.path.join(input_file,'articles_zenodo.db')

#%%
# Set up our connection to the database
#
_db = sqlite3.connect(_dbFile)
_cursor = _db.cursor()

# Pull the relevant bits of data out of the database and
# store them in a pandas dataframe.
#
print('Pulling article IDs, leanings, and text from database')
_cursor.execute("SELECT ln.id, ln.bias_final, cn.text " +
                "FROM train_lean ln, train_content cn " +
                "WHERE cn.`published-at` >= '2009-01-01' AND ln.id == cn.id AND ln.url_keep='1'")
_df = pd.DataFrame(_cursor.fetchall(), columns=('id', 'lean', 'text'))
_db.close()
# %%# %%
# Get the number of resumes included in the corpus.
print('\nExplore Raw Data: The number of resumes in the corpus is:',
      len(_df))
#%%
# Count the number of raw tokens in the dataset before removing things like
# tab charaters and other non-alpha numerics.
# This section of the script runs in about 1.4 minutes.

print('\nExplore Raw Data: This section of the code creates tokens from the '
      'raw resume data. Tokenization takes approximately two minutes.')

_df['tokens'] = _df.text.map(tokenize_only)
_df['token_count_raw'] = df.tokens.str.len()

unique_tokens = len(pd.value_counts(np.hstack(_df.tokens.values)))

#resumes_df.loc[resumes_df.token_count_raw == max(resumes_df.token_count_raw),
#               'Resumes']
#len(set(resumes_df.tokens[7659]))
#resumes_df.loc[resumes_df.token_count_raw == min(resumes_df.token_count_raw),
#               'Resumes']
#len(set(resumes_df.tokens[5223]))


print('\nExplore Raw Data: The total number of tokens in the '
      'raw set of resume files is: ', _df.token_count_raw.sum())

print('\nExplore Raw Data: The total number of unique '
      'tokens in the raw set of resume files is: ',unique_tokens)

print('\nExplore Raw Data: The shortest resume is: ',
      _df.token_count_raw.min())

print('\nExplore Raw Data: The number of unique tokens in the shortest resume '
      'is: ', len(set(_df.tokens[5223])))

print('\nExplore Raw Data: The longest resume is: ', 
      _df.token_count_raw.max())

print('\nExplore Raw Data: The number of unique tokens in the longest resume '
      'is: ', len(set(_df.tokens[7659])))

print('\nExplore Raw Data: The average resume length is: ',
      _df.token_count_raw.mean())

print('\nExplore Raw Data: The median resume length is: ',
      _df.token_count_raw.median())

#%%
# Here we want to break each resume up into tokens then count the number of
# tokens in each resume. Except that we want to take a more nuanced approach 
# by removing non-alpha characters and any zeros. 

print('\nExplore Updated Data: Here we take a nuanced approach to the '
      'exploration of the resume files. Based on review of the raw resume '
      'files we remove non-alpha characters and zeros. Following removal we ' 
      're-tokenize the resume files and explore how this affects our dataset.')

print('\nExplore Updated Data: Removing non-alpha characters and zeros')

nonAlpha = re.compile('[^a-zA-Z]+')
_df.text = _df.text.replace(nonAlpha, ' ')
Zero = re.compile('[0]')
_df.text = _df.text.replace(Zero, ' ')

print('\nExplore Updated Data: This section of the code creates tokens from ' 
      'the updated resume data. Tokenization takes approximately two minutes.')

_df['tokens_updated'] = _df.text.map(tokenize_only)
_df['token_count_updated'] = _df.tokens_updated.str.len()

unique_tokens_updated = len(pd.value_counts(np.hstack(
                                            _df.tokens_updated.values)))

_df.loc[_df.token_count_updated 
               == max(_df.token_count_updated),'text']
#duration = round((time.time() - begin_time)/60, 2)
#print('The tokenizer took {} seconds to run.'.format(duration))

print('\nExplore Updated Data: The total number of tokens in the updated set '
      'of resume files is:',_df.token_count_updated.sum())

print('\nExplore Updated Data: Updated token count: The total number of '
      'unique tokens in the updated set of resume files is:',
      _tokens_updated)

# %%
print('\nPreparing and saving histogram and bar charts of the updated resume '
      'dataset.')

plt.rcParams['figure.figsize'] = (10, 10)
ax1 = plt.subplot(211)
plt.title('Token Histogram')
plt.hist(_df.token_count_updated, bins=500, color='gray')
plt.xticks(np.arange(0, 13000, 100), rotation='vertical')
plt.ylabel('Number of Resumes per Token Count')
plt.setp(ax1.get_xticklabels(), fontsize=10)

minTcount = _df.token_count_updated.min()
firstQTcount = _df.token_count_updated.quantile(0.25)
medTcount = _df.token_count_updated.median()
meanTcount = _df.token_count_updated.mean()
thirdQTcount = _df.token_count_updated.quantile(0.75)
maxTcount = _df.token_count_updated.max()

plt.axvline(minTcount, color='r', linestyle='--', linewidth=1)
plt.axvline(firstQTcount, color='c', linestyle='--', linewidth=1)
plt.axvline(medTcount, color='y', linestyle='--', linewidth=1)
plt.axvline(meanTcount, color='g', linestyle='--', linewidth=2)
plt.axvline(thirdQTcount, color='c', linestyle='--', linewidth=1)
plt.axvline(maxTcount, color='r', linestyle='--', linewidth=1)
 
r1_line = mlines.Line2D([], [], color='r', linestyle='--', linewidth=1,
                        label='min')
c1_line = mlines.Line2D([], [], color='c', linestyle='--', linewidth=1,
                        label='first quartile')
y_line = mlines.Line2D([], [], color='y', linestyle='--', linewidth=1,
                       label='median')
g_line = mlines.Line2D([], [], color='g', linestyle='--', linewidth=2,
                       label='mean')
c2_line = mlines.Line2D([], [], color='c', linestyle='--', linewidth=1,
                        label='third quartile')
r2_line = mlines.Line2D([], [], color='r', linestyle='--', linewidth=1,
                        label='max')
plt.legend(handles=[r1_line, c1_line, y_line, g_line, c2_line, r2_line], loc=1)
plt.tight_layout()

ax2 = plt.subplot(212, sharex=ax1)
plt.title('Token Boxplot')
plt.boxplot(_df.token_count_updated, 0, 'b+', 0)
plt.xticks(rotation='vertical')
plt.xlabel('Number of Tokens Per Resume')
plt.yticks([])
plt.setp(ax2.get_xticklabels(), fontsize=10)

plt.axvline(minTcount, color='r', linestyle='--', linewidth=1)
plt.axvline(firstQTcount, color='c', linestyle='--', linewidth=1)
plt.axvline(medTcount, color='y', linestyle='--', linewidth=1)
plt.axvline(meanTcount, color='g', linestyle='--', linewidth=2)
plt.axvline(thirdQTcount, color='c', linestyle='--', linewidth=1)
plt.axvline(maxTcount, color='r', linestyle='--', linewidth=1)
 
r1_line = mlines.Line2D([], [], color='r', linestyle='--', linewidth=1,
                        label='min')
c1_line = mlines.Line2D([], [], color='c', linestyle='--', linewidth=1,
                        label='first quartile')
y_line = mlines.Line2D([], [], color='y', linestyle='--', linewidth=1,
                       label='median')
g_line = mlines.Line2D([], [], color='g', linestyle='--', linewidth=2,
                       label='mean')
c2_line = mlines.Line2D([], [], color='c', linestyle='--', linewidth=1,
                        label='third quartile')
r2_line = mlines.Line2D([], [], color='r', linestyle='--', linewidth=1,
                        label='max')
plt.legend(handles=[r1_line, c1_line, y_line, g_line, c2_line, r2_line], loc=1)
plt.tight_layout()

timeStr = time.strftime('%Y%m%d-%H%M%S')
combined_token_count_png =\
   os.path.join(directory, '..', 'script_output',
                '%s_ExploreData_nonAlpha_token_count.png' % timeStr)
plt.savefig(combined_token_count_png, dpi=100)
#%%
print('Reviewing the histogram and barplot we can see that there are quite '
      'a few resumes that are way out to the right of the mean.  Lets go '
      'figure out what if any issues there are.')

_df.loc[_df.token_count_updated 
               == max(_df.token_count_updated),'Resumes']

resumes_df.Resumes[7659]

# Ah now we see what the issue is...this resume should probably be removed from 
# the dataset. Lets see what other resumes with large word counts look like
resumes_df.loc[resumes_df.token_count_updated == 5851 ,'Resumes']
resumes_df.Resumes[8264]
resumes_df.loc[resumes_df.token_count_updated == 2280 ,'Resumes']
resumes_df.Resumes[2]

# how about low word counts
resumes_df.loc[resumes_df.token_count_updated 
               == min(resumes_df.token_count_updated),'Resumes']
resumes_df.Resumes[5223]

# print out number of resumes that are removed from corpus. 
print('\nExplore Updated Data: Removing resumes with a token count greater '
      'than 2000 and less than 75')

resumes_df = resumes_df.query('token_count_updated <= 2000')
resumes_df = resumes_df.query('token_count_updated >=75')
#%%
# Get the number of resumes included in the corpus.
print('\nExplore Shortened Data: The number of resumes in the updated corpus '
      'is:',len(_df))
#%%
print('\nExplore Shortened Data: Re-examining the number of tokens after '
      'removing resumes that are likely to contain duplicates data or '
      'resumes that are likely incomplete' )

unique_tokensx2 = len(pd.value_counts(
                      np.hstack(_df.tokens_updated.values)))

#duration = round((time.time() - begin_time)/60, 2)
#print('The tokenizer took {} seconds to run.'.format(duration))

print('\nExplore Shortened Data: Raw token count: The total number of tokens '
      'in the raw set of resume files is: ',
      _df.token_count_updated.sum())

print('\nExplore Shortened Data: Unique token count: The total number of '
      'unique tokens in the raw set of resume files is: ',unique_tokensx2)
# %%
# Plot the histogram and box plot of the token count.
print('\nExplore Shortened Data: Preparing and saving histogram and bar '
      'charts of the shortened resume dataset.')

plt.rcParams['figure.figsize'] = (10, 10)
ax1 = plt.subplot(211)
plt.title('Token Histogram')
plt.hist(_df.token_count_updated, bins=500, color='gray')
plt.xticks(np.arange(0, 2100, 100), rotation='vertical')
plt.ylabel('Number of Resumes per Token Count')
plt.setp(ax1.get_xticklabels(), fontsize=10)

minTcount = _df.token_count_updated.min()
firstQTcount = _df.token_count_updated.quantile(0.25)
medTcount = _df.token_count_updated.median()
meanTcount = _df.token_count_updated.mean()
thirdQTcount = _df.token_count_updated.quantile(0.75)
maxTcount = _df.token_count_updated.max()


plt.axvline(minTcount, color='r', linestyle='--', linewidth=1)
plt.axvline(firstQTcount, color='c', linestyle='--', linewidth=1)
plt.axvline(medTcount, color='y', linestyle='--', linewidth=1)
plt.axvline(meanTcount, color='g', linestyle='--', linewidth=2)
plt.axvline(thirdQTcount, color='c', linestyle='--', linewidth=1)
plt.axvline(maxTcount, color='r', linestyle='--', linewidth=1)
 
r1_line = mlines.Line2D([], [], color='r', linestyle='--', linewidth=1,
                        label='min')
c1_line = mlines.Line2D([], [], color='c', linestyle='--', linewidth=1,
                        label='first quartile')
y_line = mlines.Line2D([], [], color='y', linestyle='--', linewidth=1,
                       label='median')
g_line = mlines.Line2D([], [], color='g', linestyle='--', linewidth=2,
                       label='mean')
c2_line = mlines.Line2D([], [], color='c', linestyle='--', linewidth=1,
                        label='third quartile')
r2_line = mlines.Line2D([], [], color='r', linestyle='--', linewidth=1,
                        label='max')
plt.legend(handles=[r1_line, c1_line, y_line, g_line, c2_line, r2_line], loc=1)
plt.tight_layout()

ax2 = plt.subplot(212, sharex=ax1)
plt.title('Token Boxplot')
plt.boxplot(resumes_df.token_count_updated, 0, 'b+', 0)
plt.xticks(rotation='vertical')
plt.xlabel('Number of Tokens Per Resume')
plt.yticks([])
plt.setp(ax2.get_xticklabels(), fontsize=10)

plt.axvline(minTcount, color='r', linestyle='--', linewidth=1)
plt.axvline(firstQTcount, color='c', linestyle='--', linewidth=1)
plt.axvline(medTcount, color='y', linestyle='--', linewidth=1)
plt.axvline(meanTcount, color='g', linestyle='--', linewidth=2)
plt.axvline(thirdQTcount, color='c', linestyle='--', linewidth=1)
plt.axvline(maxTcount, color='r', linestyle='--', linewidth=1)
 
r1_line = mlines.Line2D([], [], color='r', linestyle='--', linewidth=1,
                        label='min')
c1_line = mlines.Line2D([], [], color='c', linestyle='--', linewidth=1,
                        label='first quartile')
y_line = mlines.Line2D([], [], color='y', linestyle='--', linewidth=1,
                       label='median')
g_line = mlines.Line2D([], [], color='g', linestyle='--', linewidth=2,
                       label='mean')
c2_line = mlines.Line2D([], [], color='c', linestyle='--', linewidth=1,
                        label='third quartile')
r2_line = mlines.Line2D([], [], color='r', linestyle='--', linewidth=1,
                        label='max')
plt.legend(handles=[r1_line, c1_line, y_line, g_line, c2_line, r2_line], loc=1)
plt.tight_layout()

timeStr = time.strftime('%Y%m%d-%H%M%S')
combined_token_count_png =\
   os.path.join(directory, '..', 'script_output',
                '%s_ExploreData_noLongShort_token_count.png' % timeStr)
plt.savefig(combined_token_count_png, dpi=100)
