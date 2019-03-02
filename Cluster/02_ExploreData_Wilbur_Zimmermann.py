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
# Data Exploration
#         This script reads in the timestamped .csv file from the data ingest
#         script and captures useful information about the corpus of resumes.
#         This information will help us during the conditioning phase to make 
#         sure we are working with good resume data. 
#
# Outputs:
#         (1.) Number of resumes in the corpus
#         (2.) The min, max, average, median, first quartile and third 
#              quartile
#         (3.) Formatted histograms and boxplots visually representing the 
#              descriptive statistics identified above.
#         (4.) Prints to standard out of actual resumes for min, max and other 
#              selected values.  This will help in deciding whether or not 
#              we should selectively prune the resume list.
#         (5.) Formatted histograms and boxplots of the pruned resume data.
#         (6.) The result of this .py file is a timestamped .csv file in the
#             'script_output' directory that is a modified list of resumes 
#              with problem resumes removed. The output file form is 
#              YYYYMMDD-HHMMSS_long_short_removed_resumes_FINAL.csv
#
# This script takes about 3 minutes to run.
###############################################################################
#%%
# Remove variables, ensures we are not using old values.
print('\nExplore Raw Data: Cleaning up variable names')
from IPython import get_ipython
get_ipython().magic('reset -sf') 
#%%
import time
begin_total_time = time.time()
#%%
# Required python packages
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from Functions_Wilbur_Zimmermann import tokenize_only
#%%
# Set global path variables
directory = os.path.dirname('__file__')
input_file = os.path.join(directory,'..','script_output')
output = os.path.join(directory,'..','script_output')
# %%
# Read in the .csv file created at the end of STEP 1.
print('\nExplore Raw Data: Reading in consolidated list of resumes from '
      'ingest.', '\n', os.path.realpath(input_file))

r = re.compile(r'\d{8}-\d{6}\_compiled_resumes_FINAL\.csv$')
latest_file = max(filter(r.search,os.listdir(input_file)))

print('\nExplore Data: The file being read in is:',latest_file)

resumes_df = pd.read_csv(os.path.join(output,latest_file))
# %%
# Get the number of resumes included in the corpus.
print('\nExplore Raw Data: The number of resumes in the corpus is:',
      len(resumes_df))
#%%
# Count the number of raw tokens in the dataset before removing things like
# tab charaters and other non-alpha numerics.
# This section of the script runs in about 1.4 minutes.

print('\nExplore Raw Data: This section of the code creates tokens from the '
      'raw resume data. Tokenization takes approximately two minutes.')

resumes_df['tokens'] = resumes_df.Resumes.map(tokenize_only)
resumes_df['token_count_raw'] = resumes_df.tokens.str.len()

unique_tokens = len(pd.value_counts(np.hstack(resumes_df.tokens.values)))

#resumes_df.loc[resumes_df.token_count_raw == max(resumes_df.token_count_raw),
#               'Resumes']
#len(set(resumes_df.tokens[7659]))
#resumes_df.loc[resumes_df.token_count_raw == min(resumes_df.token_count_raw),
#               'Resumes']
#len(set(resumes_df.tokens[5223]))


print('\nExplore Raw Data: The total number of tokens in the '
      'raw set of resume files is: ', resumes_df.token_count_raw.sum())

print('\nExplore Raw Data: The total number of unique '
      'tokens in the raw set of resume files is: ',unique_tokens)

print('\nExplore Raw Data: The shortest resume is: ',
      resumes_df.token_count_raw.min())

print('\nExplore Raw Data: The number of unique tokens in the shortest resume '
      'is: ', len(set(resumes_df.tokens[5223])))

print('\nExplore Raw Data: The longest resume is: ', 
      resumes_df.token_count_raw.max())

print('\nExplore Raw Data: The number of unique tokens in the longest resume '
      'is: ', len(set(resumes_df.tokens[7659])))

print('\nExplore Raw Data: The average resume length is: ',
      resumes_df.token_count_raw.mean())

print('\nExplore Raw Data: The median resume length is: ',
      resumes_df.token_count_raw.median())

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
resumes_df.Resumes = resumes_df.Resumes.replace(nonAlpha, ' ')
Zero = re.compile('[0]')
resumes_df.Resumes = resumes_df.Resumes.replace(Zero, ' ')

print('\nExplore Updated Data: This section of the code creates tokens from ' 
      'the updated resume data. Tokenization takes approximately two minutes.')

resumes_df['tokens_updated'] = resumes_df.Resumes.map(tokenize_only)
resumes_df['token_count_updated'] = resumes_df.tokens_updated.str.len()

unique_tokens_updated = len(pd.value_counts(np.hstack(
                                            resumes_df.tokens_updated.values)))

resumes_df.loc[resumes_df.token_count_updated 
               == max(resumes_df.token_count_updated),'Resumes']
#duration = round((time.time() - begin_time)/60, 2)
#print('The tokenizer took {} seconds to run.'.format(duration))

print('\nExplore Updated Data: The total number of tokens in the updated set '
      'of resume files is:',resumes_df.token_count_updated.sum())

print('\nExplore Updated Data: Updated token count: The total number of '
      'unique tokens in the updated set of resume files is:',
      unique_tokens_updated)

# %%
print('\nPreparing and saving histogram and bar charts of the updated resume '
      'dataset.')

plt.rcParams['figure.figsize'] = (10, 10)
ax1 = plt.subplot(211)
plt.title('Token Histogram')
plt.hist(resumes_df.token_count_updated, bins=500, color='gray')
plt.xticks(np.arange(0, 13000, 100), rotation='vertical')
plt.ylabel('Number of Resumes per Token Count')
plt.setp(ax1.get_xticklabels(), fontsize=10)

minTcount = resumes_df.token_count_updated.min()
firstQTcount = resumes_df.token_count_updated.quantile(0.25)
medTcount = resumes_df.token_count_updated.median()
meanTcount = resumes_df.token_count_updated.mean()
thirdQTcount = resumes_df.token_count_updated.quantile(0.75)
maxTcount = resumes_df.token_count_updated.max()

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
                '%s_ExploreData_nonAlpha_token_count.png' % timeStr)
plt.savefig(combined_token_count_png, dpi=100)
#%%
print('Reviewing the histogram and barplot we can see that there are quite '
      'a few resumes that are way out to the right of the mean.  Lets go '
      'figure out what if any issues there are.')

resumes_df.loc[resumes_df.token_count_updated 
               == max(resumes_df.token_count_updated),'Resumes']
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
      'is:',len(resumes_df))
#%%
print('\nExplore Shortened Data: Re-examining the number of tokens after '
      'removing resumes that are likely to contain duplicates data or '
      'resumes that are likely incomplete' )

unique_tokensx2 = len(pd.value_counts(
                      np.hstack(resumes_df.tokens_updated.values)))

#duration = round((time.time() - begin_time)/60, 2)
#print('The tokenizer took {} seconds to run.'.format(duration))

print('\nExplore Shortened Data: Raw token count: The total number of tokens '
      'in the raw set of resume files is: ',
      resumes_df.token_count_updated.sum())

print('\nExplore Shortened Data: Unique token count: The total number of '
      'unique tokens in the raw set of resume files is: ',unique_tokensx2)
# %%
# Plot the histogram and box plot of the token count.
print('\nExplore Shortened Data: Preparing and saving histogram and bar '
      'charts of the shortened resume dataset.')

plt.rcParams['figure.figsize'] = (10, 10)
ax1 = plt.subplot(211)
plt.title('Token Histogram')
plt.hist(resumes_df.token_count_updated, bins=500, color='gray')
plt.xticks(np.arange(0, 2100, 100), rotation='vertical')
plt.ylabel('Number of Resumes per Token Count')
plt.setp(ax1.get_xticklabels(), fontsize=10)

minTcount = resumes_df.token_count_updated.min()
firstQTcount = resumes_df.token_count_updated.quantile(0.25)
medTcount = resumes_df.token_count_updated.median()
meanTcount = resumes_df.token_count_updated.mean()
thirdQTcount = resumes_df.token_count_updated.quantile(0.75)
maxTcount = resumes_df.token_count_updated.max()


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

#%%

# Writing dataframe to a .csv file that can be read-in at a later point in
# time. 
timeStr = time.strftime('%Y%m%d-%H%M%S')

print('\nExplore Final Data: Writing dataframe to .csv file.',
      '%s_long_short_removed_resumes_FINAL.csv' % (timeStr) )

cols_to_keep = ['FileNames','Resumes']
resumes_df[cols_to_keep].to_csv(
        os.path.join(output,
                     '%s_long_short_removed_resumes_FINAL.csv' % (timeStr)), 
                     index=False)
#%%
duration = round((time.time() - begin_total_time)/60, 2)
print('\nExplore Final Data: This script ran in {} minutes.'.format(duration))