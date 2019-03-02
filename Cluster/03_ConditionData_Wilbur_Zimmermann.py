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
# DATA CONDITIONING
#      This script reads in the timestamped .csv file from the Data
#      Exploration script and
#          *** Removes stopwords
#          *** Runs Stanford University's Named Entity Recognizer and removes
#              all tokens identified as PERSONS
#          *** Using data from the US Census bureau removes all first and last 
#              names identified in the 1990 census.
#          *** Removes resumes database section headers
#          *** Removes other extraneous words identified during recursive
#              examination of K-Means and LDA clusters. 
#      Output:
#      (1.) The output of this .py file is a timestamped .csv file in the
#           'script_output' directory that is a fully conditioned set of 
#           resumes that is ready for cluster analysis. The output file form 
#           is YYYYMMDD-HHMMSS_conditioned_resumes_FINAL.csv
#
#      Each of the following clustering scripts ingest the output from this 
#      script to begin cluster analysis.
#
#      This script takes approximately 60 minutes (without NER)
#      This scriot takes approximately 5 hours to run with NER uncommented
#
#      To decrease the running time, the list of identified PERSONS from NER
#      was run in advance, saved to a CSV file and ingested here.  
###############################################################################
#%%
# Remove variables, ensures we are not using old values.
print('\nCondition Data: Cleaning up variable names')
from IPython import get_ipython
get_ipython().magic('reset -sf') 
#%%
# Code that is used to measure the running time of this script.
import time
begin_total_time = time.time()
#%%
# import packages required to run this script.
import os
import re
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from nltk.corpus import stopwords
from Functions_Wilbur_Zimmermann import tokenize_only                                      
#%%
# Set path variables. To run this script a file titled
# 20171118_person_list_updatedvXX.csv is required to be in the project file
# directory. Also to remove names found in the census data three text files 
# must be in a directory titled 'token_removal'. 

directory = os.path.dirname('__file__')
input_file = os.path.join(directory,'..','script_output')
output = os.path.join(directory,'..','script_output')
personsRemoved = os.path.join(directory,'..',
                              '20171118_person_list_updatedv27.csv')
path = os.path.join(directory, '..', 'token_removal')
censusNames = glob.glob(os.path.join(path, '*.txt'))

stanford_classifier =\
    os.path.join(directory, '..', 'parser', 'stanford-ner-2017-06-09',
                 'classifiers', 'english.all.3class.distsim.crf.ser.gz')
 
stanford_ner_path =\
    os.path.join(directory, '..', 'parser', 'stanford-ner-2017-06-09',
                 'stanford-ner.jar')
# %%
print('\nCondition Data: Reading in resumes from the data exploration script.')
print('\nCondition Data: Reading in resume files from: ','\n',
      os.path.realpath(input_file))


r = re.compile(r'\d{8}-\d{6}\_long_short_removed_resumes_FINAL\.csv$')
latest_file = max(filter(r.search,os.listdir(input_file)))

print('\nCondition Data: The filename being read in is: ', latest_file)

resumes_df = pd.read_csv(os.path.join(input_file,latest_file))

#%%
# Remove stopwords.  Stopwords originate from NLTK
print('\nCondition Data: Removing stopwords. Removing stopwords takes '
      'between 15 and 45 seconds to run depending on computer.')

cachedStopWords = stopwords.words('english')
resumes_df.Resumes = resumes_df.Resumes.apply(
                          lambda x: (' '.join([word for word in x.split() 
                          if word not in cachedStopWords])))
resumes_df.Resumes = resumes_df.Resumes.str.lower()


print('\nCondition Data: This section of the code creates tokens from ' 
      'the resume data. Tokenization takes approximately two minutes.')

resumes_df['tokens'] = resumes_df.Resumes.map(tokenize_only)
resumes_df['token_count'] = resumes_df.tokens.str.len()
unique_tokens = len(pd.value_counts(np.hstack(resumes_df.tokens.values)))

print('\nCondition Data: The total number of tokens in the raw set of '
      'resume files is:', resumes_df.token_count.sum())
print('\nCondition Data: Unique token count: The total number of unique '
      'tokens in the raw set of resume files is:',unique_tokens)
# %%
# NER removal

# Creating Tagger Object

#st = StanfordNERTagger(stanford_classifier, stanford_ner_path,
#                       encoding='utf-8')

#personList = []
#for i in range(len(resumes_df)):
#    for sent in nltk.sent_tokenize(resumes_df.Resumes[i]):
#        tokens = word_tokenize(sent)
#        tags = st.tag(tokens)
#        for tag in tags:
#            if tag[1]=='PERSON':
#                personList.append(tag[0])
                
# I think we will want to run through this list and remove those items that 
# aren't really names.
#dupRemovedpersonList = sorted(list(set(personList)))
#dupRemovedpersonList_df = pd.DataFrame(dupRemovedpersonList)
#dupRemovedpersonList_df.to_csv('20171118_person_list.csv')
#%%
# In this section of the code we are explicitly removing databae headings that 
# were included with each resume.  In an ideal world these headings would never
# have been pulled and we wouldn't have to do this. 

print('\nCondition Data: Removing database headers. Removing database headers '
      'takes about X minutes.')

resumes_df.Resumes.replace({'salary grade code': 'salarygradecode'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'network account': 'networkaccount'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'email address': 'emailaddress'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'additional attributes': 'additionalattributes'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'current hire date': 'currenthiredate'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'salary hourly indicator': 
                            'salaryhourlyindicator'},regex=True, inplace=True)
resumes_df.Resumes.replace({'engility employee class': 
                            'engilityemployeeclass'},regex=True, inplace=True)
resumes_df.Resumes.replace({'resume submitted': 'resumesubmitted'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'place of birth': 'placeofbirth'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'employee salary job history':
                            'employeesalaryjobhistory'},
                            regex=True, inplace=True)
resumes_df.Resumes.replace({'effective date': 'effectivedate'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'management structure': 'managementstructure'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'employee workforce': 'employeeworkforce'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'project id': 'projectid'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'off site': 'offsite'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'manager name': 'managername'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'manager email': 'manageremail'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'sa submit date': 'sasubmitdate'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'sa brief date': 'sabriefdate'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'latest brief date': 'latestbriefdate'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'current status': 'currentstatus'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'sa grant date': 'sagrantdate'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'clearance status': 'clearancestatus'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'polygraph current status':
                            'polygraphcurrentstatus'},regex=True, inplace=True)
resumes_df.Resumes.replace({'plc descr': 'plcdescr'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'sold to': 'soldto'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'term date': 'termdate'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'adj hire date': 'adjhiredate'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'employee class descr': 'employeeclassdescr'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'salary grade': 'salarygrade'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'full time': 'fulltime'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'security accesses': 'securityaccesses'},
                           regex=True, inplace=True)
resumes_df.Resumes.replace({'job history': 'jobhistory'},
                           regex=True, inplace=True)

#%%
# Write out CSV file after this step. Also need to consider doing a lower 
#case after this code but before writing to csv
begin_time = time.time()
start_time = time.strftime('%H:%M:%S')
print('\n\nStart time: {}  (should take about 5 mins)'.format(start_time))
print('Condition Data: Removing all persons from NER, persons from cenusus '
      'and other extraneous words identified during repetitive search for '
      'the very best clusters. This section of code takes approximately '
      '5 minutes to run.')
personsRemoved_df = pd.read_csv(personsRemoved, header = None, dtype=object)
personsRemoved_df = personsRemoved_df[0].str.lower()
personsRemoved_l= personsRemoved_df.values.tolist()
resumes_df.Resumes = resumes_df.Resumes.apply(
                          lambda x: (' '.join([word for word in x.split() 
                                               if word not in 
                                               personsRemoved_l])))

duration = round((time.time() - begin_time)/60, 2)
print('The kmeans routine took {} minutes to run.'.format(duration))

print('\nCondition Data: This section of the code creates tokens from the '
      'updated resume data. Tokenization takes approximately two minutes.')
resumes_df['tokens'] = resumes_df.Resumes.map(tokenize_only)
resumes_df['token_count'] = resumes_df.tokens.str.len()
unique_tokens = len(pd.value_counts(np.hstack(resumes_df.tokens.values)))

print('\nCondition Data, Raw token count: The total number of tokens in the '
      'raw set of resume files is: ', resumes_df.token_count.sum())

print('\nCondition Data, Unique token count: The total number of unique '
      'tokens in the raw set of resume files is:',unique_tokens)
# %%
# Using first and last names from the 1990 Census remove any remaining names
# from each resume. 
begin_time = time.time()
start_time = time.strftime('%H:%M:%S')
print('\n\nStart time: {}  (should take about x mins)'.format(start_time))

print('\nCondition Data: Reading in census .txt files from: ','\n', 
      os.path.realpath(path))
print('\n Names are extracted from the census files. Names remaining after NER'
      ' are removed from each resume')

censusNames_list = []
for file in censusNames:
    df = pd.read_csv(file, header=None, delim_whitespace=True)
    censusNames_list.append(df)
    temp_df = pd.concat(censusNames_list)
temp_df = temp_df.drop(columns = [1,2,3])
temp_df = temp_df[0].str.lower()
temp_df_l= temp_df.values.tolist()

resumes_df.Resumes = resumes_df.Resumes.apply(
                          lambda x: (' '.join([word for word in x.split()
                                               if word not in temp_df_l])))

duration = round((time.time() - begin_time)/60, 2)
print('The kmeans routine took {} minutes to run.'.format(duration))


timeStr = time.strftime('%Y%m%d-%H%M%S')
print('\nCondition Data: Writing dataframe to .csv file.', 
      '%s_census_names_removed_FINAL.csv' % (timeStr) )
resumes_df.to_csv(os.path.join(output,
                               '%s_census_names_removed_FINAL.csv' 
                               % (timeStr)),
                   index=False)
# %%
#print('\nCondition Data: Reading in resumes conditioned by NER and Census.')
#print('\nCondition Data: Reading in resume files from: ','\n',
#      os.path.realpath(input_file))

#r = re.compile(r'\d{8}-\d{6}\_census_names_removed_FINAL\.csv$')
#latest_file = max(filter(r.search,os.listdir(input_file)))

#print('\nCondition Data: The filename being read in is: ', latest_file)

#resumes_df = pd.read_csv(os.path.join(input_file,latest_file))

#%%
print('\nCondition Data: This section of the code creates tokens from the '
      'updated resume data. Tokenization takes approximately two minutes.')

resumes_df['tokens'] = resumes_df.Resumes.map(tokenize_only)
resumes_df['token_count'] = resumes_df.tokens.str.len()
unique_tokens = len(pd.value_counts(np.hstack(resumes_df.tokens.values)))

print('\nCondition Data, Raw token count: The total number of tokens in the '
      'raw set of resume files is:', resumes_df.token_count.sum())
print('\nCondition Data, Unique token count: The total number of unique '
      'tokens in the raw set of resume files is:',unique_tokens)
# %%
# Plot the histogram and box plot of the token count.
# Plot the histogram and box plot of the token count.
plt.rcParams['figure.figsize'] = (10, 10)
ax1 = plt.subplot(211)
plt.title('Token Histogram')
plt.hist(resumes_df.token_count, bins=500, color='gray')
plt.xticks(np.arange(0, 2100, 100), rotation='vertical')
plt.ylabel('Number of Resumes per Token Count')
plt.setp(ax1.get_xticklabels(), fontsize=10)

minTcount = resumes_df.token_count.min()
firstQTcount = resumes_df.token_count.quantile(0.25)
medTcount = resumes_df.token_count.median()
meanTcount = resumes_df.token_count.mean()
thirdQTcount = resumes_df.token_count.quantile(0.75)
maxTcount = resumes_df.token_count.max()

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
plt.boxplot(resumes_df.token_count, 0, 'b+', 0)
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
                '%s_ConditionData_noLongShort_token_count.png' % timeStr)
plt.savefig(combined_token_count_png, dpi=100)
#%%
# Write csv file to disk

timeStr = time.strftime('%Y%m%d-%H%M%S')
print('\nCondition Data: Writing dataframe to .csv file.', 
      '%s_conditioned_resumes_FINAL.csv' % (timeStr) )
cols_to_keep = ['FileNames','Resumes']
resumes_df[cols_to_keep].to_csv(os.path.join(output,
                               '%s_conditioned_resumes_FINAL.csv' % (timeStr)),
                               index=False)

#%%
#Print out the amount of time this script takes to run.
duration = round((time.time() - begin_total_time)/60, 2)
print('\nCondition Data Final: '
      'This script ran in {} minutes.'.format(duration))

