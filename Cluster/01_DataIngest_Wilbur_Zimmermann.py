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
import re
import glob
import numpy as np
import pandas as pd
#%%
# Set global path variables
directory = os.path.dirname('__file__')

# Set path to the .csv resume files
path = os.path.join(directory, '..', 'data')

# Set path for the concatenated files
output = os.path.join(directory,'..','script_output')
#%%
# Gather up the path and names of the individual resume files into a list
allResumes = glob.glob(os.path.join(path, '*.csv'))
fileNames = []
for i in range(len(allResumes)):
    fileNames.append(os.path.split(allResumes[i])[1])
# %%
# Read resumes into a Pandas DataFrame and assign a column name 'Resumes'
print('\nIngest Data: Reading in resume files from: ',
      '\n',os.path.realpath(path))

resumes_l = []
file_names_l = []
for file in allResumes:
    temp_row = open(file, 'r', encoding='latin-1').read()
    resumes_l.append(temp_row)
resumes_df = pd.DataFrame(resumes_l, columns=['Resumes'])
resumes_df['FileNames'] = fileNames
# %%
# This checks to make sure that the length of the dataframe is equal in length
# to the allResumes list.
print('\nIngest Data: Checking to see if the resulting dataframe is equal '
      'in length to the number of files in the data directory')

if len(allResumes) == len(resumes_df):
    print('        *** Good news! The number of rows in the dataframe matches '
          'the number of files in the \'data\' directory')
else:
    print('        *** File lenghts do not match')

# %%
# This checks for empty or NaN cells in the dataframe.
print('\nIngest Data: Checking for empty or NaN cells in the dataframe.')

if  resumes_df.Resumes.isnull().values.any():
    print('        *** Dataframe has empty rows')
    print('        *** The dataframe has length: ', len(resumes_df))
else:
    print('        *** Good news! Dataframe does not have empty cells')
    print('        *** The dataframe has length: ', len(resumes_df))
#%%
# Writing dataframe to a .csv file that can be read-in at a later point in
# time.

print('\nIngest Data: Writing dataframe to .csv format in the following '
      'directory: ','\n', os.path.realpath(output))

# Set time string
timeStr = time.strftime('%Y%m%d-%H%M%S')

# Write .csv file to output
print('\nIngest Data: The output filename is:',
      '%s_compiled_resumes_FINAL.csv' % (timeStr))

resumes_df.to_csv(os.path.join(output,
                               '%s_compiled_resumes_FINAL.csv' % (timeStr)),
                  index=False)
#%%
# Script reads in .csv file and compares length and contents to previous file
# to ensure that export happened correctly.

print('\nIngest Data: Ensuring that the dataframe exported to .csv format '
      'correctly')

r = re.compile(r'\d{8}-\d{6}\_compiled_resumes_FINAL\.csv$')

latest_file = max(filter(r.search,os.listdir(output)))
test_df = pd.read_csv(os.path.join(output,latest_file))

if resumes_df.equals(test_df):
    print('          *** Good news! File exported correctly')
else:
    print('          *** File did not export correctly')
#%%
duration = round((time.time() - begin_time)/60, 2)
print('\nIngest completed in {} minutes.'.format(duration))