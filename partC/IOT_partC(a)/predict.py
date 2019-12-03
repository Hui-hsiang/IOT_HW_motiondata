import os
import glob
import pandas as pd 
import numpy as np
import plotly.graph_objects as go
import sys


# Gives the Directory path
DataDir = os.path.dirname(os.path.abspath('IOT.py')) + '/' + 'Data_Acc'
# print(DataDir)

Window_size = 50

## Riding csv    
Riding_df = pd.read_csv(DataDir + '/' + 'Riding1_Acc.csv',usecols=['Timestamp','X','Y','Z'])
# Riding_df = Riding_df.append(pd.read_csv(DataDir + '/' + 'Riding2_Acc.csv',usecols=['Timestamp','X','Y','Z']),ignore_index = True)
# print('Riding: ',Riding_df)

# Sliding Window std
Sliding_riding = Riding_df.rolling(Window_size).std()
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(Sliding_riding)


## Study csv
Study_df = pd.read_csv(DataDir + '/' + 'Study1_Acc.csv',usecols=['Timestamp','X','Y','Z'])
# Study_df = Study_df.append(pd.read_csv(DataDir + '/' + 'Study2_Acc.csv',usecols=['Timestamp','X','Y','Z']),ignore_index = True)
# print('Study: ',Study_df)

# Sliding Window std
Sliding_study = Study_df.rolling( Window_size).std()
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(Sliding_study)


## Walk csv
Walk_df = pd.read_csv(DataDir + '/' + 'Walk_Acc.csv',usecols=['Timestamp','X','Y','Z'])
# print('Walk: ', Walk_df)

# Sliding Window std
Sliding_walk = Walk_df.rolling(Window_size).std()
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(Sliding_walk)

# Mean of std
# print('Walk\n',Sliding_walk.mean(axis = 0, skipna = True))

# print('Study\n',Sliding_study.mean(axis = 0, skipna = True))

# print('Riding\n', Sliding_riding.mean(axis = 0, skipna = True))

# use y to verify, 因為差距最多
# Std_mean = [0.063763,0.050141,3.253057]
# mean of std of y
Std_mean = [Sliding_walk.mean(axis = 0, skipna = True)['Y'],Sliding_study.mean(axis = 0, skipna = True)['Y'],Sliding_riding.mean(axis = 0, skipna = True)['Y']]
Class_name = ['walk','study','ride']

# input argument
files = sys.argv[1]

# read input files
input_df = pd.read_csv(DataDir + '/' + files,usecols=['X','Y','Z'])

# Compute std and mean od sliding window
Sliding_input = input_df.rolling(Window_size).std()
input_std_mean = Sliding_input.mean(axis = 0, skipna = True)['Y']
# print(input_std_mean)

# the least difference between the Std_Mean is the categories of it
diff = 999999999
same = 0
for i in range(len(Std_mean)):
    if abs(input_std_mean - Std_mean[i]) < diff:
        diff = abs(input_std_mean - Std_mean[i])
        same = i
        # print(diff, i)

# print(same)
print(Class_name[same])

# label the input file
if(files[0] == 'R'):
    label = 2
elif(files[0] == 'W'):
    label = 0
else:
    label = 1

wrong = 0
# All Rows Accurancy
for Y in input_df['Y'].values:
    diff = 999999999
    same = 0
    for i in range(len(Std_mean)):
        if abs(Y - Std_mean[i]) < diff:
            diff = abs(Y - Std_mean[i])
            same = i
    if(same != label):
        wrong += 1
print('Error rate: ', wrong/len(input_df.values))