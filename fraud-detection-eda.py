

import pandas as pd
import numpy as np

import os
os.chdir('/Users/DanLans/Projects/Fraud-detection')

# Read csv files
train_id = pd.read_csv('train_identity.csv')
train_trans = pd.read_csv('train_transaction.csv')
test_id = pd.read_csv('test_identity.csv')
test_trans = pd.read_csv('test_transaction.csv')

# Rows and Columns of the data
print('train identity:', train_id.shape)
print('train transaction:', train_trans.shape)
print('test identity:', test_id.shape)
print('test transaction:', test_trans.shape)

# Merge training data sets together and test data sets together
train = train_trans.merge(train_id, on='TransactionID', how = 'left')
test = test_trans.merge(test_id, on='TransactionID', how = 'left')

# Shapes of the training and test data
print(f'training data rows: {train.shape[0]}, columns: {train.shape[1]}')
train.head()

print(f'test data rows: {test.shape[0]}, columns: {test.shape[1]}')
test.head()

# Missing Data =====
