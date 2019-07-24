

import pandas as pd
import numpy as np

import os
os.chdir('/Users/DanLans/Projects/Fraud-detection')

# Read csv files
train_id = pd.read_csv('train_identity.csv')
train_trans = pd.read_csv('train_transaction.csv')
test_id = pd.read_csv('test_identity.csv')
test_trans = pd.read_csv('test_transaction.csv')

# Merge training data sets together and test data sets together
train = train_id.merge(train_trans, on='TransactionID', how = 'left')
test = train_id.merge(test_trans, on='TransactionID', how = 'left')

train_id.shape
train_trans.shape
train.shape

train.columns

train_trans.head()
