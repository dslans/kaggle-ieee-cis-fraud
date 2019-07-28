# Load Python Script to read in competition data
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import os

new_path = '/Users/DanLans/Projects/kaggle-ieee-cis-fraud'
if new_path not in sys.path:
    sys.path.append(new_path)

import vesta_main_functions as vmf


# generate the dictionary of values from loaddata
os.chdir('/Users/DanLans/Projects/Fraud-detection')
fraud_main = vmf.vLoad()


train = vmf.reduce_mem_usage(fraud_main['train_df'])
test = vmf.reduce_mem_usage(fraud_main['test_df'])
cat_vars = fraud_main['cat_vars_list']
num_vars = fraud_main['num_vars_list']

# Save train and test data for quick access
train.to_csv('train_reduced.csv')
test.to_csv('test_reduced.csv')

# Exploration of numerical variables
train[num_vars].columns

# Mising Data table
null_train = pd.DataFrame({'miss_count': train.isnull().sum()})
null_train['miss_pct'] = null_train.miss_count / len(train)

# Identity variables
num_varlist_id = [col for col in train[num_vars] if col.startswith('id_')]
num_id = train[num_varlist_id]

def histplot(data, var, path):
    data = data[var]
    ax = data.hist(bins=20, by=train['isFraud'], figsize=(15,10))
    plt.suptitle(f'Histogram of {var} by isFraud')
    plt.savefig(f'{path}{var}.png')
    plt.clf()
    plt.close()

for col in num_varlist_id:
    histplot(num_id, col, 'plots/identity/num/')


# Transaction variables
num_trans = train[num_vars].drop(num_varlist_id, axis=1)
num_varlist_trans = num_trans.columns

for col in num_varlist_trans:
    histplot(num_trans, col, 'plots/transaction/num/')

v_vars = [col for col in num_trans if col.startswith('V')]
num_trans_v = num_trans[v_vars].replace(0,np.nan)

num_trans[v_vars].head()
num_trans_v.head()
np.average(num_trans_v)
num_trans_v = np.log(num_trans_v)
for col in v_vars:
    histplot(num_trans_v, col, 'plots/transaction/num/log_v/')


"""
=========================================================================
Univariate Tests
=========================================================================

- Check associations with outcome variable

=========================================================================
"""

from sklearn.linear_model import LogisticRegression
predictors = train.drop(['isFraud','TransactionID', 'TransactionDT'], axis=1)
y = train['isFraud']

X.columns[0]
X = predictors.iloc[:,0:1]

X.head()

lr = LogisticRegression().fit(X,y)


import statsmodels.api as sm
logit = sm.Logit(y, X['TransactionAmt'].astype('float32'))
result = logit.fit()
print(result.summary())
result.pvalues


"""
=========================================================================
Transaction amounts
=========================================================================

- Check distributions by isFraud
- Look for any outliers

=========================================================================
"""


## Transaction Amounts

transAmt0 = train[train.isFraud==0].TransactionAmt
transAmt1 = train[train.isFraud==1].TransactionAmt

fig, axes = plt.subplots(1,2)
plot1 = transAmt0.plot.hist(title='TransactionAmt',bins=100, figsize=(10,6), ax=axes[0])
plot2 = np.log(transAmt0).plot.hist(title='log(TransactionAmt)',bins=100, ax=axes[1])
plt.suptitle('Transaction Amount, Fraud = 0')
plt.show()

fig, axes = plt.subplots(1,2)
plot1 = transAmt1.plot.hist(title='TransactionAmt',bins=100, figsize=(10,6), ax=axes[0])
plot2 = np.log(transAmt1).plot.hist(title='log(TransactionAmt)',bins=100, ax=axes[1])
plt.suptitle('Transaction Amount, Fraud = 1')
plt.show()


outliers = [i for i, value in enumerate()]
for i, value in enumerate(transAmt0):
    if value > 10.046:
        plt.annotate(value, (i, value))
plt.show()

def outcome_scatteri(var1, var2, labels=[]):
    plt.figure(figsize=(10,6))
    plot1 = plt.scatter(var1.index, var1, label=labels[0])
    plot1 = plt.scatter(var2.index, var2, label=labels[1])
    plt.legend()
    plt.show()

outcome_scatteri(transAmt0, transAmt1, ['isFraud=0','isFraud=1'])

# Large outlier
print(f'------ Outliers ------\n{transAmt0[transAmt0>30000]}')

'''
It's odd that the 2 large outliers would be for the exact same amount.
I wonder if this is some sort of error, or was there actually 1 really large
transaction that was counted twice?
'''

# Try plotting without outlier
transAmt0_no_outlier = transAmt0.loc[~transAmt0.isin([31936])]
outcome_scatteri(transAmt0_no_outlier, transAmt1, ['isFraud=0','isFraud=1'])


# Plot the log values
outcome_scatteri(np.log(transAmt0), np.log(transAmt1), ['isFraud=0','isFraud=1'])

'''
Using log-values of the TransactionAmt may be more useful in tree methods, as
there is less noise in the data, and outliers will have less of an influence
'''

'''
TransactionAmt Summary
--------------------------------------------
- Amounts for the fraudulent activity appears to vary less (has a tighter
distribution) than the non-fraud group
- Peculiar outlier for 274336 and 274339 indices: both have amount of 31,936

'''


"""
=========================================================================
Outlier Investigation
=========================================================================

- If the transaction for indices 274336 and 274339 are duplicates,
can that tell anything about the id values

=========================================================================
"""

outlierdf = train.iloc[[274336, 274339]]
card_vars = [col for col in train.columns if col.startswith('card')]
id_vars = [col for col in train.columns if col.startswith('id_')]
m_vars = [col for col in train.columns if col.startswith('M')]
c_vars = [col for col in train.columns if col.startswith('C')]
d_vars = [col for col in train.columns if col.startswith('D')]
v_vars = [col for col in train.columns if col.startswith('V')]

# card details
outlierdf[card_vars]

# create date column
import datetime
START_DATE = '2017-12-01'
startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
train['date'] = train['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))


# Are there any other cards with these specific details in train?
train[(train['card1'] == 16075)
    & (train['card2'] == 514)
    & (train['card3'] == 150)
    & (train['card4'] == 'mastercard')
    & (train['card5'] == 102)
    & (train['card6'] == 'credit')
    ][['date'] + id_vars].sort_values(by=['date'])



# Checking if there are any values filled in for id variables
id_null = pd.DataFrame({'miss_count': outlierdf[id_vars].isnull().sum()})
id_null[id_null.miss_count < id_null.shape[1] + 1]

# M variables
outlierdf[m_vars]

# email
outlierdf[['P_emaildomain','R_emaildomain']]

# address
outlierdf[['addr1','addr2']]

outlierdf['ProductCD']

outlierdf[c_vars]

outlierdf[d_vars]

outlierdf[v_vars]

coldiffs =[col for col in outlierdf
    if outlierdf[:1][col].values != outlierdf[1:][col].values and
    pd.notna(outlierdf[:1][col].values) and
    pd.notna(outlierdf[1:][col].values)
]

colsames =[col for col in outlierdf
    if outlierdf[:1][col].values == outlierdf[1:][col].values and
    pd.notna(outlierdf[:1][col].values) and
    pd.notna(outlierdf[1:][col].values)
]

outlierdf[['TransactionAmt'] + coldiffs]

outlierdf[['TransactionAmt'] + coldiffs]

'''
Summary
--------------
These transactions were only 31 seconds apart from one another
The only thing that differs are the TransactionDT and the V variables
Everything else is the same for this transaction
Can we tell if a transaction came from the same card?
Can we filter out duplicate purchases?
'''

outlierdf[colsames]


"""
=========================================================================
Transaction Time
=========================================================================

- Time of Transaction TransactionDT

=========================================================================
"""

## Transaction Date Time (in seconds)
train.TransactionDT.head()
'''
Transaction time is in seconds from a specific start date
'''
seconds = train.TransactionDT
minutes = np.floor(train.TransactionDT / 60)
hours = np.floor(train.TransactionDT / (60*60))
days = np.floor(train.TransactionDT / (60*60*24))

np.floor(max(days))
np.floor(min(days))
# Create a day of week variable
dow = {}
for day in range(7):
    for i in range(day, int(np.floor(max(days)))+1, 7):
        dow.update({int(np.floor(i)): day})

# Check isFraud by days of week
fraud_dow = train[['TransactionID','TransactionDT','isFraud']]
fraud_dow['days'] = np.floor(train.TransactionDT / (60*60*24)).astype('int')

fraud_dow['dow'] = fraud_dow['days'].map(dow)
fraud_dow.dow.value_counts()

t = fraud_dow['isFraud'].groupby(fraud_dow['dow']).value_counts(normalize=True, dropna=False)
t.plot('barh')


# combine train and test TransactionDT

trainDT = {
    'Data': 'train',
    'TransactionDT': train.TransactionDT,
}

testDT = {
    'Data': 'test',
    'TransactionDT': test.TransactionDT,
}

dt_data = pd.DataFrame(trainDT).append(pd.DataFrame(testDT))

dt_data.groupby(dt_data.Data).sum()

# Specific Time of transaction cannot be used to predict test set
# It appears that the test set was gathered at different dates
ax = dt_data.TransactionDT.groupby(dt_data.Data).plot.hist(bins=20)



"""
=========================================================================
Card
=========================================================================

- Time of Transaction TransactionDT

=========================================================================
"""

train.card1.value_counts()


# Group transactionDT by days and check the transaction amounts by day
