

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


# Outcome Class -----
print('Outcome Class','----------------',
train.isFraud.value_counts(),
train.isFraud.value_counts(normalize=True), sep='\n')

print('Response rate in training data: \
{0:.2f}%'.format(train.isFraud.value_counts(normalize=True)[1]*100))

# Missing Data -----
null_train = pd.DataFrame({'miss_count': train.isnull().sum()})
null_train['miss_pct'] = null_train.miss_count / len(train)
null_train_vars = null_train[null_train.miss_count > 0]
print(null_train_vars.sort_values('miss_count', ascending=False).head())
print(f'Total Variables: {train.shape[1]}')
print(f'Variables with missing data: {len(null_train_vars.index)}')

# Variables with Homogenous Values -----
homo_train = pd.DataFrame({'name': train.columns,\
'unique_value_pct': \
[max(train[col].value_counts(normalize=True)) for col in train]})

homo_test = pd.DataFrame({'name': test.columns,\
'unique_value_pct': \
[max(test[col].value_counts(normalize=True)) for col in test]})

homo_train100 = homo_train.loc[homo_train.unique_value_pct==1]
homo_test100 = homo_test.loc[homo_test.unique_value_pct==1]

print(f'Vars in Train with Single Value: {len(homo_train100)}, \
{list(homo_train100.name)}')
print(f'Vars in Test with Single Value: {len(homo_test100)}, \
{list(homo_test100.name)}')

# Variable types -----

# Categorical data as stated by Vesta
cat_vars = ['ProductCD',
'card1','card2','card3','card4','card5','card6',
'addr1','addr2','P_emaildomain','R_emaildomain',
'M1','M2','M3','M4','M5','M6','M7','M8','M9',
'DeviceType','DeviceInfo',
'id_12','id_13','id_14','id_15','id_16','id_17','id_18','id_19','id_20','id_21',
'id_22','id_23','id_24','id_25','id_26','id_27','id_28','id_29','id_30','id_31',
'id_32','id_33','id_34','id_35','id_36','id_37','id_38'
]

train_cat = train[cat_vars]
train_cat.dtypes.value_counts()

train_num = train.drop(cat_vars, axis=1)
train_num.dtypes.value_counts()


# Convert all categorical variables to objects
train[cat_vars] = train[cat_vars].astype(object)
train_cat = train[cat_vars]
train_cat.dtypes.value_counts()

# unique value counts for categorical variables
def uniqueValues(df):
    return(df.apply(lambda x: len(x.unique())))

value_dict = {
'var_type': train_cat.dtypes,
'missing_values': train_cat.isnull().sum(),
'unique_values': uniqueValues(train_cat)
}

value_types = pd.DataFrame(value_dict)
value_types['non_miss_rows'] = len(train) - value_types.missing_values
value_types['unique_values_pct'] = value_types.unique_values /  \
    value_types.non_miss_rows
value_types

train_cat['id_12'].value_counts().plot('barh')

plt.figure(figsize=(10,10))
train_cat['id_13'].value_counts().plot('barh')

train_cat['id_14'].value_counts().plot('barh')

train_cat['id_15'].value_counts().plot('barh')

train_cat['id_16'].value_counts().plot('barh')

cat_plots = []
for col in train_cat.columns[[0,4,6]]:
    cat_plots.append(train_cat[col].value_counts().plot('barh'))


train_cat[col].value_counts().plot('barh')



plt.show(cat_plots[0])
plt.show()
cat_plots[0].

fig.savefig('plots/cat_vars/temp.png')
fig.add_subplot(cat_plots[0])


train_cat['ProductCD'].value_counts().plot('barh')
train_cat['id_12'].value_counts().plot('barh')
plt.savefig('plots/cat_vars/temp.png')

# unique values of card2 per card type
len(train_cat[train_cat['card4']=='visa']['card5'].unique())
len(train_cat[train_cat['card4']=='american express']['card5'].unique())
len(train_cat[train_cat['card4']=='discover']['card5'].unique())
len(train_cat[train_cat['card4']=='mastercard']['card5'].unique())
