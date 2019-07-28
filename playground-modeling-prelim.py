"""
==================================================================
Preliminary Modeling Steps
==================================================================

The purposes of this script is to set up a basic modeling
framework for the ieee-fraud detection problem provided
by Vesta at kaggle.com.

- Read in the data and see what you can get for prediction
with very little curating of any of the variables
- The data set is very large, so need to come up with an
efficient process to run on my machine

"""


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
import ieeeFeatures as ft

# generate the dictionary of values from loaddata
os.chdir('/Users/DanLans/Projects/Fraud-detection')
fraud_main = vmf.vLoad()

train = fraud_main['train_df']
test = fraud_main['test_df']
cat_vars = fraud_main['cat_vars_list']
num_vars = fraud_main['num_vars_list']
null_train = fraud_main['null_train_df']
null_vars = fraud_main['null_vars_list']

'''
============================================================
Evaluate Missing Data
-------------------------------------
- In this section, I will add any feature engineering pieces to the data
that I had previously created
-
============================================================
'''

high_nulls = null_train[null_train['miss_pct']>0.9]


null_fraud0 = pd.DataFrame({'miss_count': train[train['isFraud']==0].isnull().sum()})
null_fraud1 = pd.DataFrame({'miss_count': train[train['isFraud']==1].isnull().sum()})
null_fraud0['miss_pct'] = null_fraud0.miss_count / len(train[train['isFraud']==0])
null_fraud1['miss_pct'] = null_fraud1.miss_count / len(train[train['isFraud']==1])

high_nulls0 = null_fraud0[null_fraud0['miss_pct']>0.9].index

train_nulls = null_fraud0.loc[high_nulls_train].sort_values(by=null_fraud0.index)
null_fraud1.loc[high_nulls_train].join(null_fraud0.loc[high_nulls_train], lsuffix='_fraud1', rsuffix='_fraud0')

null_compare = null_fraud1.loc[high_nulls0].join(null_fraud0.loc[high_nulls0], lsuffix='_fraud1', rsuffix='_fraud0')
null_compare[['miss_pct_fraud0','miss_pct_fraud1']]

null_fraud1[null_fraud1['miss_pct']>0.9].sort_values(by='miss_pct', ascending=False)

null_fraud1.loc['D13']

'''
============================================================
Create Features for Input into Model
-------------------------------------
- In this section, I will add any feature engineering pieces to the data
that I had previously created
-
============================================================
'''
# Device, OS, and Browser
for d in [train, test]:
    DevInfo = ft.ft_deviceInfo(d)
    os = ft.ft_os(d)
    browser = ft.ft_browser(d)
    date = ft.ft_createDate(d)

    d['DeviceInfo'] = DevInfo
    d['id_30'] = os
    d['id_31'] = browser
    d['date'] = date

# Transaction Amount
for d in [train, test]:
    d['TransactionAmt'] = np.log(d.TransactionAmt)

# date format
for d in [train, test]:
    d['date'] = np.log(d.TransactionAmt)

'''
===============================================
Prep data for the model
===============================================
'''
# Adjust for imbalance in response variable

# Encode categorical features
from sklearn import preprocessing
for col in cat_vars:
    if col in train.columns:
        le = preprocessing.LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))

# Set input and output variables for the model
X_train = train.sort_values('TransactionDT').drop(['isFraud','TransactionDT','date'], axis=1)
y_train = train.sort_values('TransactionDT')['isFraud']

X_test = test.sort_values('TransactionDT').drop(['TransactionDT','date'], axis=1)


# Reduce the memory usage
X_train = vmf.reduce_mem_usage(X_train)
X_test = vmf.reduce_mem_usage(X_test)


'''
mini-model

'''

fraud = train[train['isFraud']==1]
nofraud = train[train['isFraud']==0]
fraud.shape
nofraud.shape

sub_nofraud = nofraud.sample(fraud.shape[0], random_state=3)

tr_small = fraud.append(sub_nofraud)

X_small = tr_small.sort_values('TransactionDT').drop(['isFraud','TransactionDT','date'], axis=1)
y_small = tr_small.sort_values('TransactionDT')['isFraud']

from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X_small, y_small, test_size=0.2, random_state=32)

clf = xgb.XGBClassifier(n_estimators=500,
                        n_jobs=4,
                        max_depth=9,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        missing=None)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Model Accuracy:", metrics.accuracy_score(y_test, y_pred))

auc = metrics.roc_auc_score(y_test, y_pred)
auc


############################################ ROC Curve



# Compute micro-average ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
roc_auc
#xgb.plot_importance(gbm)
#plt.show()
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
        lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()
##################################################









from xgboost import plot_importance
importances = clf.feature_importances_

indices = np.argsort(importances)[::-1]

topvars = [X_train.columns[indices[x]] for x in range(X_train.shape[1]) if importances[indices[x]] > 0]
topimp = [indices[x] for x in range(X_train.shape[1]) if importances[indices[x]] > 0]
nvars = 50
topvarsN = importances[topimp][0:nvars]

# Plot the feature importances of the ensemble
plt.figure(figsize = (10,10)) # increase size to fit all variables
plt.title("Top 50 Features")
plt.barh(range(len(topvarsN)), topvarsN, align="center") # horizontal bar chart
plt.yticks(range(len(topvarsN)), topvars[0:len(topvarsN)])
plt.ylim([-1, len(topvarsN)]) # fit for the y axis
plt.gca().invert_yaxis() #correct the order from highest importance to lowest
plt.show()


submit_test = test.drop(['TransactionDT','date'], axis=1)
submit_preds = clf.predict(submit_test)

submission = {'TransactionID': submit_test.TransactionID,'isFraud': submit_preds}
submission_df = pd.DataFrame(submission)
submission_df.isFraud.value_counts()
submission_df.to_csv('submission.csv', index = False)












#-----------------------------------


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from operator import itemgetter
import xgboost as xgb
import random
import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from numpy import genfromtxt
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc,recall_score,precision_score
import datetime as dt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory




def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance

def get_features(train, test):
    trainval = list(train.columns.values)
    output = trainval
    return sorted(output)



def run_single(train, test, features, target, random_state=0):
    eta = 0.1
    max_depth= 6
    subsample = 1
    colsample_bytree = 1
    min_chil_weight=1
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "min_chil_weight":min_chil_weight,
        "seed": random_state,
        #"num_class" : 22,
    }
    num_boost_round = 500
    early_stopping_rounds = 20
    test_size = 0.1



    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    print('Length train:', len(X_train.index))
    print('Length valid:', len(X_valid.index))
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train, missing=-99)
    dvalid = xgb.DMatrix(X_valid[features], y_valid, missing =-99)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration+1)

    #area under the precision-recall curve
    score = average_precision_score(X_valid[target].values, check)
    print('area under the precision-recall curve: {:.6f}'.format(score))


    check2=check.round()
    score = precision_score(X_valid[target].values, check2)
    print('precision score: {:.6f}'.format(score))

    score = recall_score(X_valid[target].values, check2)
    print('recall score: {:.6f}'.format(score))

    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print("Predict test set... ")
    test_prediction = gbm.predict(xgb.DMatrix(test[features],missing = -99), ntree_limit=gbm.best_iteration+1)
    score = average_precision_score(test[target].values, test_prediction)

    print('area under the precision-recall curve test set: {:.6f}'.format(score))

     ############################################ ROC Curve



    # Compute micro-average ROC curve and ROC area
    fpr, tpr, _ = roc_curve(X_valid[target].values, check)
    roc_auc = auc(fpr, tpr)
    #xgb.plot_importance(gbm)
    #plt.show()
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
    ##################################################


    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction, imp, gbm.best_iteration+1


# Any results you write to the current directory are saved as output.
start_time = dt.datetime.now()
print("Start time: ",start_time)

# data=pd.read_csv('../input/creditcard.csv')

x_train, x_test = train_test_split(train, test_size=.1, random_state=random.seed(2016))


features = list(train.columns.values)
features.remove('Class')
print(features)


print("Building model.. ",dt.datetime.now()-start_time)
preds, imp, num_boost_rounds = run_single(X_train, X_test, features, 'Class',42)

print(dt.datetime.now()-start_time)
