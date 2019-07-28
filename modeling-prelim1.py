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
- This gives a fairly poor score, but serves as a starting point

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
===============================================
Prep data for the model
===============================================
'''
# Encode categorical features

#    LightGBM can use categorical features as input directly.
#    It doesn’t need to convert to one-hot coding, and is much
#    faster than one-hot coding (about 8x speed-up).

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import lightgbm as lgb

for col in cat_vars:
    if col in train.columns:
        le = preprocessing.LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))

# Try using LightGBM Dataset
X = train.drop(['TransactionDT', 'isFraud'], axis=1)
y = train['isFraud']
train_data = lgb.Dataset(X, label=y)

# Save for quick loading later on
train_data.save_binary('train.bin')
# call with train_data = lgb.Dataset('train.bin')


# Create training and validation data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=3)


'''
===============================================
Set parameters and train LightGBM
===============================================
'''

learning_rate = 0.1
num_leaves = 15
min_data_in_leaf = 2000
feature_fraction = 0.6
num_boost_round = 100
params = {"objective": "binary",
          "boosting_type": "gbdt",
          "learning_rate": learning_rate,
          "num_leaves": num_leaves,
          "max_bin": 63,
          "feature_fraction": feature_fraction,
          "verbosity": 0,
          "drop_rate": 0.1,
          "is_unbalance": True,
          # "scale_pos_weight": negatives / positives,
          "max_drop": 50,
          "min_child_samples": 10,
          "min_child_weight": 150,
          "min_split_gain": 0,
          "subsample": 0.9
          }

num_round = 10

# Train model
bst = lgb.train(params, train_data, num_round)


'''
===============================================
Predict Test data and make submission
===============================================
'''

# predict on test data
ypred = bst.predict(test)
ypred

pred_outcome = [1 if x > 0.5 else 0 for x in ypred]

submit_test = test.drop(['TransactionDT','date'], axis=1)
submit_preds = clf.predict(submit_test)

submission = {'TransactionID': test.TransactionID,'isFraud': pred_outcome}
submission_df = pd.DataFrame(submission)
submission_df.isFraud.value_counts()
submission_df.to_csv('submission2.csv', index = False)


'''
===============================================
Predictions using validation data
===============================================
'''
from sklearn import metrics
x_train = lgb.Dataset(X_train, label=y_train)
booster = lgb.train(params, x_train, num_round)
y_pred = booster.predict(X_valid)

auc = metrics.roc_auc_score(y_valid, y_pred)
auc

# Compute micro-average ROC curve and ROC area
from sklearn.metrics import roc_curve, auc,recall_score,precision_score
fpr, tpr, thresholds = roc_curve(y_valid, y_pred)
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

####################################
# The optimal cut off would be where tpr is high and fpr is low
# tpr - (1-fpr) is zero or near to zero is the optimal cut off point
####################################
i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.loc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = plt.subplots()
plt.plot(roc['tpr'])
plt.plot(roc['1-fpr'], color = 'red')
plt.xlabel('1-False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
ax.set_xticklabels([])


pred_outcome = [1 if x > 0.345308 else 0 for x in y_pred]
print("Model Accuracy:", metrics.accuracy_score(y_valid, pred_outcome))
