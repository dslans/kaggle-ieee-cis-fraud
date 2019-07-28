
# Load Python Script to read in competition data
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

new_path = '/Users/DanLans/Projects/kaggle-ieee-cis-fraud'
if new_path not in sys.path:
    sys.path.append(new_path)

sys.path

from loaddata import main

# generate the dictionary of values from loaddata
fraud_main = main()

train = fraud_main['train_df']
test = fraud_main['test_df']
cat_vars = fraud_main['cat_vars_list']
num_vars = fraud_main['num_vars_list']



# Categorical data exploration -----

# Frequency Plots
def freqplot(data, var, path):
    data = data[var]
    plt.figure(figsize=(15,10))
    data.value_counts(normalize=True, dropna=False).plot('barh')
    plt.xlim(0,1)
    plt.title(var)
    for v in range(len(data.unique())):
        prop = list(data.value_counts(normalize=True, dropna= False))[v]
        percent = '{0:.2%}'.format(prop)
        frq = list(data.value_counts(dropna= False))[v]
        label = percent, frq
        plt.annotate(label, (prop,v))
    plt.savefig(f'{path}{var}.png')
    plt.clf()
    plt.close()

def freqplotByOutcome(data, var, path):
    data = data[var]
    t = data.groupby(train['isFraud']).value_counts(normalize=True, dropna=False)
    ax = t.unstack().plot.barh(by=train['isFraud'], figsize=(15,10))
    plt.xlim(0,1)
    plt.title(f'{var} by isFraud')
    plt.savefig(f'{path}isFraud_{var}.png')
    plt.clf()
    plt.close()


# EDA of identity Variables: all id_ variables and DeviceType and DeviceInfo

cat_varlist_id = [col for col in train[cat_vars] if col.startswith('id_')]
cat_varlist_id = cat_varlist_id + ['DeviceType' ,'DeviceInfo']
cat_id = train[cat_varlist_id]

for col in cat_varlist_id:
    freqplot(cat_id, col, 'plots/identity/cat/')

for col in cat_varlist_id:
    freqplotByOutcome(cat_id, col, 'plots/identity/cat/')

# Transaction Variables
cat_trans = train[cat_vars].drop(cat_varlist_id, axis=1)
cat_varlist_trans = cat_trans.columns

for col in cat_varlist_trans:
    freqplot(cat_trans, col, 'plots/transaction/cat/')

for col in cat_varlist_trans:
    freqplotByOutcome(cat_trans, col, 'plots/transaction/cat/')



# Feature engineering for Identity variables -----

# DeviceInfo: Try to group devices by brand
DevInfo = cat_id['DeviceInfo']

samsung_devs = DevInfo[DevInfo.str.startswith(('SM','SAMSUNG')).fillna(False)].unique()
huawei_devs = DevInfo[DevInfo.str.contains('Huawei|hi6210sft', case=False).fillna(False)].unique()
motorola_devs = DevInfo[DevInfo.str.contains('Moto', case=False).fillna(False)].unique()
windows_devs = DevInfo[DevInfo.str.startswith(('rv','Trident')).fillna(False)].unique()
lg_devs = DevInfo[DevInfo.str.startswith('LG').fillna(False)].unique()
apple_devs = DevInfo[DevInfo.str.startswith(('iOS','MacOS')).fillna(False)].unique()
linux_devs = DevInfo[DevInfo.str.startswith('Linux').fillna(False)].unique()

DevInfo = DevInfo.replace(samsung_devs,'SAMSUNG')
DevInfo = DevInfo.replace(huawei_devs,'HUAWEI')
DevInfo = DevInfo.replace(motorola_devs,'MOTOROLA')
DevInfo = DevInfo.replace(windows_devs,'Windows')
DevInfo = DevInfo.replace(lg_devs,'LG')
DevInfo = DevInfo.replace(apple_devs,'APPLE')
DevInfo = DevInfo.replace(linux_devs,'LINUX')

top_devs = ['Windows','APPLE','SAMSUNG','MOTOROLA','HUAWEI','LG','LINUX']
DevInfo.value_counts()
# For now, just label the remaining lower frequencies as 'other'
DevInfo = DevInfo.replace(DevInfo.loc[~DevInfo.isin(top_devs) & \
 ~DevInfo.isnull()].unique(),'OTHER')

# Replace DeviceInfo variable
cat_id['DeviceInfo'] = DevInfo

cat_id['DeviceInfo'].value_counts(dropna=False)

# id_30: Group together OS systems

osInfo = cat_id['id_30']
windows_os = osInfo[osInfo.str.startswith('Windows').fillna(False)].unique()
ios_os = osInfo[osInfo.str.startswith('iOS').fillna(False)].unique()
mac_os = osInfo[osInfo.str.startswith('Mac').fillna(False)].unique()
android_os = osInfo[osInfo.str.startswith('Android').fillna(False)].unique()

osInfo = osInfo.replace(windows_os,'Windows')
osInfo = osInfo.replace(ios_os,'iOS')
osInfo = osInfo.replace(mac_os,'Mac')
osInfo = osInfo.replace(android_os,'Android')

cat_id['id_30'] = osInfo
cat_id['id_30'].value_counts(dropna=False)

# combine browser info

browserInfo = cat_id['id_31']

safari_browser = browserInfo[browserInfo.str.contains('safari', case=False).fillna(False)].unique()
chrome_browser = browserInfo[browserInfo.str.contains('chrome', case=False).fillna(False)].unique()
firefox_browser = browserInfo[browserInfo.str.contains('firefox', case=False).fillna(False)].unique()
ie_browser = browserInfo[browserInfo.str.contains('ie', case=False).fillna(False)].unique()
edge_browser = browserInfo[browserInfo.str.contains('edge', case=False).fillna(False)].unique()
samsung_browser = browserInfo[browserInfo.str.contains('samsung', case=False).fillna(False)].unique()


browserInfo = browserInfo.replace(safari_browser,'safari')
browserInfo = browserInfo.replace(chrome_browser,'chrome')
browserInfo = browserInfo.replace(firefox_browser,'firefox')
browserInfo = browserInfo.replace(ie_browser,'ie')
browserInfo = browserInfo.replace(edge_browser,'edge')
browserInfo = browserInfo.replace(samsung_browser,'samsung')

cat_id['id_31'] = browserInfo
cat_id['id_31'].value_counts(dropna=False)


## Feature for email data
cat_trans.P_emaildomain.value_counts()
