
'''
Functions useful for managing categorical data
kaggle Vesta fraud detection
'''

# Variables starting with 'id':
# Use train for data input and cat_vars for vars
def create_id_df(data, vars):
    cat_varlist_id = [col for col in data[vars] if col.startswith('id_')]
    cat_varlist_id = cat_varlist_id + ['DeviceType' ,'DeviceInfo']
    cat_id = data[cat_varlist_id]
    return(cat_id)

def create_trans_df(data, vars):
    cat_varlist_id = [col for col in data[vars] if col.startswith('id_')]
    cat_trans = data[vars].drop(cat_varlist_id, axis=1)
    return(cat_trans)


# Frequency Plots: input categorical data
# cat_varlist_id = cat_id.columns
# for col in cat_varlist_id:
#     freqplot(cat_id, col, 'plots/identity/cat/')
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

# Frequency ploy by isFraud
def freqplotByOutcome(data, var, path):
    data = data[var]
    t = data.groupby(train['isFraud']).value_counts(normalize=True, dropna=False)
    ax = t.unstack().plot.barh(by=train['isFraud'], figsize=(15,10))
    plt.xlim(0,1)
    plt.title(f'{var} by isFraud')
    plt.savefig(f'{path}isFraud_{var}.png')
    plt.clf()
    plt.close()
