'''
Import data from ieee-cis fraud detection competition
Returns:
1. train_df = training data
2. test_df = test data
3. cat_vars_list = list of categorical columns
4. num_vars_list = list of numerical columns

'''
def vLoad():
    import pandas as pd
    # Read csv files
    train_id = pd.read_csv('train_identity.csv')
    train_trans = pd.read_csv('train_transaction.csv')
    test_id = pd.read_csv('test_identity.csv')
    test_trans = pd.read_csv('test_transaction.csv')

    # Merge training data sets together and test data sets together
    train = train_trans.merge(train_id, on='TransactionID', how = 'left')
    test = test_trans.merge(test_id, on='TransactionID', how = 'left')

    # descriptions
    print(f'training data rows: {train.shape[0]}, columns: {train.shape[1]}')
    print(f'test data rows: {test.shape[0]}, columns: {test.shape[1]}')

    print('Outcome Class','----------------',
    train.isFraud.value_counts(),
    train.isFraud.value_counts(normalize=True), sep='\n')

    print('Response rate in training data: \
    {0:.2f}%'.format(train.isFraud.value_counts(normalize=True)[1]*100))

    # missing data
    null_train = pd.DataFrame({'miss_count': train.isnull().sum()})
    null_train['miss_pct'] = null_train.miss_count / len(train)
    null_train_vars = null_train[null_train.miss_count > 0]

    print(f'Total Variables: {train.shape[1]}')
    print(f'Variables with missing data: {len(null_train_vars.index)}')


    # Variables with Homogenous Values
    homo_train = pd.DataFrame({'name': train.columns,\
    'unique_value_pct': \
    [max(train[col].value_counts(normalize=True)) for col in train]})

    homo_test = pd.DataFrame({'name': test.columns,\
    'unique_value_pct': \
    [max(test[col].value_counts(normalize=True)) for col in test]})

    homo_train100 = homo_train.loc[homo_train.unique_value_pct==1]
    homo_test100 = homo_test.loc[homo_test.unique_value_pct==1]

    print(f'Variables in Train with Single Value: {len(homo_train100)}, \
    {list(homo_train100.name)}')
    print(f'Variables in Test with Single Value: {len(homo_test100)}, \
    {list(homo_test100.name)}')

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
    print(f'Categorical Variables: {len(cat_vars)}')

    # Numerical Variables
    train_num = train.drop(cat_vars, axis=1)
    num_vars = train_num.columns
    print(f'Numerical Variables: {len(num_vars)}')

    # return values

    export_dict = {
        'train_df': train,
        'test_df': test,
        'null_train_df': null_train,
        'cat_vars_list': cat_vars,
        'num_vars_list': num_vars,
        'null_vars_list': null_train_vars
    }
    print('**************************************')
    print('Load data complete')
    return(export_dict)

# fraud_main = main()
#
# fraud_main.keys()


## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    import numpy as np
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
