import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from scipy.stats import boxcox

DATA_FOLDER = 'data/'

print('Loading dataset...')
train_df = pd.read_csv(os.path.join(DATA_FOLDER,'train.csv'))
test_df = pd.read_csv(os.path.join(DATA_FOLDER,'test.csv'))
print('Done.')

non_binary = ['X0','X1','X2','X3','X4','X5','X6','X8']

'''
print('Converting non_binary features usint target-based encoding (test and train sets)...')
for col in non_binary:
    group_by_df = train_df.groupby(col, as_index=False)['y'].mean()
    group_by_df.columns = [col,col+'_trans']
    #   Train
    train_df = train_df.merge(right=group_by_df, on=[col], left_index=True)
    train_df.drop(col, axis=1, inplace=True)

    #   Test
    test_df = test_df.merge(right=group_by_df, how='left', on=[col], left_index=True)
    test_df[col + '_trans'].fillna(np.mean(train_df['y']))
    test_df.drop(col, axis=1, inplace=True)
print('Done.')
'''

print('Converting non_binary features one-hot-encoding (test and train sets)...')
train_df['dataset'] = 'train'
test_df['dataset'] = 'test'
total_df = pd.concat([train_df,test_df])
encoder = preprocessing.OneHotEncoder()
for col in non_binary:
    total_df = pd.concat([total_df.drop(col,axis=1), pd.get_dummies(total_df[col], prefix = col)])

train_df = total_df[total_df['dataset']=='train'].drop('dataset',axis=1)
test_df = total_df[total_df['dataset']=='test'].drop('dataset',axis=1)
print('Done.')

print('Applying boxcox and minmax transformation to y...')
#train_df['y_trans'] = boxcox(train_df['y'])[0]
mmscaler = preprocessing.MinMaxScaler()
train_df['y_trans'] = train_df['y']#mmscaler.fit_transform(train_df['y'])
print('!!! AFTER PREDICTION REMEMBER TO REVERSE THESE TRANSFORMATIONS !!!')
print('Done.')

print('Saving dataframes...')
test_df.to_csv(os.path.join(DATA_FOLDER,'test_transformed.csv'), index=False)
train_df.to_csv(os.path.join(DATA_FOLDER,'train_transformed.csv'), index=False)

print('\t Number of columns...')
print(len(train_df.columns))
print('\tDone.')