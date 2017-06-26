import pandas as pd
import numpy as np
import os
import gc
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn import preprocessing
from scipy.stats import boxcox


DATA_FOLDER = 'data/'

print('Loading dataset...')
train_df = pd.read_csv(os.path.join(DATA_FOLDER,'train.csv'))
test_df = pd.read_csv(os.path.join(DATA_FOLDER,'test.csv'))
print('Done.')

print('Analysis...')

print('\t head...')
print(train_df.head(10))
print('\tDone.')

print('\t describe...')
print(train_df.describe())
print('\tDone.')

print('\t Unique values for each column...')
for col in train_df.columns:
    print('Col {} unique values:'.format(col))
    print(np.unique(train_df[col]))
print('\tDone.')
print('All features are categorical! Some are non binary:')

non_binary = ['X0','X1','X2','X3','X4','X5','X6','X8']

print('Converting non_binary features usint target-based encoding...')
for col in non_binary:
    group_by_df = train_df.groupby(col, as_index=False)['y'].mean()
    group_by_df.columns = [col,col+'_trans']
    train_df = train_df.merge(right=group_by_df, on=[col], left_index=True)
    train_df.drop(col, axis=1, inplace=True)
#helmert_encoder = ce.HelmertEncoder(cols=non_binary, return_df=True)
#helmert_encoder.fit_transform(train_df)

print('\tDone.')

print('Feature importance analysis with RandomForest...')

print('\tFitting...')
rf = RandomForestRegressor(n_estimators=100, verbose=5)
X = train_df.drop(['y','ID'], axis=1).as_matrix()
y = train_df['y'].as_matrix()
rf.fit(X, y)
print('\tDone.')

print('\tPlotting (http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)...')
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
topfeatures = 40
print('Getting {} most important features...'.format(topfeatures))
most_important_features = train_df.columns[indices[0:topfeatures]]
print(most_important_features)
print('Done.')

print('Histogram of target variable...')
train_df['y'].hist()
plt.show()
print('Apply box cox and minmax scaling?')
print('Done')