import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score as accuracy

# Permute and predict

dirpath = os.path.dirname(__file__)
conn =sqlite3.connect(os.path.realpath(os.path.join(dirpath,'..','..','pflacco.sqlite')))
df = pd.read_sql('select * from repetitions2 \
                    where \
                    value is not null and \
                    feature like "ic.%" and \
                    feature not like "%time"',
                 conn)

fgrouping={1:1, 2:1, 3:1, 4:1, 5:1,
           6:2, 7:2, 8:2, 9:2,
           10:3, 11:3, 12:3, 13:3, 14:3,
           15:4, 16:4, 17:4, 18:4, 19:4,
           20:5, 21:5, 22:5, 23:5, 24:5}

#df['f2']=df['function'].map(fgrouping)
samplers = df['sampler'].unique()
features = df['feature'].unique()

results=[]
for sampler, sdf in df.groupby('sampler'):
    pivot_df=sdf.pivot_table(index=['test_id','dimensions','function'],
                             columns='feature',
                             values='value')
    pivot_df.reset_index([1,2], inplace=True)
##    funcs=pivot_df['function']
##    f2 = funcs.map(fgrouping)
    X = pivot_df.loc[:,features]
    yf = pivot_df['function']
    yfg = yf.map(fgrouping)
    rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=20)
    for fold, (train_idx, test_idx) in enumerate(rskf.split(X,yf)):
        print(f'Sampler: {sampler}, Fold: {fold}')
        X_train, y_train = X.iloc[train_idx.flatten(),:], yfg.iloc[train_idx.flatten()]
        X_test, y_test = X.iloc[test_idx.flatten(),:], yfg.iloc[test_idx.flatten()]
##        X_train, y_train = X.iloc[train_idx.flatten(),:], yf.iloc[train_idx.flatten()]
##        X_test, y_test = X.iloc[test_idx.flatten(),:], yf.iloc[test_idx.flatten()]
        rfcls = RandomForestClassifier()
        rfcls.fit(X_train,y_train)
        result={}
        result['sampler'] = sampler
        result['fold'] = fold
        result['Base'] = accuracy(y_test, rfcls.predict(X_test))
        for feature in features:
            X_test_perm = X_test.copy()
            X_test_perm[feature] = np.random.permutation(X_test_perm[feature])
            result[feature] = accuracy(y_test, rfcls.predict(X_test_perm))
        results.append(result)
rdf = pd.DataFrame(results)
rdf.to_csv('Feature impact on accuracy (fgrouping).csv',index=False)
    
