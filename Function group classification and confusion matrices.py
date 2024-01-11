import os, sqlite3
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score as accuracy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from string import Template

##from time import perf_counter
import matplotlib.pyplot as plt
import seaborn as sns

conn = sqlite3.connect('pflacco.sqlite')
conn.row_factory = sqlite3.Row
classifiers = {'knn': KNeighborsClassifier,
               'dtree': DecisionTreeClassifier,
               'rfc': RandomForestClassifier}
fgrouping = {1:1, 2:1, 3:1, 4:1, 5:1,
             6:2, 7:2, 8:2, 9:2,
             10:3, 11:3, 12:3, 13:3, 14:3,
             15:4, 16:4, 17:4, 18:4, 19:4,
             20:5, 21:5, 22:5, 23:5, 24:5}
query = """
select function, sampler, test_id, pflacco_feature, pflacco_value
from features
where
pflacco_feature != "ela_meta.quad_simple.cond" and
pflacco_feature not like "%runtime" and
pflacco_feature not like "limo%" and
test_id != 1364
"""
df=pd.read_sql(query, conn)
df=df.dropna(axis=0)
cms={}
run_hist=[]
for sampler, sampler_df in df.groupby('sampler'):
    y = sampler_df.pivot_table(values='function',index='test_id', dropna=True)
    X = sampler_df.pivot_table(values='pflacco_value',index='test_id',
                               columns='pflacco_feature', dropna=True)
    #print(f'{sampler}:\n {X[X.isna().any(axis=1)]}')
    yfg = y['function'].map(fgrouping)
    skf = StratifiedKFold()
    #skf = RepeatedStratifiedKFold()
    for i, (train_idx, test_idx) in enumerate(skf.split(X,y)):
        for name, classifier in classifiers.items():
            cls = classifier().fit(X.iloc[train_idx,:], yfg.iloc[train_idx])
            pred = cls.predict(X.iloc[test_idx,:])
            cm = confusion_matrix(yfg.iloc[test_idx],pred)
            acc = accuracy(yfg.iloc[test_idx],pred)
            if (sampler, name) in cms:
                cms[(sampler, name)] += cm
            else:
                cms[(sampler, name)] = cm
            print(f'Sampler: {sampler}, Repetition: {i}, Classifier: {name}, Accuracy: {acc}')
            run_hist.append({'Sampler': sampler, 'Repetition': i, 'Classifier': name, 'Accuracy': acc})

pd.DataFrame(run_hist).to_csv('Confusion matrices run history.csv',
                              index=False,
                              float_format='%.5f')

fig, ax = plt.subplots(3,3, figsize=(15,15))
fig.tight_layout()
for i, sampler in enumerate(df.sampler.unique()):
    for j, classifier in enumerate(classifiers.keys()):
        cm = cms[(sampler, classifier)]
        sns.heatmap(cm/cm.sum(),
                    ax=ax[i,j], vmin=0, vmax=0.2,
                    annot=True, cbar=False)
        if i == 2:
            ax[i,j].set_xlabel(classifier)
        if j == 0:
            ax[i,j].set_ylabel(sampler)
fig.savefig('Confusion matrices by classifier and sampling strategy (ic, 1000).png', dpi=400)
fig.show()

            
    
