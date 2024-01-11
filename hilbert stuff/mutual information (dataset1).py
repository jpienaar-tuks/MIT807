import sqlite3
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

with sqlite3.connect('pflacco.sqlite') as conn:
	df = pd.read_sql('select * from dataset1 \
                            where \
                            pflacco_value is not null and \
                            pflacco_feature not like "limo%" and \
                            pflacco_feature not like "%runtime" and \
                            pflacco_feature not like "ela_meta.quad_simple.cond"',
                         conn)

df.loc[df['pflacco_feature']=='ic_eps.max','pflacco_feature']='ic.eps.max'
df['f_family']=df['pflacco_feature'].map(lambda s: s.split('.')[0])

pivots={}
ydf={}
for f, fdf in df.groupby(['sampler','dimensions','sample_n','f_family']):
	pivots[f]= fdf.pivot_table(values='pflacco_value',index='test_id',columns='pflacco_feature')
	ydf[f] = fdf.pivot_table(values='function', index='test_id')
        #assert ydf[f].shape[0]==pivots[f].shape[0]

mi={}
for k, X in pivots.items():
    try:
        mi[k]=mutual_info_classif(X,ydf[k].squeeze())
    except ValueError as e:
        print(f'Ecountered ValueError in {k}: {e}')

for k in mi.keys():
	sampler, dimension, sample_n, ffamily = k
	if sampler in ['Hilbert curve','Latin Hypercube'] and sample_n==1000 and ffamily in ['ela_meta','ic']:
		print(f'\nMutual information of {ffamily}, using {sampler}, in {dimension} dimensions:')
		for f, v in zip(pivots[k].columns, mi[k]):
			print(f'{f} : {v}')
