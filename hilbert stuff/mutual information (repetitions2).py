import sqlite3
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

with sqlite3.connect('pflacco.sqlite') as conn:
	df = pd.read_sql('select * from repetitions2 \
                            where \
                            value is not null and \
                            feature not like "%runtime"',
                         conn)

df.loc[df['feature']=='ic_eps.max','feature']='ic.eps.max'
df['f_family']=df['feature'].map(lambda s: s.split('.')[0])

fgrouping={1:1, 2:1, 3:1, 4:1, 5:1,
           6:2, 7:2, 8:2, 9:2,
           10:3, 11:3, 12:3, 13:3, 14:3,
           15:4, 16:4, 17:4, 18:4, 19:4,
           20:5, 21:5, 22:5, 23:5, 24:5}

pivots={}
ydf={}
for f, fdf in df.groupby(['sampler','dimensions','sample_n','f_family']):
	pivots[f]= fdf.pivot_table(values='value',index='test_id',columns='feature')
	ydf[f] = fdf.pivot_table(values='function', index='test_id')
	ydf[f] = ydf[f].iloc[:,0].map(fgrouping)
        #assert ydf[f].shape[0]==pivots[f].shape[0]

mi={}
for k, X in pivots.items():
    try:
        mi[k]=mutual_info_classif(X,ydf[k].squeeze())
    except ValueError as e:
        print(f'Ecountered ValueError in {k}: {e}')

results=[]
for k in mi.keys():
	sampler, dimension, sample_n, ffamily = k
	if ffamily in ['ic']:
		print(f'\nMutual information of {ffamily}, using {sampler}, in {dimension} dimensions:')
		for f, v in zip(pivots[k].columns, mi[k]):
			print(f'{f} : {v}')
			res={'sampler':sampler, 'dimension':dimension,
			     'sample_n':sample_n, 'feature':f, 'value':v}
			results.append(res)
pd.DataFrame(results).to_csv('mutual info.csv',index=False)
