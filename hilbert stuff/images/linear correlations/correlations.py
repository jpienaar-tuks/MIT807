import sqlite3
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
from scipy.stats import linregress

conn = sqlite3.connect(os.path.realpath(os.path.join(os.path.dirname(__file__),'..','..','pflacco.sqlite')))
df = pd.read_sql('select * from repetitions2 where feature like "ic%" and feature not like "%time" and sample_n=1000', conn)
pivot_df=df.pivot_table(index=['repetition','function','feature'], columns='sampler', values='value')
pivot_df.reset_index([1,2],inplace=True)
pivot_df['f2'] = pivot_df['function'].map(str)

features={'ic.h_max':r'$H_{max}$',
		  'ic.eps_s':r'$\epsilon_s$',
		  'ic.eps_max':r'$\epsilon_{max}$',
		  'ic.eps_ration':r'$\epsilon_{0.5}$',
		  'ic.m0':r'$M_0$'}

results=[]
samplers = [i for i in pivot_df.columns if 'Latin' in i]
for i, feature in enumerate(pivot_df['feature'].unique()):
	for j, (s1, s2) in enumerate(combinations(samplers,2)):
		fig, ax = plt.subplots()
		idx = pivot_df['feature']==feature
		#sns.regplot(pivot_df.loc[idx,:], x=s1, y=s2, scatter=False, units='f2', ax=ax)
		sns.scatterplot(pivot_df.loc[idx,:], x=s1, y=s2, hue='f2', ax=ax)
		if feature == 'ic.eps_max':
			ax.set_xscale('log')
			ax.set_yscale('log')
		m, c, r, p, _ =linregress(pivot_df.loc[idx,[s1,s2]])
		ax.text(0.8, 0.1, f'm: {m:.3f}\nc: {c:.3f}\n$r^2$: {r**2:.3f}\np: {p:.3f}',
			transform=ax.transAxes)
		result={'m':m,'c':c,'r':r,'p':p,
			'feature':feature,
			'sampler 1':s1,
			'sampler 2':s2}
		results.append(result)
		ax.set_title(features[feature])
		ax.set_xlabel(s1)
		ax.set_ylabel(s2)
		ax.get_legend().remove()
		fig.tight_layout()
		fig.savefig(f'{feature} - {s1} vs {s2}.png')
		plt.close()
regressions = pd.DataFrame(results)
regressions.to_csv('regressions.csv',index=False)
