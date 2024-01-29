import os, re
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import seaborn as sns

pattern = re.compile(r'([A-Za-z ]*)\(([a-z]*)\)')
style_cycler = (cycler(color=['r','g','y','b']) +
                cycler(linestyle=['-','--', ':', '-.']))
plt.rc('axes', prop_cycle = style_cycler)

def upcase(sampler):
    match = pattern.match(sampler)
    if match:
        return f'{match.group(1)}({match.group(2).upper()})'
    else:
        return sampler

dirpath = os.path.dirname(__file__)
conn =sqlite3.connect(os.path.realpath(os.path.join(dirpath,'..','..','pflacco.sqlite')))
df = pd.read_sql('select * from timings2 where feature like "t.%"', conn)
df['n']=df['dimensions']*df['sample_n']
df.sampler = df.sampler.apply(upcase)


pivot_df = df.pivot_table(index=['test_id','n','sampler'], values='value', columns='feature')
pivot_df['t.total']=pivot_df.sum(axis=1)
pdf = pivot_df.groupby(level=[1,2]).mean()
pdf.reset_index(inplace=True)

pdf2=pdf.copy()
pdf2.loc[pdf.sampler.map(lambda x: 'Latin' in x),'sampler'] = 'Latin Hypercube'

for chart in ['t.generation','t.total', 't.ic.calculate']:
    #fig, ax = plt.subplots()
    if chart == 't.generation':
        sns.lmplot(pdf2, x='n', y=chart, hue='sampler', order=2,
               ci=None, facet_kws={'legend_out':None})
    else:
        sns.lmplot(pdf.loc[pdf['sampler'].map(lambda x: 'hc' not in x),:], # upcase 'hc'???
               x='n', y=chart, hue='sampler', order=2,
               ci=None, facet_kws={'legend_out':None})
    fig, ax = plt.gcf(), plt.gca()
    ax.set_xlabel('Sample ($n * d$)')
    ax.set_ylabel('Time (s)')
    fig.tight_layout()
    fig.savefig(f'{chart}2.png', dpi=400)
    fig.show()
