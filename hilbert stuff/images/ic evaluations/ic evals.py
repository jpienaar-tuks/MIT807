import sqlite3
import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pattern = re.compile(r'([A-Za-z ]*)\(([a-z]*)\)')
def upcase(sampler):
    match = pattern.match(sampler)
    if match:
        return f'{match.group(1)}({match.group(2).upper()})'
    else:
        return sampler

conn = sqlite3.connect(os.path.realpath(os.path.join(os.getcwd(),'..','..','pflacco.sqlite')))
cur = conn.cursor()

df=pd.read_sql('select * from repetitions2',conn)
df.loc[df['feature']=='ic_eps.max','feature']='ic.eps.max'
df.sampler = df.sampler.apply(upcase)
fdir=os.path.dirname(__file__)


for dim, ddf in df.groupby('dimensions'):
    for n, ndf in ddf.groupby('sample_n'):
        for feature, fdf in ndf.groupby('feature'):
            if feature[:2]=='ic':
                fig, ax = plt.subplots(figsize=[12,6])
                if feature == 'ic.eps_max':
                    ax.set_yscale('log')
                else:
                    ax.set_yscale('linear')
                if feature in ['ic.eps_max', 'ic.eps_s', 'ic.eps_ration']:
                    sns.stripplot(fdf, y='value', x='function',
                                orient='v', hue='sampler', dodge=True,
                                ax=ax)
                else:
                    g=sns.boxplot(fdf, y='value', x='function',
                                orient='v', hue='sampler',
                                fliersize=0,
                                ax=ax)
                ax.vlines(np.arange(0.5,24.5), *ax.get_ybound(), linewidth=0.5, linestyles='dashed')
                os.makedirs(os.path.join(fdir, str(dim), str(n)), exist_ok=True)
                fig.tight_layout()
                fig.savefig(os.path.join(fdir, str(dim), str(n), f'{feature} - {n} - {dim}.png'),dpi=200)
                print(os.path.join(fdir, str(dim), str(n), f'{feature} - {n} - {dim}.png'))
                plt.close()
        
