import sqlite3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from ioh import ProblemClass

conn = sqlite3.connect(os.path.realpath(os.path.join(os.getcwd(),'..','..','pflacco.sqlite')))
cur = conn.cursor()

df=pd.read_sql('select * from repetitions2 where feature like "ic%" and feature not like "%time"',conn)
fdir=os.path.dirname(__file__)
pc = ProblemClass.BBOB

for dim, ddf in df.groupby('dimensions'):
    for n, ndf in ddf.groupby('sample_n'):
        fig, ax = plt.subplots(2, 2, figsize=[10,10],sharex=False, sharey=False)
        axes = ax.ravel()
        for i, (sampler, sdf) in enumerate(ndf.groupby('sampler')):
            pivot_df = sdf.pivot_table(index=['test_id', 'function'], columns='feature', values='value')
            pivot_df.reset_index(1, inplace=True)
            #pivot_df['ic.eps_max'] = np.log(pivot_df['ic.eps_max']) # log scaling the variable doens't seem to make much difference
            tsne = TSNE()
            scaler=MinMaxScaler()
            X = scaler.fit_transform(pivot_df.loc[:,['ic' in i for i in pivot_df.columns]])
            X = tsne.fit_transform(X)
            markers = pivot_df['function'].map(str)

            sns.scatterplot(x=X[:,0], y=X[:,1], hue=markers, style=markers, ax=axes[i])
            #axes[i].axis('off')
            axes[i].get_legend().remove()
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_title(sampler)

        h, l = axes[0].get_legend_handles_labels()
        axes[3].legend(h, l, borderaxespad=0, loc='upper left', bbox_to_anchor = [0,1], ncols=2)
        axes[3].axis("off")
        fig.tight_layout()
        #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5, 0.2,0.1),
        #          mode='expand', borderaxespad=0, ncol=1)
        fig.savefig(f'tSNE visual {dim}d - {n*dim}.png', dpi=200)
        #fig.show()
        plt.close()
        
