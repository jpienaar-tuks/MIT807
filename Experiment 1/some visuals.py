import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('Experiment 1 Accuracy scores.csv', index_col=0)

fig, ax = plt.subplots(1,2)
fig.tight_layout()
pdf = df.loc[df['sample_size']==1000]
pdf['accuracy']*=100
order=['Random walk','Hilbert curve','Latin Hypercube']

for i, (cls, cdf) in enumerate(pdf.groupby('classifier')):
    sns.barplot(cdf, x='feature_set', y='accuracy', hue='sampler', hue_order=order,
                errorbar=None, ax=ax[i])
    #for sampler, sdf in cdf.groupby('sampler'):
    #    ax[i].bar(x=df['feature_set'], height=df['accuracy'], label=sampler)
    ax[i].legend()
    ax[i].set_title(f'{cls.upper()} classifier')
    ax[i].set_xlabel('FLA feature set')
plt.show()

#FG = sns.FacetGrid(df, row='sample_size', col='classifier', hue='sampler')
#g=FG.map_dataframe(plt.bar, x='feature_set', height='accuracy')
#FG.add_legend()
