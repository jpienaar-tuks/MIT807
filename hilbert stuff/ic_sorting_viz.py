from scipy.stats.qmc import LatinHypercube
import my_pflacco
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewtest
from ioh import get_problem, ProblemClass
import warnings

warnings.filterwarnings('ignore')

n=200
d=2
LHS = LatinHypercube(d)
X = LHS.random(n)*10-5
f = get_problem(1,1,d,ProblemClass.BBOB)
y = np.array(list(map(f, X)))

methods = {'nn':'Nearest neighbor',
           'hc':'Hilbert Curve',
           #'random':'Random'
           }
d = {}
p = {}
for method in methods.keys():
    d[method], p[method], _ = my_pflacco.calculate_information_content(X,y,ic_sorting=method, return_step_sizes=True, return_permutation=True)

fig, ax = plt.subplots(len(methods), 3, sharex='col')
for i, method in enumerate(methods.keys()):
    ax[i,0].plot(X[p[method],0], X[p[method],1], 'r-')
    s=ax[i,0].scatter(X[:,0], X[:,1], c=y, cmap='viridis')
    ax[i,0].set_ylabel(methods[method])
    if i==0:
        ax[i,0].set_title("Sample and ordering")
    fig.colorbar(s,ax=ax[i,0])

    d[method].insert(0,0)
    d[method]=np.array(d[method])

    mean = d[method].mean()
    std = d[method].std()
    skew, p_value = skewtest(d[method])
    d_max = d[method].max()
    total=d[method].sum()
    ax[i,1].hist(d[method], bins=25)
    ax[i,1].text(0.75,0.95,f'Mean: {mean:.2f}\nStd.dev: {std:.2f}\nMax: {d_max:.2f}\nSkew: {skew:.2f}\nTotal: {total:.2f}',
                 va='top', ha='left',
                 transform=ax[i,1].transAxes)
    if i==0:
        ax[i,1].set_title(f'Step size distribution')
    
    ax[i,2].plot(np.cumsum(d[method]),y[p[method]])
    #ax[i,2].get_shared_x_axes().remove(ax[2,2])
    #ax[i,2].set_xticks([])
    if i==0:
        ax[i,2].set_title(f'Fitness vs cumulative distance')

fig.show()
