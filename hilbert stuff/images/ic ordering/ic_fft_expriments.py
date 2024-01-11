from scipy.stats.qmc import LatinHypercube

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewtest
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from ioh import get_problem, ProblemClass
import warnings
import my_pflacco

warnings.filterwarnings('ignore')

def ordering_viz(X,y,p,d, ax):
    ax.set_title('Sample and Ordering')
    ax.plot(X[p,0], X[p,1], 'r-')
    s = ax.scatter(X[:,0], X[:,1], c=y, cmap='brg')
    plt.gcf().colorbar(s, ax=ax)
    return ax

def step_size_viz(X,y,p,d, ax):
    mean = d.mean()
    std = d.std()
    skew, p_value = skewtest(d)
    d_max = d.max()
    total=d.sum()
    ax.hist(d, bins=np.linspace(0,4,30))
    ax.text(0.75,0.95,f'Mean: {mean:.2f}\nStd.dev: {std:.2f}\nMax: {d_max:.2f}\nSkew: {skew:.2f}\nTotal: {total:.2f}',
                 va='top', ha='left',
                 transform=ax.transAxes)
    ax.set_title('Step size distribution')
    return ax
    

def fitness_viz(X,y,p,d, ax):
    ax.plot(np.cumsum(d),y[p])
    ax.set_title('Fitness vs cumulative distance')
    return ax


n=200
d=5
LHS = LatinHypercube(d)
X = LHS.random(n)*10-5
f = get_problem(5,1,d,ProblemClass.BBOB)
y = np.array(list(map(f, X)))

methods = {'nn':'Nearest neighbor',
           'hc':'Hilbert Curve',
           'random':'Random'
           }

visualisations = {'Ordering viz': ordering_viz,
                  'Step size viz': step_size_viz,
                  'Fitness viz': fitness_viz,
                  }

d = {}
p = {}
for method in methods.keys():
    d[method], p[method], _ = my_pflacco.calculate_information_content(X,y,ic_sorting=method, return_step_sizes=True, return_permutation=True)

if False:
    for method in methods.keys():
        d[method].insert(0,0)
        d[method] = np.array(d[method])
        for viz_name, viz in visualisations.items():
            fig, ax = plt.subplots()
            viz(X, y, p[method], d[method], ax)
            fig.tight_layout()
            fig.savefig(f'{viz_name} - {methods[method]}.png')
        

##fig, ax = plt.subplots(len(methods), 3, sharex='col')
##for i, method in enumerate(methods.keys()):
##    ax[i,0].set_ylabel(methods[method])
##
##    ax[i,2].plot(np.cumsum(d[method]),y[p[method]])
##    #ax[i,2].get_shared_x_axes().remove(ax[2,2])
##    #ax[i,2].set_xticks([])
##    if i==0:
##        ax[i,2].set_title(f'Fitness vs cumulative distance')
##
##fig.show()

fig, ax = plt.subplots(3,2)
xf=fftfreq(n)
scaler=StandardScaler()
scaler.fit(y.reshape(-1,1))
for row, method in enumerate(methods.keys()):
    d[method].insert(0,0)
    ax[row,0].plot(np.cumsum(d[method]), y[p[method]])
    
    yf = fft(scaler.transform(y[p[method]].reshape(-1,1)))

    real_peak_idx = np.abs(yf.real).argmax()
    o=xf.argsort()
    ax[row,1].plot(xf[o], yf[o].real)
    ax[row,1].text(0.5, 0.9,
                   f'({xf[real_peak_idx]:.3f}, {yf.real[real_peak_idx,0]:.3f})',
                   transform=ax[row,1].transAxes)

##    ax[row,2].plot(xf[o], yf[o].imag)
##    imag_peak_idx=np.abs(yf.imag).argmax()
##    ax[row,2].text(0.5, 0.9,
##                   f'({xf[imag_peak_idx]:.3f}, {yf.imag[imag_peak_idx,0]:.3f})',
##                   transform=ax[row,1].transAxes)

##    abs_peak_idx=np.abs(yf).argmax()
##    ax[row,2].plot(xf[o], np.abs(yf[o]))
##    ax[row,2].text(0.5, 0.9,
##                   f'({xf[abs_peak_idx]:.3f}, {np.abs(yf)[abs_peak_idx,0]:.3f})',
##                   transform=ax[row,2].transAxes)

    ax[row,0].set_title('Fitness vs distance')
    ax[row,1].set_title('FFT - Real')
##    ax[row,2].set_title('FFT - Imag')
##    ax[row,2].set_title('Abs')
    ax[row,0].set_ylabel(methods[method])
fig.show()
