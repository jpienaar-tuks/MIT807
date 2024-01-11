from hilbertcurve.hilbertcurve import HilbertCurve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

p, d = 1, 2
hc = HilbertCurve(p, d) #p=3, d=2
dists = list(range(2**(p*d)))
points = hc.points_from_distances(dists)

sigmas = [{'sigma': 0.1, 'color': 'red'},
          {'sigma': 0.3, 'color': 'blue'},
          {'sigma': 0.5, 'color': 'green'},
          {'sigma': 0.7, 'color': 'magenta'}
          ]

points = np.array(points)

fig, axs = plt.subplots(2,2,figsize=(9.6,5.4))
for i in range(4):
    ax = axs.ravel()[i]
    ax.plot(points[:,0], points[:,1])
    sigma=sigmas[i]
    for j, point in enumerate(points):
        if j==0:
            ax.add_patch(plt.Circle(point, radius=sigma['sigma'], color=sigma['color'],
                                    alpha=0.9, label=f'$\sigma = {sigma["sigma"]}$'))
            ax.add_patch(plt.Circle(point, radius=2*sigma['sigma'], color=sigma['color'],
                                    alpha=0.1, label=f'$2\sigma = {2*sigma["sigma"]}$'))
        else:
            ax.add_patch(plt.Circle(point, radius=sigma['sigma'], color=sigma['color'],
                                    alpha=0.9))
            ax.add_patch(plt.Circle(point, radius=2*sigma['sigma'], color=sigma['color'],
                                    alpha=0.1))
        ax.legend(loc=1)#upper right
        ax.set_xlim(-1,2)
        ax.set_ylim(-1,2)
fig.tight_layout()
fig.savefig('Sigma decision.png',dpi=200)
fig.show()
