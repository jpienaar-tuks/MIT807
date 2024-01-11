from hilbertcurve.hilbertcurve import HilbertCurve
import matplotlib.pyplot as plt
import numpy as np

d=2
for p in [1,2,3,4,5,6]:
    hc = HilbertCurve(p,d)
    distances = list(range(2**(p*d)))
    points = hc.points_from_distances(distances)
    points = np.array(points)

    fig, ax = plt.subplots()
    ax.plot(points[:,0], points[:,1],'b-')
    if p in [1,2,3]:
        for i, point in enumerate(points):
            ax.annotate(i, point)
    fig.savefig(f'Hilbert curve (p_eq_{p}).png', dpi=300)
    fig.show()
