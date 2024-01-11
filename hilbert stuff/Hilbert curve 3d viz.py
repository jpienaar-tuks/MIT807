from hilbertcurve.hilbertcurve import HilbertCurve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

p=2
d=3
h_max = 2**(p*d)
hc = HilbertCurve(p,d)
points=[]
for i in range(h_max):
    points.append(hc.point_from_distance(i))
points=np.array(points)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for i in range(h_max-1):
	ax.plot(points[i:i+2,0],points[i:i+2,1],points[i:i+2,2], c=cm.viridis(i/h_max))
fig.tight_layout()
fig.savefig(f'Hilbert curve {p} order.png', dpi=200)
fig.show()	
