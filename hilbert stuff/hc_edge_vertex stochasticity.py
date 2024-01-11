from hilbertcurve.hilbertcurve import HilbertCurve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def dist(points):
    d=[]
    for i in range(len(points)-1):
        d.append(np.sqrt(np.sum(np.power(points[i]-points[i+1],2))))
    return d
        

d = 2
p = 8
hc = HilbertCurve(p,d)
distances = list(range(2**(p*d)))
points = hc.points_from_distances(distances)
points = np.array(points)
print('done base curve')
points_on_edges=[]
for i in range(hc.max_h):
    r = np.random.rand()
    points_on_edges.append(r*points[i]+(1-r)*points[i+1])
points_on_edges =np.array(points_on_edges)
print('done edges...')
points_near_vertexes=[]
for point in points:
    points_near_vertexes.append(np.random.normal(point,0.3))
points_near_vertexes = np.array(points_near_vertexes)
print('done vertexes')
#### set p = 3 and comment section below
##fig, ax = plt.subplots()
##ax.plot(points[:,0], points[:,1],'b-')
##ax.scatter(points_on_edges[:,0],points_on_edges[:,1],color='r')
##fig.savefig('Hilbert curve stochasticity along edges.png', dpi=300)
##
##fig, ax = plt.subplots()
##ax.plot(points[:,0], points[:,1],'b-')
##for point, vertex in zip(points, points_near_vertexes):
##    ax.scatter(vertex[0],vertex[1],color='r')
##    ax.plot((point[0], vertex[0]),(point[1],vertex[1]),'g-', linewidth=0.5)
##fig.savefig('Hilbert curve stochasticity at vertexes.png', dpi=300)
##

## set p=8 and comment section above
dist_dict={'Base curve':dist(points),
           'Along edges': dist(points_on_edges)+[0],
           'Near vertices': dist(points_near_vertexes)}

dist_df = pd.DataFrame(dist_dict)
fig, ax = plt.subplots()
h=sns.histplot(dist_df.iloc[:,1:], kde=True)
ax.set_xlabel('Step size')
fig.savefig('stoch_stepSize.png',dpi=300)
fig.show()

