# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 11:39:45 2023

@author: Johann
"""
from hilbertcurve.hilbertcurve import HilbertCurve
from scipy.stats import qmc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def greedy_nearest_neighbor(points):
    result=np.zeros(points.shape)
    result[0] = points[0]
    points = np.delete(points,0,0)
    for i in range(len(points)):
        distances = np.sum((result[i]-points)**2, axis=1)
        nearest=np.argmin(distances)
        result[i+1]=points[nearest]
        points = np.delete(points, nearest, 0)
    return result

def step_sizes(points):
    return [np.sqrt(np.sum((points[i] - points[i+1])**2)) for i in range(len(points)-1)]

def random_walk(steps, dimensions):
    walk = np.cumsum(np.random.uniform(-1,1,(steps, dimensions)), axis=0)
    scaled_walk = (walk - np.min(walk, axis=0))/(np.max(walk, axis=0)-np.min(walk, axis=0))
    return scaled_walk
        
        

# number of points generated: 2**(iterations*dimension)
iterations=4
dimensions=2


# Generate Hilbert curve:
hilbert_curve = HilbertCurve(iterations, dimensions)
distances = list(range(hilbert_curve.max_h+1))
points = hilbert_curve.points_from_distances(distances)
# Scale to unit square
points = np.array(points)/hilbert_curve.max_x
hilbert_curve_df = pd.DataFrame(points, columns=['x','y'])
hilbert_curve_df['name'] = 'Hilbert Curve'

# Select random points along edges:
points_on_edges=[]
for i in range(hilbert_curve.max_h):
    r = np.random.rand()
    points_on_edges.append(r*points[i]+(1-r)*points[i+1])
points_on_edges =np.array(points_on_edges)
points_on_edges_df = pd.DataFrame(points_on_edges, columns=['x','y'])
points_on_edges_df['name'] = 'Points on Edges'
    
# Select random points around vertexes:
points_near_vertexes=[]
for point in points:
    points_near_vertexes.append(np.random.normal(point,0.5/hilbert_curve.max_x))
points_near_vertexes = np.array(points_near_vertexes)
points_near_vertexes_df = pd.DataFrame(points_near_vertexes, columns=['x','y'])
points_near_vertexes_df['name'] = 'Points near Vertexes'

# Latin Hypercube sampling and ordering
LHS_sampler = qmc.LatinHypercube(dimensions)
LHS_samples = LHS_sampler.random(2**(dimensions*iterations))
LHS_samples_ordered = greedy_nearest_neighbor(LHS_samples)
LHS_samples_df = pd.DataFrame(LHS_samples_ordered, columns=['x','y'])
LHS_samples_df['name'] = 'Latin Hypercube'

# Random walk sampling
RW_samples = random_walk(2**(dimensions*iterations),dimensions)
RW_samples_df = pd.DataFrame(RW_samples, columns=['x','y'])
RW_samples_df['name']='Random walk'
    
# Plotting
fig, ax = plt.subplots()
ax.plot(points[:,0], points[:,1], 'b-', linewidth=0.5, label='Hilbert Curve')
ax.plot(points_on_edges[:,0], points_on_edges[:,1],'ro', markersize=1, label='Points on edges')
ax.plot(points_near_vertexes[:,0], points_near_vertexes[:,1], 'go', markersize=1, label='Points near vertexes')
ax.plot(LHS_samples[:,0], LHS_samples[:,1], color='black', marker='+', linestyle='none', markersize=1.5, label='LHS')
legend=plt.figlegend(bbox_to_anchor=(0.5,0.9),loc='lower center', ncol=3)
fig.savefig('2D sample of hilbert curve and sampling strategies.png', dpi=400, bbox_inches='tight')
fig.show()

for df, name in zip([hilbert_curve_df, points_on_edges_df, points_near_vertexes_df, LHS_samples_df, RW_samples_df],
                    ['Hilbert Curve', 'Points on Edges','Points near Vertexes (0.5)','Latin Hypercube', 'Random walk']):
    sns.jointplot(data=df, x='x', y='y', marginal_kws=dict(bins=50)).figure.savefig(f'{name}.png', dpi=400)

# df=pd.concat([hilbert_curve_df, points_near_vertexes_df, points_on_edges_df, LHS_samples_df, RW_samples_df],axis=0, ignore_index=True)
# g=sns.JointGrid(data=df, x='x', y='y', hue='name')
# g.plot_joint(sns.scatterplot)
# g.plot_marginals(sns.histplot, bins=25)
# g.savefig('JointPlot hist.png',dpi=400)

sample_step_sizes = {j: step_sizes(i) for i, j in zip([points_near_vertexes, points_on_edges, LHS_samples_ordered, RW_samples], 
                                               ['Points near Vertexes','Points on Edges', 'Ordered Latin Hypercube','Random walk'])}

h=sns.histplot(data= sample_step_sizes, kde=True)
h.figure.savefig('Step size distribution.png',dpi=400)

