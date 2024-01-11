from hilbertcurve.hilbertcurve import HilbertCurve
from scipy.stats import qmc
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances_argmin_min as min_dists
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from time import perf_counter
import re, os, sqlite3, json, gc, warnings, time

def step_sizes(points):
    return [np.sqrt(np.sum((points[i] - points[i+1])**2)) for i in range(len(points)-1)]

def hausdorff(X,Y):
    dists = euclidean_distances(X,Y)
    return max([dists.min(axis=0).mean(), dists.min(axis=1).mean()])

def hausdorff2(X,Y):
	return max([min_dists(X,Y)[1].mean(), min_dists(Y,X)[1].mean()])

def hypercube_scaler(points, bounding_box):
    span = max(bounding_box)-min(bounding_box)
    offset = min (bounding_box)
    # print(f'span:{span} offset:{offset}')
    return points*span+offset

def LHS_sampler(dimensions, n, bounding_box=[-5,5]):
    LHS_sampler = qmc.LatinHypercube(dimensions)
    LHS_samples = LHS_sampler.random(n)
    LHS_samples_scaled = hypercube_scaler(LHS_samples, bounding_box)
    # pflacco 'calculate_information_content' already does nearest neighbor sorting
    # LHS_samples_ordered = greedy_nearest_neighbor(LHS_samples_scaled)
    return LHS_samples_scaled

def RW_sampler(dimensions, n, bounding_box=[-5,5]):
    walk = np.cumsum(np.random.uniform(-1,1,(n, dimensions)), axis=0)
    walk = MinMaxScaler().fit_transform(walk)
    scaled_walk = hypercube_scaler(walk, bounding_box)
    return scaled_walk

def HC_sampler(dimensions, n, bounding_box=[-5,5], sigma=0.3):
    # Pick number of iterations such that it will generate more than 
    # the required number of samples. Also set a minimum of 3 iterations
    iterations = max(3,int(np.ceil(np.log2(n)/dimensions)))
    # print(f'Iterations: {iterations}, estimated no of points generated: {2**(iterations*dimensions)}')
    hilbert_curve = HilbertCurve(iterations, dimensions, -1)
    # distances = list(range(hilbert_curve.max_h+1)) # <--- SUBSAMPLE THIS
    # print('selecting distances')
    int64_max = np.iinfo(np.int64).max
    if 2**(iterations*dimensions)+1 <= int64_max:
        distances = randomstate.choice(2**(iterations*dimensions)+1, n, replace = False, shuffle=False)
    else:
        distances = randomstate.choice(int64_max, n, replace = False, shuffle=False)
        distances = list(map(lambda x: int(x*(2**(iterations*dimensions)+1)/int64_max), distances))
    distances.sort()
    # print('generating points')
    points = []
    for i, d in enumerate(distances):
        points.append(hilbert_curve.point_from_distance(d))
        # if i%100:
        #     print(f'i: {i}, d: {d}')
    # Scale to unit square
    # points = np.array(points)/hilbert_curve.max_x
    points_near_vertexes=[]
    # print('randomising points')
    for point in points:
        points_near_vertexes.append(np.random.normal(point, sigma))
    points_near_vertexes = np.array(points_near_vertexes)
    # print('scaling points')
    points_near_vertexes = MinMaxScaler().fit_transform(points_near_vertexes)
    points_near_vertexes = hypercube_scaler(points_near_vertexes, bounding_box)
    # sub_sample = np.random.choice(len(points_near_vertexes),n,False)
    # sub_sample.sort()
    # return points_near_vertexes[sub_sample]
    return points_near_vertexes


randomstate=np.random.default_rng()

samplers = {
            'Latin Hypercube': LHS_sampler,
            'Random walk': RW_sampler,
            'Hilbert curve 0.1': lambda dim, n: HC_sampler(dim, n, sigma=0.1),
            'Hilbert curve 0.3': lambda dim, n: HC_sampler(dim, n, sigma=0.3),
            'Hilbert curve 0.5': lambda dim, n: HC_sampler(dim, n, sigma=0.5),
            'Hilbert curve 0.7': lambda dim, n: HC_sampler(dim, n, sigma=0.7),
            }
dims = [5,10,20,30]
sample_sizes=[100,316,1000]
results=[]
for run in range(30):
    for dim in dims:
        for sample_size in sample_sizes:
            for name, sampler in samplers.items():
                sample = sampler(dim, sample_size*dim)
                dh = hausdorff2(sample, randomstate.uniform(-5,5,size=sample.shape))
                result = {'run':run, 'dimensions':dim, 'sample_n':sample_size, 'sampler':name,'dh':dh}
                print(result)
                results.append(result)
df=pd.DataFrame(results)
df.to_csv('Hausdorff distances2.csv', index=False)
