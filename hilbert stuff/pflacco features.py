# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:07:30 2023

@author: b0727
"""
from hilbertcurve.hilbertcurve import HilbertCurve
from scipy.stats import qmc
from sklearn.preprocessing import MinMaxScaler
from functools import lru_cache
import cocoex
import pandas as pd
import numpy as np
import re, os, sqlite3, json, gc, warnings, time
from pflacco.classical_ela_features import calculate_dispersion, calculate_ela_distribution, calculate_ela_meta, calculate_limo
from my_pflacco import calculate_information_content

@lru_cache(maxsize=1)
def LRU_HilbertCurve(iterations, dimensions, n_procs=0):
    return HilbertCurve(iterations, dimensions, n_procs)


# pflacco 'calculate_information_content' already does nearest neighbor sorting
# def greedy_nearest_neighbor(points):
#     result=np.zeros(points.shape)
#     result[0] = points[0]
#     points = np.delete(points,0,0)
#     for i in range(len(points)):
#         distances = np.sum((result[i]-points)**2, axis=1)
#         nearest=np.argmin(distances)
#         result[i+1]=points[nearest]
#         points = np.delete(points, nearest, 0)
#     return result

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

def HC_sampler(dimensions, n, bounding_box=[-5,5]):
    # Pick number of iterations such that it will generate more than 
    # the required number of samples. Also set a minimum of 3 iterations
    iterations = max(3,int(np.ceil(np.log2(n)/dimensions)))
    # print(f'Iterations: {iterations}, estimated no of points generated: {2**(iterations*dimensions)}')
    hilbert_curve = HilbertCurve(iterations, dimensions, -1)
    # distances = list(range(hilbert_curve.max_h+1)) # <--- SUBSAMPLE THIS
    # print('selecting distances')
    distances = randomstate.choice(2**(iterations*dimensions)+1, n, replace = False, shuffle=False)
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
        points_near_vertexes.append(np.random.normal(point,0.3))
    points_near_vertexes = np.array(points_near_vertexes)
    # print('scaling points')
    points_near_vertexes = MinMaxScaler().fit_transform(points_near_vertexes)
    points_near_vertexes = hypercube_scaler(points_near_vertexes, bounding_box)
    # sub_sample = np.random.choice(len(points_near_vertexes),n,False)
    # sub_sample.sort()
    # return points_near_vertexes[sub_sample]
    return points_near_vertexes

def my_calculate_information_content(X, y):
    if name == 'Latin Hypercube':
        result=calculate_information_content(X, y)
    else:
        result=calculate_information_content(X, y, ic_sorting='none')
    return result


randomstate=np.random.default_rng()

r1=re.compile(r'BBOB suite problem f(\d+) instance (\d+) in (\d+)D')

# suite = cocoex.Suite('bbob','','dimensions:2,5,10,20')
suite = cocoex.Suite('bbob','','dimensions:2,5,10,20')
df=pd.DataFrame(data=np.array([r1.findall(i) for i in suite.problem_names]).reshape(-1,3),
                columns=['function no','instance no','dimensions'])
conn = sqlite3.connect(r'E:\MIT807\pflacco.sqlite')
cur = conn.cursor()
bounding_box=[-5,5]

# CREATE TABLE features (rowid integer primary key autoincrement, 
#                         function integer, 
#                         instance integer, 
#                         dimensions integer, 
#                         sample_n integer, 
#                         sampler text, 
#                         test_id integer, 
#                         pflacco_feature text, 
#                         pflacco_value float);
    

r2 = re.compile(r'bbob_f(\d+)_i(\d+)_d(\d+)')
samplers = {
            'Latin Hypercube': LHS_sampler,
            'Random walk': RW_sampler,
            'Hilbert curve': HC_sampler
            }
calculations = [calculate_dispersion, 
                calculate_ela_distribution, 
                calculate_ela_meta, 
                lambda X, y: calculate_limo(X,y,-5,5),
                my_calculate_information_content]

test_id=0
warnings.filterwarnings('ignore')
for problem in suite:
    function, instance, dimensions = [int(i) for i in r2.findall(problem.id)[0]]
    print(f'Starting with {problem.name}')
    start = time.perf_counter()
    for sample_size in [100,178,316, 562, 1000]:
        eval_budget = sample_size*dimensions
        for name, sampler in samplers.items():
            test_id+=1
            if test_id<18404: #to be continued tomorrow....
                continue
            samples = sampler(dimensions, eval_budget)
            db_list=[]
            y = [problem(i) for i in samples]
            for calculation in calculations:
                result = calculation(samples, y)
                for k, v in result.items():
                    db_list.append((function, instance, dimensions, sample_size, name, test_id, k, v))
            cur.executemany('insert into features (function, instance, dimensions, sample_n, sampler, test_id, pflacco_feature, pflacco_value) values (?,?,?,?,?,?,?,?)',
                            db_list)
            conn.commit()
            del db_list
            gc.collect()
    print('That took {}s'.format(time.perf_counter()-start))
conn.close()

            
                
