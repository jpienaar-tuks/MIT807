from hilbertcurve.hilbertcurve import HilbertCurve
from scipy.stats import qmc
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pandas as pd
from time import perf_counter

def step_sizes(points):
    return [np.sqrt(np.sum((points[i] - points[i+1])**2)) for i in range(len(points)-1)]

def hausdorff(X,Y):
    dists = euclidean_distances(X,Y)
    return max([dists.min(axis=0).mean(), dists.min(axis=1).mean()])

eval_budget=1000
randomstate = np.random.default_rng()
results=[]
for d in [2,3,5]:
    start=perf_counter()
    # make sure enough points are generated to:
    # a) fill the budget and
    # b) explore the interior of the hypercube (i.e.p=3)
    p = int(max(3,np.ceil(np.log2(eval_budget*d*5)/d)))
    print(f'd: {d}, p: {p}')
    hc = HilbertCurve(p,d)
    # Generate points and scale to unit hypercube as required by qmc.discrepancy
    Points=np.array(hc.points_from_distances(list(range(hc.max_h+1))))/hc.max_x 
    subsample=randomstate.choice(hc.max_h+1,eval_budget, replace=False)
    subsample.sort()
    for strat in ['base','subsample']:
        r={}
        if strat=='base':
            points=Points
        else:
            points=Points[subsample]
        r['strat']=strat
        r['dim']=d
        ss = step_sizes(points)
        r['mean_step_size'] = np.mean(ss)
        r['step_size_std_dev'] = np.std(ss)
        r['discrepancy'] = qmc.discrepancy(points, method='MD')
        r['hausdorff'] = hausdorff(points, randomstate.uniform(size=points.shape))
        results.append(r)
    print(f'That took {perf_counter()-start} s \n')

# put all results into a dataframe and copy to clipboard
df = pd.DataFrame(results)
df.melt(id_vars=['strat','dim'], var_name='Metric').to_clipboard()
    
