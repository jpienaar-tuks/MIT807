# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:39:16 2023

@author: Johann
"""
import numpy as np
import pandas as pd
import time

from datetime import timedelta

from sklearn.neighbors import NearestNeighbors 
from sklearn.preprocessing import MinMaxScaler

from typing import Dict, List, Optional, Union, Any, Tuple

from pflacco.utils import _validate_variable_types

from hilbertcurve.hilbertcurve import HilbertCurve

def calculate_information_content(
      X: Union[pd.DataFrame, np.ndarray, List[List[float]]],
      y: Union[pd.Series, np.ndarray, List[float]],
      ic_sorting: str = 'nn',
      ic_nn_neighborhood: int = 20,
      ic_nn_start: Optional[int] = None,
      ic_epsilon: List[float] = np.insert(10 ** np.linspace(start = -5, stop = 15, num = 1000), 0, 0),
      ic_settling_sensitivity: float = 0.05,
      ic_info_sensitivity: float = 0.5,
      seed: Optional[int] = None,
      return_step_sizes: Optional[bool] = False,
      return_permutation: Optional[bool] = False) -> Union[Dict[str, Union[int, float]], 
                                                          List[Union[List[float], List[int], Dict[str, Union[int, float]]]]]:
      """Calculation of Information Content features, similar to the R-package `flacco`.
      
      Modified to accept 'none' as a sorting option if the sample is already sorted.

      Parameters
      ----------
      X : Union[pd.DataFrame, np.ndarray, List[List[float]]]
          A collection-like object which contains a sample of the decision space.
          Can be created with :py:func:`pflacco.sampling.create_initial_sample`.
      y : Union[pd.Series, np.ndarray, List[float]]
          A list-like object which contains the respective objective values of `X`.
      ic_sorting : str, optional
          Sorting strategy, which is used to define the tour through the landscape.
          Possible values are 'none', nn' and 'random, by default 'nn'.
      ic_nn_neighborhood : int, optional
          Number of neighbours to be considered in the computation, by default 20.
      ic_nn_start : Optional[int], optional
          Indices of the observation which should be used as starting points.
          When none are supplied, these are chosen randomly, by default None.
      ic_epsilon : List[float], optional
          Epsilon values as described in section V.A of [1],
          by default `np.insert(10 ** np.linspace(start = -5, stop = 15, num = 1000), 0, 0)`.
      ic_settling_sensitivity : float, optional
          Threshold, which should be used for computing the settling sensitivity of [1], by default 0.05.
      ic_info_sensitivity : float, optional
          Portion of partial information sensitivity of [1], by default 0.5
      seed : Optional[int], optional
          Seed for reproducability, by default None

      Returns
      -------
      Dict[str, Union[int, float]]
          Dictionary consisting of the calculated features.

      References
      ----------
      [1] Muñoz, M.A., Kirley, M. and Halgamuge, S.K., 2014.
          Exploratory landscape analysis of continuous space optimization problems using information content.
          IEEE transactions on evolutionary computation, 19(1), pp.74-87.

      """      
      start_time = time.monotonic()
      X, y = _validate_variable_types(X, y)

      n = X.shape[1]
      ic_aggregate_duplicated = 'mean'
      if not np.issubdtype(ic_epsilon.dtype, np.number): 
            raise Exception('ic_epsilon contains non numeric data.')
      if ic_epsilon.min() < 0:
            raise Exception('ic_epsilon can only contain numbers in the intervall [0, inf)')
      if 0 not in ic_epsilon:
            raise Exception("One component of ic_epsilon has to be 0. Please add this component")
      if ic_sorting not in ['nn', 'random', 'none', 'hc']:
            raise Exception('Only "nn", "random" and "none" are valid parameter values for "ic_sorting"')
      if ic_settling_sensitivity < 0:
            raise Exception('"ic_settling_sensitivity must be larger than zero')
      if ic_info_sensitivity < -1 or ic_info_sensitivity > 1:
            raise Exception('"ic_settling_sensitivity must be larger than zero')
      if sum(X.duplicated()) == X.shape[0]:
            raise Exception('Can not IC compute information content features, because provided values are identical')
      epsilon = np.unique(ic_epsilon)


      # Duplicate check and mean aggregation for the objective variable, if only variables in the decision space are duplicated.
      dup_index = X.duplicated(keep = False)
      if dup_index.any():
            complete = pd.concat([X, pd.DataFrame(y, columns = ['y'])], axis = 1).duplicated()
            # Remove complete duplicates, because these cannot be aggregated using e.g. the mean of y
            if complete.any():
                  X = X[~complete]
                  y = y[~complete]
                  dup_index = X.duplicated(keep = False)
            
            # TODO Check with Pascal: the next line seems utterly pointless, a flip of the second array is missing. yes double flip.. that is why it is pointless.
            #dup_index = np.bitwise_or(dup_index, np.array(np.flip(dup_index)))
            Z = X[dup_index]
            v = y[dup_index]
            X = X[~dup_index]
            y = y[~dup_index]

            while len(v) > 1:
                  index = np.array([(Z.iloc[0] == Z.iloc[idx]).all() for idx in range(Z.shape[0])])
                  X = pd.concat([X, Z.iloc[[0]]], ignore_index = True)
                  Z = Z[~index]
                  y = pd.concat([y, pd.DataFrame([v[index].mean()])], ignore_index = True)
                  v = v[~index]

            
      if seed is not None and isinstance(seed, int):
            np.random.seed(seed)

      # dist based on ic_sorting
      if ic_sorting == 'random':
            permutation = np.random.choice(range(X.shape[0]), size = X.shape[0], replace = False)
            X = X.iloc[permutation].reset_index(drop = True)
            d = [np.sqrt((X.iloc[idx] - X.iloc[idx + 1]).pow(2).sum()) for idx in range(X.shape[0] - 1)]
      elif ic_sorting == 'none':
            permutation = list(range(X.shape[0]))
            d = [np.sqrt((X.iloc[idx] - X.iloc[idx + 1]).pow(2).sum()) for idx in range(X.shape[0] - 1)]
      elif ic_sorting == 'hc':
            hc_p = np.ceil(np.log2(X.shape[0]))
            hc = HilbertCurve(hc_p ,n)
            scaled_points = np.round(MinMaxScaler().fit_transform(X)*(2**hc_p-1))
            permutation = np.array(hc.distances_from_points(scaled_points), dtype=np.float64).argsort()
            X = X.iloc[permutation].reset_index(drop = True)
            d = [np.sqrt((X.iloc[idx] - X.iloc[idx + 1]).pow(2).sum()) for idx in range(X.shape[0] - 1)]
      else:
            if ic_nn_start is None:
                  ic_nn_start = np.random.choice(range(X.shape[0]), size = 1)[0]
            if ic_nn_neighborhood < 1 and ic_nn_neighborhood > X.shape[0]:
                  raise Exception(f'[{ic_nn_neighborhood}] is an invalid option for the NN neighborhood, because the sample only covers 1 to {X.shape[0]} observations.')
            nbrs = NearestNeighbors(n_neighbors = min(ic_nn_neighborhood, X.shape[0]), algorithm='kd_tree').fit(X)
            distances, indices = nbrs.kneighbors(X)

            current = ic_nn_start
            candidates = np.delete(np.array([x for x in range(X.shape[0])]), current)
            permutation = [current]
            permutation.extend([None] * (X.shape[0] - 1))
            dists = [None] * (X.shape[0])

            for i in range(1, X.shape[0]):
                  currents = indices[permutation[i-1]]
                  current = np.array([x for x in currents if x in candidates])
                  if len(current) > 0:
                        current = current[0]
                        permutation[i] = current
                        candidates = candidates[candidates != current]
                        dists[i] = distances[permutation[i - 1], currents == current][0]
                  else:
                        nbrs2 = NearestNeighbors(n_neighbors = min(1, X.shape[0])).fit(X.iloc[candidates])
                        distances2, indices2, = nbrs2.kneighbors(X.iloc[permutation[i - 1]].to_numpy().reshape(1, X.shape[1]))
                        current = candidates[np.ravel(indices2)[0]]
                        permutation[i] = current
                        candidates = candidates[candidates != current]
                        dists[i] = np.ravel(distances2)[0]

            d = dists[1:]

      # Calculate psi eps
      psi_eps = []
      y_perm = y[permutation]
      diff_y = np.ediff1d(y_perm)
      ratio = diff_y/d
      for eps in ic_epsilon:
            psi_eps.append([0 if abs(x) < eps else np.sign(x) for x in ratio])

      psi_eps = np.array(psi_eps)
      H = []
      M = []
      for row in psi_eps:
            # Calculate H values
            a = row[:-1]
            b = row[1:]
            probs = []
            probs.append(np.bitwise_and(a == -1, b == 0).mean())
            probs.append(np.bitwise_and(a == -1, b == 1).mean())
            probs.append(np.bitwise_and(a == 0, b == -1).mean())
            probs.append(np.bitwise_and(a == 0, b == 1).mean())
            probs.append(np.bitwise_and(a == 1, b == -1).mean())
            probs.append(np.bitwise_and(a == 1, b == 0).mean())
            H.append(-sum([0 if x == 0 else x * np.log(x)/np.log(6) for x in probs]))

            # Calculate M values
            n = len(row)
            row = row[row != 0]
            len_row = len(row[np.insert(np.ediff1d(row) != 0, 0, False)]) if len(row) > 0 else 0
            M.append(len_row/ (n - 1))

      H = np.array(H)
      M = np.array(M)
      eps_s = epsilon[H < ic_settling_sensitivity]
      eps_s = np.log(eps_s.min()) / np.log(10) if len(eps_s) > 0 else None

      m0 = M[epsilon == 0]
      eps05 = np.where(M > ic_info_sensitivity * m0)[0]
      eps05 = np.log(epsilon[eps05].max()) / np.log(10) if len(eps05) > 0 else None
      
      result_dict = {
            'ic.h_max': H.max(),
            'ic.eps_s': eps_s,
            'ic_eps.max': np.median(epsilon[H == H.max()]),
            'ic.eps_ration': eps05,
            'ic.m0': m0[0],
            'ic.costs_runtime': timedelta(seconds=time.monotonic() - start_time).total_seconds()
      }
      
      return_list=[]
      if return_step_sizes:
        return_list.append(d)
      if return_permutation:
        return_list.append(permutation)
      if len(return_list)>0:
        return_list.append(result_dict)
      else:
        return_list = result_dict
      return return_list
        
        
