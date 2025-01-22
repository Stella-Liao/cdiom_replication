import pandas as pd
import numpy as np
import geopandas as gpd
import libpysal.weights as sw
import copy
from copy import deepcopy

from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
import warnings

class ConstantTerm():
    
    def __init__(self, n):
        self._x = np.ones(n).reshape(-1,1)     
        
    @property
    def X(self):
        return self._x
    
    def __str__(self):
        return '<%s, %s:%s>' % (self.__class__.__name__)
    
class LinearTerm():
    
    def __init__(self, df, *idx, standard = False, log = False):
        self._df = df
        self._idx = list(idx) 
        X = self._df.iloc[:, self._idx].apply(pd.to_numeric).values
        
        if log == True:
            X = np.log(np.where(X > 0, X, 1.0)) # replace it to 1 then log(1) is 0 
          
        if standard == True:
            #X = (X-np.min(X))/(np.max(X)-np.min(X))
            #X = (X-np.mean(X))/np.std(X)
            
            Xmean = np.mean(X, axis=0)
            Xstd = np.std(X, axis=0)
            X = (X - Xmean) / Xstd
            
        self._x = X
    
    @property
    def X(self):
        return self._x
    
    def __str__(self):
        return '<%s, %s:%s>' % (self.__class__.__name__)
    
class SATerm:
    def __init__(self, od_data, dest_data, o_ids, d_ids, dest_ids, dest_attr,  
                 log=False, standard=False, lower_bound = None, upper_bound = None, CI_step = None,
                 initial_sigma = None):
        
        # Initilization
        self._od_data = od_data
        self._o_ids = od_data[o_ids].values.flatten()
        self._d_ids = od_data[d_ids].values.flatten()
    
        self._dest_data = dest_data
        self._dest_ids = dest_data[dest_ids].values.flatten()
        self._dest_attr = dest_data[dest_attr].values.flatten()
        
        self.initial_sigma = initial_sigma if initial_sigma is not None else -1.0
        self.sigma = None
        self.neighbors = None
        self.weights = None
        self.int_score = False
        self.CI_step = CI_step if CI_step is not None else 0.01
        
        self.log = log
        self.standard = standard
        self.lower_bound = lower_bound if lower_bound is not None else -3.0
        self.upper_bound = upper_bound if upper_bound is not None else 0.0
        
        # Get the Distance Matrix for `cal()`
        distmat = sw.distance.DistanceBand.from_dataframe(dest_data, threshold=9999999999, binary=False, alpha = 1)
        distmat = distmat.full()[0] # check whether there is distance smaller than 1, expect for distance to itself
        self._distmat = distmat.copy()
        self._distmat[self._distmat < 1] = 1.0 
        
    def cal(self, sigma = None):
        
        if sigma is None:
            sigma = self.initial_sigma
            warnings.warn(f"Sigma is not set. Defaulting to the initial one will.", UserWarning)
        
        self.sigma = sigma
        sat_w = np.power(self._distmat, sigma)
        np.fill_diagonal(sat_w, 0) # exclude self-beighbor because np.power(0, 0) = 1 
        
        sat_array = sat_w @ self._dest_attr.reshape(-1,1)
        
        if self.log == True:
            sat_array = np.log(np.where(sat_array > 0, sat_array, 1.0)) # replace it to 1 then log(1) is 0 
    
        if self.standard == True:
            sat_array = (sat_array-np.mean(sat_array))/np.std(sat_array)
        
        dest_df = pd.DataFrame({'dest': self._dest_ids})
        dest_df = dest_df.assign(sat = sat_array)
        flow_df = pd.DataFrame({'orig': self._o_ids, 'dest': self._d_ids})
        
        flow_df['merge_order'] = range(len(flow_df))
        # Perform the merge
        fin_df = flow_df.merge(dest_df, on='dest')
        fin_df.sort_values('merge_order', inplace=True)
        fin_df.reset_index(inplace = True)
        # Optionally, drop the unnecessary columns
        fin_df.drop(['merge_order', 'index'], axis=1, inplace=True)
        
        fin_sat_array = fin_df.iloc[:, -1].values.flatten().reshape(-1,1)
        
        self.fin_df = fin_df
        self.dest_df = dest_df
        
        return fin_sat_array
    
    def show(self, sigma = None):
        """
        it is function to show the weights and neighbors for each destination in od_data with given sigma
        """
        
        if sigma is None:
            sigma = self.sigma
            warnings.warn(f"Sigma is not set. Defaulting to the latest one will.", UserWarning)
        
        dest_coords = np.array(list(self._dest_data.geometry.apply(lambda geom: (geom.x, geom.y))))
        tree = cKDTree(dest_coords)
        distances = {}  
        neighbors = {} 
        weights = {}
        
        k_neighbors = len(self._dest_data) 
        
        for i, point in enumerate(dest_coords):
            dists, inds = tree.query(point, k = k_neighbors)
            """
            because we only consider one spatial support so far, so removing the first one is necessary
            """
            dists, inds = dists[1:], inds[1:] 
            neighbor_ids = [self._dest_ids[ind] for ind in inds]
            powered_dists = np.power(dists, sigma)
        
            neighbors[self._dest_ids[i]] = neighbor_ids
            distances[self._dest_ids[i]] = dists
            weights[self._dest_ids[i]] = powered_dists
            
        self.weights = weights
        self.neighbors = neighbors
        
        return weights, neighbors 

class DIOTerm:
    def __init__(self, od_data, orig_data, o_ids, d_ids, orig_ids, orig_attr,  
                 log=False, standard=False, lower_bound = None, upper_bound = None, CI_step = None,
                 initial_sigma = None):
        
        # Initilization
        self._od_data = od_data
        self._o_ids = od_data[o_ids].values.flatten()
        self._d_ids = od_data[d_ids].values.flatten()
    
        self._orig_data = orig_data
        self._orig_ids = orig_data[orig_ids].values.flatten()
        self._orig_attr = orig_data[orig_attr].values.flatten()
        
        self.initial_sigma = initial_sigma if initial_sigma is not None else -1.0
        self.sigma = None
        self.neighbors = None
        self.weights = None
        self.int_score = False
        self.CI_step = CI_step if CI_step is not None else 0.01
        
        self.log = log
        self.standard = standard
        self.lower_bound = lower_bound if lower_bound is not None else -3.0
        self.upper_bound = upper_bound if upper_bound is not None else 0.0
        
        # Get the Distance Matrix for `cal()`
        distmat = sw.distance.DistanceBand.from_dataframe(orig_data, threshold=9999999999, binary=False, alpha = 1)
        distmat = distmat.full()[0] # check whether there is distance smaller than 1, expect for distance to itself
        self._distmat = distmat.copy()
        self._distmat[self._distmat < 1] = 1.0 
        
    def cal(self, sigma = None):
        
        if sigma is None:
            sigma = self.initial_sigma
            warnings.warn(f"Sigma is not set. Defaulting to the initial one will.", UserWarning)
        
        self.sigma = sigma
        iot_w = np.power(self._distmat, sigma)
        np.fill_diagonal(iot_w, 0) # exclude self-beighbor because np.power(0, 0) = 1 
        
        iot_array = iot_w @ self._orig_attr.reshape(-1,1)
        
        if self.log == True:
            iot_array = np.log(np.where(iot_array > 0, iot_array, 1.0)) # replace it to 1 then log(1) is 0 
    
        if self.standard == True:
            iot_array = (iot_array-np.mean(iot_array))/np.std(iot_array)
        
        orig_df = pd.DataFrame({'orig': self._orig_ids})
        orig_df = orig_df.assign(iot = iot_array)
        flow_df = pd.DataFrame({'orig': self._o_ids, 'dest': self._d_ids})
        
        flow_df['merge_order'] = range(len(flow_df))
        # Perform the merge
        fin_df = flow_df.merge(orig_df, on='orig')
        fin_df.sort_values('merge_order', inplace=True)
        fin_df.reset_index(inplace = True)
        # Optionally, drop the unnecessary columns
        fin_df.drop(['merge_order', 'index'], axis=1, inplace=True)
        
        fin_iot_array = fin_df.iloc[:, -1].values.flatten().reshape(-1,1)
        
        self.fin_df = fin_df
        self.orig_df = orig_df
        
        return fin_iot_array
    
    def show(self, sigma = None):
        """
        it is function to show the weights and neighbors for each destination in od_data with given sigma
        """
        
        if sigma is None:
            sigma = self.sigma
            warnings.warn(f"Sigma is not set. Defaulting to the latest one will.", UserWarning)
        
        orig_coords = np.array(list(self._orig_data.geometry.apply(lambda geom: (geom.x, geom.y))))
        tree = cKDTree(orig_coords)
        distances = {}  
        neighbors = {} 
        weights = {}
        
        k_neighbors = len(self._orig_data) 
        
        for i, point in enumerate(orig_coords):
            dists, inds = tree.query(point, k = k_neighbors)
            dists, inds = dists[1:], inds[1:] # removing itself
            neighbor_ids = [self._orig_ids[ind] for ind in inds]
            powered_dists = np.power(dists, sigma)
        
            neighbors[self._orig_ids[i]] = neighbor_ids
            distances[self._orig_ids[i]] = dists
            weights[self._orig_ids[i]] = powered_dists
            
        self.weights = weights
        self.neighbors = neighbors
        
        return weights, neighbors 
    