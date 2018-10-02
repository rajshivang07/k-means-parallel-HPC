%%cython -a

import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
from functools import partial
from distance_func import distance

def min_distance(d, centroids):
    
    """ Calculate the minimum distance from point d 
        to its nearest center in centroids."""
    dist = np.min(np.sum((centroids - d)**2, axis=1))
    return dist


# speed up by multiprocessing
def cost_p(data, centroids): 
    
    """ Calculate the cost of data with respect to 
    the current centroids and the new probability 
    distribution of each point for next sample"""    

    with Pool(processes = cpu_count()) as pool:
        partial_dist = partial(min_distance, centroids = centroids)
        min_dist = pool.map(partial_dist, data)
        cost = np.sum(min_dist)
        p = min_dist/cost
    return cost,p


def randomSample(x, a, p):
    np.random.seed()
    return np.random.choice(a = a, size = x , p =p)


## speed up with multiprocessing
def sample_new_p(data, distribution, l):
    
    """ Sample new centers"""  
    
    with Pool(processes = cpu_count()) as pool:
        partial_rc = partial(randomSample, a = len(distribution), p=distribution)
        index = pool.map(partial_rc,[1]*l)
    return np.squeeze(data[index,:],axis=(1,))


def min_distance_index_p(d, centroids):
    
    """ Return the index of the minimum distance from point d 
        to its nearest center in centroids."""
    
    minInd = np.argmin(np.sum((centroids - d)**2, axis=1))
    return minInd 

## speed up with multiprocessing
def get_weight_p(data, centroids):
    
    """ Return weight of all centroids """

    with Pool(processes = cpu_count()) as pool:
        partial_minInd = partial(min_distance_index_p, centroids = centroids )
        min_index = pool.map(partial_minInd, data)
        count = np.array([np.sum(np.array(min_index) == i) for i in range(centroids.shape[0])])
    return count/np.sum(count)

def ScalableKMeansPlusPlus_p(data, k, l,iter=5):
    
    """ Apply the KMeans|| clustering algorithm"""
    
    centroids = data[np.random.choice(range(data.shape[0]),1), :]    
    
    for i in range(iter):   
        
        # Calculate the cost and new distribution
        norm_const = cost_p(data, centroids)[0]
        p = cost_p(data, centroids)[1] 
        
        # Sample the several(l) new centers and append them to the original ones
        centroids = np.r_[centroids, sample_new_p(data, p, l)]
        
    ## reduce k*l to k using KMeans++ 
    weights = get_weight_p(data,centroids)
    
    return centroids[np.random.choice(len(weights), k, replace= False, p = weights),:]