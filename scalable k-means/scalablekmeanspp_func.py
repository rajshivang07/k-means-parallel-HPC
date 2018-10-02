import numpy as np
from distance_func import distance
from kmeanspp_func import cost, distribution, sample_new

def get_weight(dist,centroids):
    min_dist = np.zeros(dist.shape)
    min_dist[range(dist.shape[0]), np.argmin(dist, axis=1)] = 1
    count = np.array([np.count_nonzero(min_dist[:, i]) for i in range(centroids.shape[0])])
    return count/np.sum(count)

def ScalableKMeansPlusPlus(data, k, l,iter=5):
    
    """ Apply the KMeans|| clustering algorithm
    
    Parameters:
      data     ndarrays data 
      k        number of cluster
      l        number of point sampled in each iteration
    
    Returns:   the final centroids finded by KMeans||  
      
    """
    
    centroids = data[np.random.choice(range(data.shape[0]),1), :]
    
    
    for i in range(iter):
        #Get the distance between data and centroids
        dist = distance(data, centroids)
        
        #Calculate the cost of data with respect to the centroids
        norm_const = cost(dist)
        
        
        #Calculate the distribution for sampling l new centers
        p = distribution(dist,norm_const)
        
        #Sample the l new centers and append them to the original ones
        centroids = np.r_[centroids, sample_new(data,p,l)]
    

    ## reduce k*l to k using KMeans++ 
    dist = distance(data, centroids)
    weights = get_weight(dist, centroids)
    
    return centroids[np.random.choice(len(weights), k, replace= False, p = weights),:]