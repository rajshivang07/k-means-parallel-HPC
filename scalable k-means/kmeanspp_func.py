import numpy as np
from distance_func import distance

def cost(dist):
    """ Calculate the cost of data with respect to the current centroids
    Parameters:
       dist     distance matrix between data and current centroids
    
    Returns:    the normalized constant in the distribution 
    """
    return np.sum(np.min(dist,axis=1))

def distribution(dist,cost):
    """ Calculate the distribution to sample new centers
    Parameters:
       dist       distance matrix between data and current centroids
       cost       the cost of data with respect to the current centroids
    Returns:      distribution 
    """
    return np.min(dist, axis=1)/cost

def sample_new(data,distribution,l):
    """ Sample new centers
    
    Parameters:
       data         n*d
       distribution n*1
       l            the number of new centers to sample
    Returns:        new centers                          
    """
    return data[np.random.choice(range(len(distribution)),l,p=distribution),:]

def KMeansPlusPlus(data, k):    
    """ Apply the KMeans++ clustering algorithm to get the initial centroids   
    Parameters: 
      data                        ndarrays data 
      k                           number of cluster
    
    Returns:
      "Centroids"                 the complete initial centroids by KMeans++
      
    """
    
    #Initialize the first centroid
    centroids = data[np.random.choice(data.shape[0],1),:]
    
    while centroids.shape[0] < k :
                
        #Get the distance between data and centroids
        dist = distance(data, centroids)
        
        #Calculate the cost of data with respect to the centroids
        norm_const = cost(dist)
        
        #Calculate the distribution for sampling a new center
        p = distribution(dist,norm_const)
        
        #Sample the new center and append it to the original ones
        centroids = np.r_[centroids, sample_new(data,p,1)]
    
    return centroids