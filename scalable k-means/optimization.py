%%cython -a
import numpy as np
cimport numpy as np

def distance_cy(data, centroids):
    """ Calculate the distance from each data point to each center
    Parameters:
       data   n*d
       center k*d
    
    Returns:
       distence n*k 
    """
    ## calculate distence between each point to the centroids
    dist = np.sum((data[:, np.newaxis, :] - centroids)**2, axis=2)
    return dist

def KMeans_cy(data, k, centroids, max_iter = 10000): 
    
    """ Apply the KMeans clustering algorithm
    
    Parameters:
      data                        ndarrays data 
      k                           number of cluster
      centroids                   initial centroids
    
    Returns:
      "Iteration before Coverge"  time used to converge
      "Centroids"                 the final centroids finded by KMeans    
      "Labels"                    the cluster of each data   
    """
    
    n = data.shape[0] 
    iterations = 0
    
    while iterations < max_iter:        
        dist = distance_cy(data,centroids)
        
        ## give cluster label to each point 
        cluster_label = np.argmin(dist, axis=1)
        
        ## calculate new centroids
        newCentroids = np.zeros(centroids.shape)
        for j in range(0, k):
            if sum(cluster_label == j) == 0:
                newCentroids[j] = centroids[j]
            else:
                newCentroids[j] = np.mean(data[cluster_label == j, :], axis=0)
        
        ## Check if it is converged
        if np.array_equal(centroids, newCentroids):
            print("Converge")
            break 
            
        centroids = newCentroids
        iterations += 1
        
    return({"Iteration before Coverge": iterations, 
            "Centroids": centroids, 
            "Labels": cluster_label})

def cost_cy(dist):
    """ Calculate the cost of data with respect to the current centroids
    Parameters:
       dist     distance matrix between data and current centroids
    
    Returns:    the normalized constant in the distribution 
    """
    return np.sum(np.min(dist,axis=1))

def distribution_cy(dist,cost):
    """ Calculate the distribution to sample new centers
    Parameters:
       dist       distance matrix between data and current centroids
       cost       the cost of data with respect to the current centroids
    Returns:      distribution 
    """
    return np.min(dist, axis=1)/cost

def sample_new_cy(data,distribution,l):
    """ Sample new centers
    
    Parameters:
       data         n*d
       distribution n*1
       l            the number of new centers to sample
    Returns:        new centers                          
    """
    return data[np.random.choice(range(len(distribution)),l,p=distribution),:]

def KMeansPlusPlus_cy(data, k):    
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
        dist = distance_cy(data, centroids)
        
        #Calculate the cost of data with respect to the centroids
        norm_const = cost_cy(dist)
        
        #Calculate the distribution for sampling a new center
        p = distribution_cy(dist,norm_const)
        
        #Sample the new center and append it to the original ones
        centroids = np.r_[centroids, sample_new_cy(data,p,1)]
    
    return centroids

def get_weight_cy(dist,centroids):
    min_dist = np.zeros(dist.shape)
    min_dist[range(dist.shape[0]), np.argmin(dist, axis=1)] = 1
    count = np.array([np.count_nonzero(min_dist[:, i]) for i in range(centroids.shape[0])])
    return count/np.sum(count)


def ScalableKMeansPlusPlus_cy(data, k, l,iter=5):
    
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
        dist = distance_cy(data, centroids)
        
        #Calculate the cost of data with respect to the centroids
        norm_const = cost_cy(dist)
        
        #Calculate the distribution for sampling l new centers
        p = distribution_cy(dist,norm_const)
        
        #Sample the l new centers and append them to the original ones
        centroids = np.r_[centroids, sample_new_cy(data,p,l)]
    

    ## reduce k*l to k using KMeans++ 
    dist = distance_cy(data, centroids)
    weights = get_weight_cy(dist,centroids)
    
    return centroids[np.random.choice(len(weights), k, replace= False, p = weights),:]