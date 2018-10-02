# k-means-parallel-HPC
Clustering is the method of identifying similar group of data in a data set and for now it’s optimization a central problem in data management. K-means is one of the widely used data processing algorithms; in fact, it was recognized as one of the top 10 algorithms in data mining. The advantage of k-means is its simplicity: starting with a set of randomly chosen initial centres, one repeatedly assigns each input point to its nearest center, and then recomputes the centres given the point assignment. However, the initialization in the algorithm is crucial for obtaining a good solution. K-means++ caters to this, by obtaining an initial set of centres that is close to the optimum solution. But k-means++ is a sequential algorithm and is inefficient over large data-set, k-passes need to be made over the entire data to find a set of good initial centres. It is important to reduce the number of passes made to get good initialization. The algorithm implemented, k-means|| obtains optimal solution after a logarithmic number of passes. It is also possible to bring down the number of passes to a constant number.
