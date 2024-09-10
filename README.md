# Distributional Matrix Completion via Kernels

## Problem 

Consider a matrix where multiple measurements are contained per matrix entry. Some matrix entries are missing, meaning multiple measurements are not available for such entry. Our aim is to use neighboring observed multiple measurements to learn the empirical distribution of the missing multiple measurements that would have been otherwise observed.

## Algorithm 

Our imputation strategy is a variant of nearest neighbors, where the neighbors of the target entry is taken over the columns. Maximum mean discrepency for a kernel of choice is used in measuring distance between matrix rows. Neighbors within column are identified using the row distance and a fixed radius, and the multiple measurements within neighborhood are averaged over(MMD barycenter), which is simply the mixture of the empirical distributions(i.e. collection of all the observed measurements within neighborhood). Cross validation is implemented to optimize radius size.  

