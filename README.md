# Distributional Matrix Completion via Kernels

## Problem and goal

Here we consider a distributional matrix completion problem: some entries $(i, t) \in [N] \times [T]$ in a matrix contains $n$ measurements $X_1(i, t), ..., X_n(i, t) \in \mathbb R^d$ while the remaining entries are missing values. The goal is to impute a collection of measurements into the entries where values were missing. 

## Algorithm 

We employ our kernel nearest neighbors algorithm (Kernel-NN). It requires the following $4$ inputs:
- Collection of observed measurements: $X_1(i, t), ..., X_n(i, t)$ for entries $(i, t)$ that are observed.
- Collection of binary values $A_{i, t}$ for $(i, t)\in [N]\times[T]$ where $A_{i, t} = 1$ when entry $(i, t)$ is observed and $A_{i,t} = 0$ otherwise.
- The index of target entry that we want impute values with.
- Kernel

The (row-wise) Kernel-NN  processes the $4$ inputs through the two steps:
- Construction of the metric 

, where the neighbors of the target entry is taken over the columns. Maximum mean discrepency for a kernel of choice is used in measuring distance between matrix rows. Neighbors within column are identified using the row distance and a fixed radius, and the multiple measurements within neighborhood are averaged over(MMD barycenter), which is simply the mixture of the empirical distributions(i.e. collection of all the observed measurements within neighborhood). Cross validation is implemented to optimize radius size.  

