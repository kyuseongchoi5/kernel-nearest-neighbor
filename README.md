# Distributional Matrix Completion via Kernels

## Problem and goal

Here we consider a distributional matrix completion problem: some entries $(i, t) \in [N] \times [T]$ in a matrix contains $n$ measurements $X_1(i, t), ..., X_n(i, t) \in \mathbb R^d$ while the remaining entries are missing values. The goal is to impute a collection of measurements into the entries where values were missing. 

## Algorithm 

Our imputation algorithm (Kernel-NN) takes the following $4$ inputs:
- Collection of observed measurements $X_1(i, t), ..., X_n(i, t)$ where entries $(i, t)$ are observed.
- Collection of binary values $0, 1$ 

, where the neighbors of the target entry is taken over the columns. Maximum mean discrepency for a kernel of choice is used in measuring distance between matrix rows. Neighbors within column are identified using the row distance and a fixed radius, and the multiple measurements within neighborhood are averaged over(MMD barycenter), which is simply the mixture of the empirical distributions(i.e. collection of all the observed measurements within neighborhood). Cross validation is implemented to optimize radius size.  

