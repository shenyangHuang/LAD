## LAD
Official python implementation of the paper:

Laplacian Change Point Detection for Dynamic Graphs (KDD 2020)

## Content:

 
## Usage:

1. first extract the edgelists in datasets/SBM_processed/hybrid, pure, resampled.zip

2. To reproduce synthetic experiments  (-n is the number of eigenvalues used) 

### python SBM_Command.py -f pure -n 499

substitute pure with hybrid or resampled for the corresponding settings

3. To reproduce real world experiments

### python Real_Command -d USLegis -n 6


## Library: 

1. python 3.8.1

2. scipy  1.4.1

3. scikit-learn 0.22.1

4. tensorly 0.4.5

5. networkx 2.4

6. matplotlib 1.3.1

## Reproducing Experiments:


## Citation:


