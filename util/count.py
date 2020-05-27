import numpy as np
import networkx as nx
from scipy.sparse.linalg import svds, eigs
from datasets import UCI_loader
from util import normal_util
from model import GRU_model



def main():
	a = [ 62, 70 ,74 ,89 ,101 ,108 ,136 ,157, 171, 191]
	a = a + [10, 13, 26, 31, 33, 40 ,60 ,77, 80, 89]
	a = a + [ 19, 62 ,70 ,74 , 89, 101, 136, 171, 185, 191]
	a = a + [ 18, 45 ,73 ,94 , 120, 136, 148, 169, 187, 189]
	a = a + [ 62, 70 ,74 , 89 ,101 ,108, 136, 157, 171, 191]

	print (len(set(a)))
	


if __name__ == "__main__":
    main()


