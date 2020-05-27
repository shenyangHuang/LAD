
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import compute_SVD, Anomaly_Detection
import argparse


def run_LAD_SBM(fname, num_eigen):
    compute_SVD.compute_synthetic_SVD(fname, num_eigen=num_eigen)
    Anomaly_Detection.synthetic(fname)


def main():
    parser = argparse.ArgumentParser(description='run LAD on synthetic experiments')
    parser.add_argument('-f','--file', 
                    help='decide which synthetic edgelist to run on', required=True)
    parser.add_argument("-n",'--num', type=int, default=499,
                    help="number of eigenvalues to compute")
    args = vars(parser.parse_args())
    run_LAD_SBM(args["file"], args["num"])

if __name__ == "__main__":
    main()
