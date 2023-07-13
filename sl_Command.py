
import numpy as np
from numpy import array
import spotlight, metrics
import argparse


def run_sl_SBM(fname, use_rrcf=True):
    outliers, sl_time, sl_scores = spotlight.run_SPOTLIGHT(fname, 
                  K=50, window = 5, 
                  percent_ranked= 0.045, 
                  use_rrcf=use_rrcf, 
                  seed=0)
    real_events=[16,31,61,76,91,106,136] 
    accu = metrics.compute_accuracy(outliers, real_events)
    print ("spotlight accuracy is ", accu)


def run_sl_real(dataset, use_rrcf=True):
    if (dataset == "USLegis"):
        outliers, sl_time, sl_scores = spotlight.run_SPOTLIGHT(dataset, 
                  K=50, window = 5, 
                  percent_ranked= 0.20, 
                  use_rrcf=use_rrcf, 
                  seed=0,
                  dataname="USLegis")
        real_events=[3,7] 
        accu = metrics.compute_accuracy(outliers, real_events)
        print ("detected anomalies are ", outliers)
        print ("spotlight accuracy is ", accu)

    if (dataset == "UCI"):
        outliers, sl_time, sl_scores = spotlight.run_SPOTLIGHT(dataset, 
                  K=50, window = 5, 
                  percent_ranked= 0.05, 
                  use_rrcf=use_rrcf, 
                  seed=0,
                  dataname="UCI")
        real_events=[65,157]
        accu = metrics.compute_accuracy(outliers, real_events)
        print ("detected anomalies are ", outliers)
        print ("spotlight accuracy is ", accu)

    if (dataset == "canVote"):
        outliers, sl_time, sl_scores = spotlight.run_SPOTLIGHT(dataset, 
                  K=50, window = 5, 
                  percent_ranked= 0.154, 
                  use_rrcf=use_rrcf, 
                  seed=0,
                  dataname="canVote")
        real_events=[7,9]
        accu = metrics.compute_accuracy(outliers, real_events)
        print ("detected anomalies are ", outliers)
        print ("spotlight accuracy is ", accu)


def main():
    parser = argparse.ArgumentParser(description='run LAD on synthetic experiments')
    parser.add_argument('-f','--file', 
                    help='decide which synthetic edgelist to run on', required=True)
    parser.add_argument('-d','--dataset', 
                    help='SBM; USLegis; UCI; canVote', required=True)
    parser.add_argument('--rrcf', dest='rrcf', action='store_true', help="To use rrcf")
    parser.set_defaults(rrcf=True)

    args = vars(parser.parse_args())
    if (args["dataset"] == "SBM"):
        run_sl_SBM(args["file"], use_rrcf=args["rrcf"])
    else:
        run_sl_real(args["dataset"], use_rrcf=args["rrcf"])


if __name__ == "__main__":
    main()
