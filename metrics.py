import sys
sys.path.append('../')

import numpy as np
import networkx as nx
import sparse
import math
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from util import normal_util
import pylab as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'

from scipy.stats import spearmanr
from numpy import linalg as LA

'''
compute the difference between predicted anomalies to the nearest grountruth
'''
def compute_difference_score(anomalies, real_events):

    '''
    precision and recall
    precision =  TP / (TP + FP)
    Recall = TP / (TP + FN)
    geometric mean = root2(x1*x2)
    normalize for percentage of anomalies
    '''


    #1. for each groundtruth, find the nearest prediction
    diff_true = 0
    for event in real_events:
        diff = [abs(event - x) for x in anomalies]
        distance = min(diff)
        diff_true = diff_true + distance

    #2. for each prediction, find the nearest groundtruth
    diff_pred = 0
    for anomaly in anomalies:
        diff = [abs(anomaly - x) for x in real_events]
        distance = min(diff)
        diff_pred = diff_pred + distance

    geo_mean = math.sqrt(diff_true*diff_pred)
    print (diff_pred)
    print (diff_true)
    return geo_mean


def compute_accuracy(anomalies, real_events):
    correct = 0
    for anomaly in anomalies:
        if anomaly in real_events:
            correct = correct + 1
    return ( correct/len(real_events) )


def generate_partition(anomalies, real_events, maxt):
    #partition the dataset based on the anomalies & real events, assume everything is a change point
    pred_pts = sorted(anomalies)
    true_pts = sorted(real_events)

    pred_par = []
    true_par = []
    
    p_idx = 0
    t_idx = 0
    for i in range(0, maxt):
        if (p_idx < len(pred_pts) and pred_pts[p_idx] == i):
            p_idx += 1
        pred_par.append(p_idx)

        if (t_idx < len(true_pts) and true_pts[t_idx] == i):
            t_idx += 1
        true_par.append(t_idx)

    # print (pred_par)
    # print (true_par)
    return (true_par, pred_par)






def compute_NMI(anomalies, real_events, maxt):
    true_par, pred_par = generate_partition(anomalies, real_events, maxt)
    nmi = normalized_mutual_info_score(true_par, pred_par, average_method='arithmetic')
    return nmi






def compute_ARI(anomalies, real_events, maxt):
    true_par, pred_par = generate_partition(anomalies, real_events, maxt)
    ari = adjusted_rand_score(true_par, pred_par)
    return ari


