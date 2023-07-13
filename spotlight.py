import sys
sys.path.append('../')
import numpy as np 
import random
import pylab as plt
import timeit
import rrcf
import pandas as pd
import metrics 
from datasets import SBM_loader, USLegis_loader, UCI_loader, canVote_loader


def set_non_negative(z_scores):
    for i in range(len(z_scores)):
        if (z_scores[i] < 0):
            z_scores[i] = 0
    return z_scores



'''
NOTE: SPOTLIGHT assumes node ordering persists over time
generate a source or a destination dictionary
G: the graph at snapshot 0
p: probability of sampling a node into the dictionary

return dict: a dictionary of selected sources or destinations for a subgraph
'''
def make_src_dict(G, p):
    out_dict = {}
    for node in G.nodes():
        if (random.random() <= p):
            out_dict[node] = 1
    return out_dict


'''
main algorithm for SPOTLIGHT
G_times: a list of networkx graphs for each snapshot in order
K: the number of subgraphs to track. 
p: probability of sampling a node into the source
q: probability of sampling a node into the destination

return a list of SPOTLIGHT embeddings (np arrays) for each snapshot
idx: which snapshot to select the src and origin nodes from
'''
def SPOTLIGHT(G_times, K, p, q, idx=0):

    '''
    initialize K spotlight sketches at step 0
    '''
    src_dicts = []
    dst_dicts = []
    for _ in range(K):
        src_dicts.append(make_src_dict(G_times[idx], p))
        dst_dicts.append(make_src_dict(G_times[idx], q))

    sl_embs = []
    for G in G_times:
        sl_emb = np.zeros(K, )
        for u,v,w in G.edges.data("weight", default=1):
            for i in range(len(src_dicts)):
                if (u in src_dicts[i] and v in dst_dicts[i]):
                    sl_emb[i] += w 
        sl_embs.append(sl_emb)

    return sl_embs






def rrcf_offline(X, num_trees=50, tree_size=50):
    n = len(X)
    X = np.asarray(X)
    sample_size_range = (n // tree_size, tree_size)

    # Construct forest
    forest = []
    while len(forest) < num_trees:
        # Select random subsets of points uniformly
        ixs = np.random.choice(n, size=sample_size_range,
                               replace=False)
        # Add sampled trees to forest
        trees = [rrcf.RCTree(X[ix], index_labels=ix)
                 for ix in ixs]
        forest.extend(trees)

    # Compute average CoDisp
    avg_codisp = pd.Series(0.0, index=np.arange(n))
    index = np.zeros(n)
    for tree in forest:
        codisp = pd.Series({leaf : tree.codisp(leaf)
                           for leaf in tree.leaves})
        avg_codisp[codisp.index] += codisp
        np.add.at(index, codisp.index.values, 1)
    avg_codisp /= index
    avg_codisp = avg_codisp.tolist()
    return avg_codisp


def run_SPOTLIGHT(fname, 
                  K=50, 
                  window = 5, 
                  percent_ranked= 0.05, 
                  use_rrcf=True, 
                  seed=0,
                  dataname="SBM"):
    random.seed(seed)
    p = 0.2
    q = 0.2

    if (dataname == "SBM"):
        edgefile = "datasets/SBM_processed/" + fname + ".txt"
        G_times = SBM_loader.load_temporarl_edgelist(edgefile)
    elif (dataname == "USLegis"):
        fname = "datasets/USLegis_processed/LegisEdgelist.txt"
        G_times = USLegis_loader.load_legis_temporarl_edgelist(fname)
    elif (dataname == "UCI"):
        fname = "datasets/UCI_processed/OCnodeslinks_chars.txt"
        max_nodes = 1901
        G_times = UCI_loader.load_temporarl_edgelist(fname, max_nodes=max_nodes)
    elif (dataname == "canVote"):
        fname = "datasets/canVote_processed/canVote_edgelist.txt"
        G_times = canVote_loader.load_canVote_temporarl_edgelist(fname)


    start = timeit.default_timer()
    sl_embs = SPOTLIGHT(G_times, K, p, q)
    end = timeit.default_timer()
    sl_time = end-start
    print ('SPOTLIGHT time: '+str(sl_time)+'\n')
    # normal_util.save_object(sl_embs, "spotlight" + str(K) + fname + ".pkl")


    if (use_rrcf):
        start = timeit.default_timer()
        num_trees = 50
        tree_size = 151
        if (dataname == "USLegis"):
            num_trees = 10
            tree_size = 10
        scores = rrcf_offline(sl_embs, num_trees=num_trees, tree_size=tree_size)
        end = timeit.default_timer()
        a_time = end-start
        print ('rrcf time: '+str(a_time)+'\n')
        scores = np.asarray(scores)
        num_ranked = int(scores.shape[0]*percent_ranked)
        outliers = scores.argsort()[-num_ranked:][::-1]
        outliers.sort()

    else:
        start = timeit.default_timer()
        scores, outliers = simple_detector(sl_embs)
        end = timeit.default_timer()
        a_time = end-start
        print ('sum predictor time: '+str(a_time)+'\n')
    '''
    ploting for RRCF
    '''
    # x = list(range(scores.shape[0]))
    # plt.plot(x, scores)
    # plt.xlabel("timestamps")
    # plt.ylabel("RRCF anomaly score")
    # for event in outliers:
    #     plt.annotate(str(event), # this is the text
    #              (event, scores[event]), # this is the point to label
    #              textcoords="offset points", # how to position the text
    #              xytext=(0,-12), # distance from text to points (x,y)
    #              ha='center') # horizontal alignment can be left, right or center
    #     plt.plot( event, scores[event], marker="*", color='#de2d26', ls='solid')
    # plt.savefig('spotlight_rrcf.pdf')
    # plt.close()
    return outliers, sl_time, scores



def find_anomalies(scores, percent_ranked, initial_window):
    scores = np.array(scores)
    for i in range(initial_window+1):
        scores[i] = 0        #up to initial window + 1 are not considered anomalies. +1 is because of difference score
    num_ranked = int(round(len(scores) * percent_ranked))
    outliers = scores.argsort()[-num_ranked:][::-1]
    outliers.sort()
    return outliers



def simple_detector(sl_embs, plot=False, ratio=0.05, initial_window=10):
    sums = [np.sum(sl) for sl in sl_embs]

    if (plot):
        x = list(range(len(sl_embs)))
        plt.plot(x, sums)
        plt.xlabel("timestamps")
        plt.ylabel("sum of spotlight embedding")
        plt.savefig('simple_detector.pdf')
        plt.close()

    diffs = [0]
    for i in range(1, len(sums)):
        diffs.append(sums[i]-sums[i-1])

    events = find_anomalies(diffs, ratio, initial_window)
    scores = diffs

    if (plot):
        plt.plot(x, diffs)
        plt.xlabel("timestamps")
        plt.ylabel("difference in sum")
        for event in events:
            plt.annotate(str(event), # this is the text
                     (event, scores[event]), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,-12), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center
            plt.plot( event, scores[event], marker="*", color='#de2d26', ls='solid')

        plt.savefig('simple_diff.pdf')
        plt.close()
    return scores, events




    
#run this for evolving SBM

if __name__ == '__main__':
    fname = "evolveSBM_0.005_0.03" #"SBM2000"
    edgefile = "../datasets/multi_SBM/" + fname + ".txt"
    use_rrcf = True
    K=50

    if (use_rrcf):
        print ("using robust random cut forest")
    else:
        print ("using simple sum predictor")

    real_events=[16,31,61,76,91,106,136] 
    accus = []
    sl_times = []
    runs = 5
    seeds = list(range(runs))
    for i in range(runs):
        anomalies, sl_time, scores = run_SPOTLIGHT(edgefile, fname, K=K, use_rrcf=use_rrcf, seed=seeds[i])
        accu = metrics.compute_accuracy(anomalies, real_events)
        accus.append(accu)
        sl_times.append(sl_time)

    accus = np.asarray(accus)
    sl_times = np.asarray(sl_times)
    print (" the mean accuracy is : ", np.mean(accus))
    print (" the std is : ",  np.std(accus))

    print (" the mean spotlight time is : ", np.mean(sl_times))
    print (" the std is : ",  np.std(sl_times))

    # fname = "dosNm10Nz100SBM1000"
    # rrcf_LAD(fname, window=5)






# '''
# https://klabum.github.io/rrcf/
# this is the online version, do not use
# sl_embs: the spotlight embedding for each snapshot
# window: size of the sliding window
# '''
# def rrcf_anomalies(sl_embs, window, num_trees=50):
#     # Set tree parameters
#     shingle_size = window
#     tree_size = sl_embs[0].shape[0]

#     # Create a forest of empty trees
#     forest = []
#     for _ in range(num_trees):
#         tree = rrcf.RCTree()
#         forest.append(tree)
        
#     # Use the "shingle" generator to create rolling window
#     points = rrcf.shingle(sl_embs, size=shingle_size)

#     # Create a dict to store anomaly score of each point
#     avg_codisp = {}

#     # For each shingle...
#     for index, point in enumerate(points):
#         # For each tree in the forest...
#         for tree in forest:
#             # If tree is above permitted size...
#             if len(tree.leaves) > tree_size:
#                 # Drop the oldest point (FIFO)
#                 tree.forget_point(index - tree_size)
#             # Insert the new point into the tree
#             tree.insert_point(point, index=index)
#             # Compute codisp on the new point...
#             new_codisp = tree.codisp(index)
#             # And take the average over all trees
#             if not index in avg_codisp:
#                 avg_codisp[index] = 0
#             avg_codisp[index] += new_codisp / num_trees

#     scores = list(avg_codisp.values())
#     return scores


# '''
# just for fun
# '''
# def rrcf_LAD(fname, window=5, percent_ranked= 0.05):
#     vecs = normal_util.load_object(fname+'.pkl')

#     vecs = np.asarray(vecs).real
#     vecs = vecs.reshape((151,-1))
#     vecs = normalize(vecs, norm='l2')

#     scores = rrcf_anomalies(vecs, window)
#     scores = np.asarray(scores)
#     num_ranked = int(scores.shape[0]*percent_ranked)
#     outliers = scores.argsort()[-num_ranked:][::-1]
#     outliers.sort()
#     print (outliers)

