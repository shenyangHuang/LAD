from util import normal_util
import networkx as nx
import pylab as plt
from networkx.drawing.nx_agraph import graphviz_layout
import re



from datasets import UCI_loader
from datasets import DBLP_loader
from datasets import Nature_loader
from datasets import SBM_loader
from datasets import USLegis_loader
from datasets import canVote_loader



def plot_UCI():

    '''
    plot statistics from UCI Message
    '''
    fname = "datasets/UCI_processed/OCnodeslinks_chars.txt"
    max_nodes = 1901
    G_times = UCI_loader.load_temporarl_edgelist(fname, max_nodes=max_nodes)

    graph_name = "UCI_Message"
    '''
    dictionary of weak labels
    '''
    labels_dict = {}
    print ("edge")
    labels_dict['edge'] = normal_util.plot_edges(G_times, graph_name)
    print ("acc")
    labels_dict['acc'] = normal_util.plot_avg_clustering(G_times, graph_name)
    print ("component")
    labels_dict['component'] = normal_util.plot_num_components_directed(G_times, graph_name)
    print ("weights")
    labels_dict['weights'] = normal_util.plot_weighted_edges(G_times, graph_name)
    print ("degree")
    labels_dict['degree'] = normal_util.plot_degree_changes(G_times, graph_name)
    return labels_dict

def plot_UCI_allinOne():
    fname = "datasets/UCI_processed/OCnodeslinks_chars.txt"
    max_nodes = 1901
    G_times = UCI_loader.load_temporarl_edgelist(fname, max_nodes=max_nodes)

    LAD = [69,70,184,185,187,188,189,191,192,194]
    activity = [57,75,78,85,89,90,104,176,188,192]
    CPD = [13,16,17,20,22,23,24,31,38,40,124]
    label_sets = []
    label_sets.append(LAD)
    label_sets.append(activity)
    label_sets.append(CPD)

    graph_name = "UCI_Message"
    normal_util.all_in_one_compare(G_times, graph_name, label_sets, True)







    graph_name = "UCI_Message"
    normal_util.all_plots_in_one(G_times, graph_name)

def print_labels(labels_dict):
    for label in labels_dict:
        print (label)
        print (labels_dict[label])




def plot_DBLP():
    '''
    plot statistics from DBLP dataset
    '''
    fname = "datasets/DBLP_processed/DBLP_1000_edgelist.txt"
    max_nodes = 6905
    G_times = DBLP_loader.load_dblp_temporarl_edgelist(fname, max_nodes=max_nodes)

    graph_name = "DBLP"
    '''
    dictionary of weak labels
    '''
    labels_dict = {}
    print ("edge")
    labels_dict['edge'] = normal_util.plot_edges(G_times, graph_name)
    print ("acc")
    labels_dict['acc'] = normal_util.plot_avg_clustering(G_times, graph_name)
    print ("component")
    labels_dict['component'] = normal_util.plot_num_components_undirected(G_times, graph_name)
    print ("weights")
    labels_dict['weights'] = normal_util.plot_weighted_edges(G_times, graph_name)
    print ("degree")
    labels_dict['degree'] = normal_util.plot_degree_changes(G_times, graph_name)
    return labels_dict


def plot_Nature_category():

    '''
    plot statistics from Nature category dataset
    '''
    fname = "datasets/Nature_processed/Nature_category_edgelist.txt"
    max_nodes = 13
    G_times = Nature_loader.load_nature_category_edgelist(fname, max_nodes=max_nodes)
    graph_name = "Nature interdiscipline"
    normal_util.plot_edges(G_times, graph_name)
    normal_util.plot_weighted_edges(G_times, graph_name)


def plot_Nature():

    '''
    plot statistics from Nature dataset
    '''
    fname = "datasets/Nature_processed/Nature_edgelist.txt"
    max_nodes = 88282
    G_times = Nature_loader.load_nature_temporal_edgelist(fname)
    graph_name = "Nature"
    print ("edge")
    outliers = normal_util.plot_edges(G_times, graph_name)
    print (outliers)
    print ("degree")
    outliers = normal_util.plot_degree_changes(G_times, graph_name)
    print (outliers)
    
    #normal_util.plot_weighted_edges(G_times, graph_name)


def plot_DBLP_all_in_one():
    fname = "datasets/DBLP_processed/DBLP_1000_edgelist.txt"
    max_nodes = 6905
    G_times = DBLP_loader.load_dblp_temporarl_edgelist(fname, max_nodes=max_nodes)
    LAD = [6,12,13]
    activity = [13, 14, 16]
    CPD = [8, 9, 10]
    label_sets = []
    label_sets.append(LAD)
    label_sets.append(activity)
    label_sets.append(CPD)

    graph_name = "DBLP"
    normal_util.all_in_one_compare(G_times, graph_name, label_sets, False)


def plot_synthetic():
    fname = "datasets/SBM_processed/config_edgelist.txt"
    #fname = "datasets/SBM_processed/ER_synthetic_edgelist_sudden_0.002_0.3.txt"
    max_nodes = 100
    max_time = 150
    G_times = SBM_loader.load_temporarl_edgelist(fname, max_nodes=max_nodes, max_time=max_time)
    graph_name = "synthetic"
    outliers = normal_util.plot_edges(G_times, graph_name)
    normal_util.plot_num_components_undirected(G_times,  graph_name)
    print (outliers)


def plot_legislative_allinOne():
    fname = "datasets/USLegis_processed/LegisEdgelist.txt"
    G_times = USLegis_loader.load_legis_temporarl_edgelist(fname)
    LAD = [3,7]
    label_sets = []
    label_sets.append(LAD)

    graph_name = "USLegislative"
    normal_util.all_in_one_compare(G_times, graph_name, label_sets, False)


def plot_canVote_allinOne():
    fname = "datasets/canVote_processed/canVote_edgelist.txt"
    G_times = canVote_loader.load_canVote_temporarl_edgelist(fname)
    LAD = [2,7,11]
    label_sets = []
    label_sets.append(LAD)
    window = 1
    initial_window = 2
    percent_ranked = 0.2

    graph_name = "canVote"
    normal_util.all_in_one_compare(G_times, graph_name, label_sets, True, window, initial_window, percent_ranked)


def plot_spectrum(pkl_name, graph_name):
    eigen_slices = normal_util.load_object(pkl_name)
    normal_util.plot_activity_intensity(eigen_slices, graph_name)


def plot_vis(G):
    pos = nx.spring_layout(G)

    node_list = []
    for node in G:
        if (G.degree[node] > 50):
            node_list.append(node)
    print (len(node_list))

    print(node_list)

    colors = range(len(G))
    options = {
        "nodelist" : node_list,
        "node_size" : 5,
        "node_color": "#ffa600",
        "edge_color": "#ff6361",
        "width": 0.05
    }

    
    nx.draw(G, pos, **options)

    plt.axis("off")
    plt.savefig('graph_vis.pdf')


def plot_illus():
    G = nx.Graph()

    G.add_edges_from([(0, 1), (0, 2), (0,3), (0,4), (0,5)])
    node_list = list (G.nodes)
    pos = nx.spring_layout(G)

    options = {
        "nodelist" : node_list,
        "node_size" : 500,
        "node_color": "#ffa600",
        "edge_color": "#66E3D8",
        "width": 3
    }

    nx.draw(G, pos, **options)

    labels={}
    labels[0]='0'
    labels[1]='1'
    labels[2]='2'
    labels[3]='3'
    labels[4]='4'
    labels[5]='5'
    
    nx.draw_networkx_labels(G,pos,labels,font_size=16)

    plt.axis("off")
    plt.savefig('graph_illus.pdf')


def load_csv():
	MP_dict = {}
	fname = "party_politics.csv"
	file = open(fname, "r")
	file.readline()
	for line in file.readlines():
		line = line.strip("\n")
		values = line.split(",")
		u = values[-2]
		party = values[-1]
		MP_dict[u] = party
	return MP_dict







def export_gephi():

    G_times = canVote_loader.load_canVote_temporarl_edgelist("datasets/canVote_processed/canVote_edgelist.txt")
    MP_dict = load_csv()
    labels = list(range(2006,2020,1))

    parties = []
    for key in MP_dict.keys():
    	if MP_dict[key] not in parties:
    		parties.append(MP_dict[key])

    print (parties)


    for i in range(len(G_times)):
        #party for everyone!
        G = G_times[i]
        count = 0
        for node in G.nodes:
            if (node in MP_dict):
            	if (MP_dict[node] == 'Conservative'):
            		#blue
            		G.nodes[node]['viz'] = {'color': {'r': 49, 'g': 130, 'b': 189, 'a': 0}}
            	
            	if (MP_dict[node] == 'Liberal'):
            		#red
            		G.nodes[node]['viz'] = {'color': {'r': 227, 'g': 74, 'b': 51, 'a': 0}}

            	if (MP_dict[node] == 'Bloc'):
            		#purple
            		G.nodes[node]['viz'] = {'color': {'r': 136, 'g': 86, 'b': 167, 'a': 0}}

            	if (MP_dict[node] == 'NDP'):
            		#green 
            		G.nodes[node]['viz'] = {'color': {'r': 49, 'g': 163, 'b': 84, 'a': 0}}

            else:
            	#black is default color
            	G.nodes[node]['viz'] = {'color': {'r': 99, 'g': 99, 'b': 99, 'a': 0}}




  #       graph.node['red']['viz'] = {'color': {'r': 255, 'g': 0, 'b': 0, 'a': 0}}
		# graph.node['green']['viz'] = {'color': {'r': 0, 'g': 255, 'b': 0, 'a': 0}}
		# graph.node['blue']['viz'] = {'color': {'r': 0, 'g': 0, 'b': 255, 'a': 0}}
        # print (count)

        nx.write_gexf(G, "gephi/" + str(labels[i]) + ".gexf")











def main():
    export_gephi()
    #plot_illus()

    # fname = "datasets/canVote_processed/canVote_edgelist.txt"
    # G_times = canVote_loader.load_canVote_temporarl_edgelist(fname)
    # G_0 = G_times[0]
    # plot_vis(G_0)

    #plot_spectrum("USLegis_L_singular6.pkl", "USLegis")
    # plot_canVote_allinOne()


    #plot_DBLP_all_in_one()
    # fname = "datasets/UCI_processed/OCnodeslinks_chars.txt"
    # max_nodes = 1901
    # G_times = UCI_loader.load_temporarl_edgelist(fname, max_nodes=max_nodes)
    # normal_util.plot_compare_weak_labels_edge(G_times, "UCI message")



    # plot_Nature()
    # fname = "datasets/DBLP_processed/DBLP_1000_edgelist.txt"
    # max_nodes = 6905
    # G_times = DBLP_loader.load_dblp_temporarl_edgelist(fname, max_nodes=max_nodes)

    # graph_name = "DBLP"
    # normal_util.all_plots_in_one(G_times, graph_name)
    # labels_dict = plot_DBLP()
    # print_labels(labels_dict)



if __name__ == "__main__":
    main()
