# General
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import time
from collections import Counter
import sys
import os
import datetime
from causalnex.structure.dynotears import from_pandas_dynamic

code_folder = os.getcwd()
files_folder = os.path.join(code_folder, '..', 'External Packages and Files')
results_folder = os.path.join(code_folder, '..', 'Results')
data_folder = os.path.join(code_folder, '..', 'Data')

# parameters and variables
alpha_dynotears = [0.005, 0.01, 0.05, 0.1, 0.15, 0.3, 0.5, 0.7, 0.9,1]

tau_max = 2
tau_min = 0

num_simple_networks = 9
num_rep_simple_net = 60

# functions
def dynotears(data, tau_max=5, alpha=0.0):
    graph_dict = dict()
    for name in data.columns:
        graph_dict[name] = []

    sm = from_pandas_dynamic(data, p=tau_max, w_threshold = alpha, lambda_w=0.05, lambda_a=0.05)

    tname_to_name_dict = dict()
    count_lag = 0
    idx_name = 0
    for tname in sm.nodes:
        tname_to_name_dict[tname] = data.columns[idx_name]
        if count_lag == tau_max:
            idx_name = idx_name +1
            count_lag = -1
        count_lag = count_lag +1

    for ce in sm.edges:
        c = ce[0]
        e = ce[1]
        tc = int(c.partition("lag")[2])
        te = int(e.partition("lag")[2])
        t = tc - te
        if (tname_to_name_dict[c], -t) not in graph_dict[tname_to_name_dict[e]]:
            graph_dict[tname_to_name_dict[e]].append((tname_to_name_dict[c], -t))

    return graph_dict


# This function creats a time lag graph from the dictionary
def make_time_lag_graph_from_dict(graph_dict, tau_max):
    dict_vars = graph_dict.keys()
    num_vars = len(dict_vars)

    time_lag_graph = [[['' for t in range(0, tau_max+1)] for j in range(0,num_vars)] for i in range(0, num_vars)]

    for v1 in range(0,num_vars):
        var_name = 'X%d'%(v1+1)
        var_time_lags = graph_dict[var_name]

        for v2 in range(0,num_vars):
            for t in range(0,tau_max+1):
                if ('X%d'%(v2+1),-t) in var_time_lags:
                    time_lag_graph[v2][v1][t] = '-->'
                else:
                    time_lag_graph[v2][v1][t] = ''

    return time_lag_graph


all_net_nodes_dict = {}
simple_net_all_data = [0]*num_simple_networks

for n in range(0,num_simple_networks):
    path_to_Simple_Net = os.path.join(data_folder,"DataSets_Feedbacks/1_Simple_Networks/Network"+str(n+1)+"_amp/data_fslfilter_concat")

    if n == 7 or n == 8:
       path_to_Simple_Net = os.path.join(data_folder,"DataSets_Feedbacks/1_Simple_Networks/Network"+str(n+1)+"_amp_amp/data_fslfilter_concat")

    simple_net = [0]*num_rep_simple_net

    for i in range (1,num_rep_simple_net + 1):
        if i <= 9:
            current_str = "0"+str(i)
        elif 10 <= i:
            current_str = str(i)

        simple_net [i-1]  = pd.read_csv(path_to_Simple_Net+"/concat_BOLDfslfilter_"+current_str+".txt", delimiter = '\t')

    num_nodes_temp = simple_net [i-1].shape[1]
    nodes_simple_net = ['X%d'%i for i in range(1,num_nodes_temp+1)]
    all_net_nodes_dict[str(n+1)]= nodes_simple_net

    simple_net_all_data[n] = simple_net
    print('nodes in simple network %d :'%(n+1),nodes_simple_net)

# run dynotears
dynotears_link_matrix_all_simple_net = [[['' for t in range(0, len(alpha_dynotears))] for j in range(0,num_rep_simple_net)] for i in range(0,num_simple_networks )]
dynotears_simple_net_execution_time = [[['' for t in range(0, len(alpha_dynotears))] for j in range(0,num_rep_simple_net)] for i in range(0,num_simple_networks )]

for n in range(0,num_simple_networks):

       current_simple_net = simple_net_all_data[n]
       nodes_simple_net = all_net_nodes_dict[str(n+1)]

       for i in range(0,num_rep_simple_net):

              dataframe  = pd.DataFrame(current_simple_net[i].to_numpy(), columns = nodes_simple_net)

              for a in range(0,len(alpha_dynotears)):

                     start = time.time()
                     graph = dynotears(dataframe, tau_max=tau_max, alpha=alpha_dynotears[a])
                     end = time.time()

                     print('network %d'% (n+1), 'repetition %d'%(i+1), 'alpha = %0.6f'%alpha_dynotears[a], 'execution time: %0.3f s'%(end-start) )

                     dynotears_simple_net_execution_time[n][i][a] = end-start
                     dynotears_link_matrix_all_simple_net[n][i][a] = make_time_lag_graph_from_dict(graph,tau_max)

                    # np.savez(os.path.join(results_folder,'Dynotears_all_simple_nets'),dynotears_link_matrix_all_simple_net = dynotears_link_matrix_all_simple_net, dynotears_simple_net_execution_time = dynotears_simple_net_execution_time)