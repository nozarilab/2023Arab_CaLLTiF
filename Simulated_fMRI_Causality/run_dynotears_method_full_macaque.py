# General
import numpy as np
import time
import sys
import os
import datetime
import pandas as pd

sys.path.insert(0, "/home/fahimeh/Volitional_Control_Neurofeedback/Non-DecNef/causal_discovery_packages/causal_discovery_for_time_series_master")
from causalnex.structure.dynotears import from_pandas_dynamic

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


## parameters and variables

alpha_dynotears = [0.01, 0.02, 0.03, 0.04,0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

tau_max = 3
tau_min = 0

num_rep_macaque = 60
# num_rep_macaque = 10

## Load simulated data
# path_to_Macaque_small_degree = "/home/fahimeh/Volitional_Control_Neurofeedback/Non-DecNef/codes/Causality_Analysis/Dataset/DataSets_Feedbacks/2_Macaque_Networks/SmallDegree/data_fslfilter_concat"
# path_to_Macaque_long_range = "/home/fahimeh/Volitional_Control_Neurofeedback/Non-DecNef/codes/Causality_Analysis/Dataset/DataSets_Feedbacks/2_Macaque_Networks/LongRange/data_fslfilter_concat"
path_to_Macaque_full = "/home/fahimeh/Volitional_Control_Neurofeedback/Non-DecNef/codes/Causality_Analysis/Dataset/DataSets_Feedbacks/2_Macaque_Networks/Full/data_fslfilter_concat"

# num_nodes_macaque_small_degree = 28
# num_nodes_macaque_long_range = 67
num_nodes_macaque_full = 91

# nodes_macaque_small_degree = ['X%d'%i for i in range(1,num_nodes_macaque_small_degree+1)]
# nodes_macaque_long_range = ['X%d'%i for i in range(1,num_nodes_macaque_long_range+1)]
nodes_macaque_full = ['X%d'%i for i in range(1,num_nodes_macaque_full+1)]


# Lets load data 

macaque_all_data_full = [0]*num_rep_macaque

for i in range (1,num_rep_macaque + 1):
    if i <= 9:
        current_str = "0"+str(i)
    elif 10 <= i:
        current_str = str(i)

    df_temp  = pd.read_csv(path_to_Macaque_full+"/concat_BOLDfslfilter_"+current_str+".txt", delimiter = '\t')
    macaque_all_data_full[i-1] = df_temp.to_numpy()

# run dynotears
dynotears_link_matrix_all_macaque = [['' for t in range(0, len(alpha_dynotears))] for j in range(0,num_rep_macaque)]
dynotears_macaque_execution_time = [[0 for t in range(0, len(alpha_dynotears))] for j in range(0,num_rep_macaque)]

for r in range(0,num_rep_macaque):
    
    dataframe  = pd.DataFrame(macaque_all_data_full[r], columns = nodes_macaque_full)

    for a in range(0,len(alpha_dynotears)):

            start = time.time()
            graph = dynotears(dataframe, tau_max=tau_max, alpha=alpha_dynotears[a])
            end = time.time()

            print('repetition %d'%(r+1), 'alpha = %0.6f'%alpha_dynotears[a], 'execution time: %0.3f s'%(end-start) )
            
            dynotears_macaque_execution_time[r][a] = end-start
            dynotears_link_matrix_all_macaque[r][a] = make_time_lag_graph_from_dict(graph,tau_max)
            # np.savez('results_all_method_mecaque/Dynotears_macaque_full_sweep_alpha_lag_3',dynotears_link_matrix_all_macaque = dynotears_link_matrix_all_macaque, dynotears_macaque_execution_time = dynotears_macaque_execution_time)