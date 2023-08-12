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


code_folder = os.getcwd()
files_folder = os.path.join(code_folder, '..', 'External Packages and Files')
results_folder = os.path.join(code_folder, '..', 'Results')
data_folder = os.path.join(code_folder, '..', 'Data')
path_to_causal_discovery_for_time_series = os.path.join(files_folder,'causal_discovery_for_time_series_master')

sys.path.insert(0, path_to_causal_discovery_for_time_series)

from baselines.scripts_python.varlingam import varlingam
from baselines.scripts_python.python_packages.lingam_master.lingam.var_lingam import VARLiNGAM

# Parameters and variables
alpha = [0.005, 0.01, 0.05, 0.1, 0.15, 0.3, 0.5, 0.7, 0.9,1]

tau_max = 2
tau_min = 0

num_simple_networks = 9
num_rep_simple_net = 60

# functions
def varlingam(data, tau_max=1, alpha=0.05):
    min_causal_effect = alpha
    split_by_causal_effect_sign = True

    model = VARLiNGAM(lags=tau_max, criterion='bic', prune=True)
    model.fit(data)

    m = model._adjacency_matrices
    am = np.concatenate([*m], axis=1)

    dag = np.abs(am) > min_causal_effect

    if split_by_causal_effect_sign:
        direction = np.array(np.where(dag))
        signs = np.zeros_like(dag).astype('int64')
        for i, j in direction.T:
            signs[i][j] = np.sign(am[i][j]).astype('int64')
        dag = signs

    dag = np.abs(dag)
    names = data.columns
    res_dict = dict()
    for e in range(dag.shape[0]):
        res_dict[names[e]] = []
    for c in range(dag.shape[0]):
        for te in range(dag.shape[1]):
            if dag[c][te] == 1:
                e = te%dag.shape[0]
                t = te//dag.shape[0]
                res_dict[names[e]].append((names[c], -t))
    return res_dict

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
    
# load data
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


# run varlingam
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

varlingam_link_matrix_all_simple_net = [[['' for t in range(0, len(alpha))] for j in range(0,num_rep_simple_net)] for i in range(0,num_simple_networks )]
varlingam_simple_net_execution_time = [[['' for t in range(0, len(alpha))] for j in range(0,num_rep_simple_net)] for i in range(0,num_simple_networks )]

for n in range(0,num_simple_networks):
       
       current_simple_net = simple_net_all_data[n]
       nodes_simple_net = all_net_nodes_dict[str(n+1)]

       for i in range(0,num_rep_simple_net):


              dataframe  = pd.DataFrame(current_simple_net[i].to_numpy(), columns = nodes_simple_net)
              for a in range(0,len(alpha)):

                     start = time.time()
                     res = varlingam(dataframe, tau_max = tau_max, alpha = alpha[a])
                     end = time.time()
                     print('network %d'% (n+1), 'repetition %d'%(i+1), 'alpha = %0.3f'%alpha[a], 'execution time: %0.3f s'%(end-start) )

                     varlingam_simple_net_execution_time[n][i][a] = end-start
                     varlingam_link_matrix_all_simple_net[n][i][a] = make_time_lag_graph_from_dict(res,tau_max)

            # np.savez(os.path.join(results_folder,'Varlingam_all_simple_nets'),varlingam_link_matrix_all_simple_net = varlingam_link_matrix_all_simple_net, varlingam_simple_net_execution_time = varlingam_simple_net_execution_time)