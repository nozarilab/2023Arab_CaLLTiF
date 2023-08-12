# General

import numpy as np
import time
import sys
import os
import datetime
import pandas as pd

code_folder = os.getcwd()
files_folder = os.path.join(code_folder, '..', 'External Packages and Files')
results_folder = os.path.join(code_folder, '..', 'Results')
data_folder = os.path.join(code_folder, '..', 'Data')
path_to_causal_discovery_for_time_series = os.path.join(files_folder,'causal_discovery_for_time_series_master')

sys.path.insert(0, path_to_causal_discovery_for_time_series)

from baselines.scripts_python.varlingam import varlingam
from baselines.scripts_python.python_packages.lingam_master.lingam.var_lingam import VARLiNGAM


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


## parameters and variables
alpha =[0.000001, 0.000005, 0.00001, 0.00005,0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04,0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

tau_max = 3
tau_min = 0

num_rep_macaque_all = 60

## Load simulated data
# path_to_Macaque_small_degree  = os.path.join(data_folder,"DataSets_Feedbacks/2_Macaque_Networks/SmallDegree/data_fslfilter_concat") 
# path_to_Macaque_long_range = os.path.join(data_folder,"DataSets_Feedbacks/2_Macaque_Networks/LongRange/data_fslfilter_concat") 
path_to_Macaque_full = os.path.join(data_folder,"DataSets_Feedbacks/2_Macaque_Networks/Full/data_fslfilter_concat") 

# num_nodes_macaque_small_degree = 28
# num_nodes_macaque_long_range = 67
num_nodes_macaque_full = 91

# nodes_macaque_small_degree = ['X%d'%i for i in range(1,num_nodes_macaque_small_degree+1)]
# nodes_macaque_long_range = ['X%d'%i for i in range(1,num_nodes_macaque_long_range+1)]
nodes_macaque_full = ['X%d'%i for i in range(1,num_nodes_macaque_full+1)]


# Lets load data 
macaque_all_data_full = [0]*num_rep_macaque_all

for i in range(1,num_rep_macaque_all+1):
    if i <= 9:
        current_str = "0"+str(i)
    elif 10 <= i:
        current_str = str(i)

    df_temp  = pd.read_csv(path_to_Macaque_full+"/concat_BOLDfslfilter_"+current_str+".txt", delimiter = '\t')
    macaque_all_data_full[i-1] = df_temp.to_numpy()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

varlingam_link_matrix_macaque = [['' for t in range(0, len(alpha))] for j in range(0,num_rep_macaque_all)]
varlingam_macaque_execution_time =[[0 for t in range(0, len(alpha))] for j in range(0,num_rep_macaque_all)]

for r in range(0,num_rep_macaque_all):

    dataframe  = pd.DataFrame(macaque_all_data_full[r], columns = nodes_macaque_full)
    for a in range(0, len(alpha)):

        start = time.time()
        res = varlingam(dataframe, tau_max = tau_max, alpha = alpha[a])
        end = time.time()
        print('repetition %d'%(r+1), 'alpha = %0.3f'%alpha[a], 'execution time: %0.3f s'%(end-start) )

        varlingam_macaque_execution_time [r][a] = end-start
        varlingam_link_matrix_macaque [r][a] = make_time_lag_graph_from_dict(res,tau_max)
        # np.savez('os.path.join(results_folder,'Varlingam_macaque_full_sweep_alpha_lag_3'),varlingam_link_matrix_macaque = varlingam_link_matrix_macaque, varlingam_macaque_execution_time = varlingam_macaque_execution_time)

