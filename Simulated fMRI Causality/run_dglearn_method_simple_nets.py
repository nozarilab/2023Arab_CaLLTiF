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

# Causal Inference
sys.path.insert(0, os.path.join(files_folder,'dglearn/dglearn-master'))
import dglearn as dg

# parameters and variables
bic_coef_all = [0.005, 0.01, 0.05, 0.1, 0.15, 0.3, 0.5, 0.7, 0.9,1]

num_simple_networks = 9
num_rep_simple_net = 60
tabu_length = 4
patience = 4

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

# run dglearn
dglearn_link_matrix_all_simple_net = [[['' for t in range(0, len(bic_coef_all))] for j in range(0,num_rep_simple_net)] for i in range(0,num_simple_networks )]
dglearn_simple_net_execution_time = [[['' for t in range(0, len(bic_coef_all))] for j in range(0,num_rep_simple_net)] for i in range(0,num_simple_networks )]

# learn structure using tabu search, plot learned structure

for n in range(0, num_simple_networks):

       current_simple_net = simple_net_all_data[n]
       nodes_simple_net = all_net_nodes_dict[str(n+1)]
       var_names = nodes_simple_net 
       n_vars = len(var_names)

       for r in range(0,num_rep_simple_net):

              X  = current_simple_net[r].to_numpy()
       
              for b in range(0, len(bic_coef_all)):
                     start = time.time()

                     manager = dg.CyclicManager(X, bic_coef = bic_coef_all[b])
                     learned_support, best_score, log = dg.tabu_search(manager, tabu_length, patience, first_ascent = False, verbose = 0)

                     # perform virtual edge correction
                     learned_support = dg.virtual_refine(manager, learned_support, patience = 0, max_path_len = 6, verbose = 0)

                     # remove any reducible edges
                     learned_support = dg.reduce_support(learned_support, fill_diagonal = False)

                     end = time.time()
                     print('network %d'% (n+1), 'repetition %d'%(r+1), 'bic_coef: %0.3f'%bic_coef_all[b], 'execution time: %0.3f s'%(end-start) )
                     
                     dglearn_simple_net_execution_time[n][r][b] = end-start
                     dglearn_link_matrix_all_simple_net[n][r][b] = learned_support

                    # np.savez(os.path.join(results_folder,'DGlearn_all_simple_nets'),dglearn_link_matrix_all_simple_net = dglearn_link_matrix_all_simple_net, dglearn_simple_net_execution_time = dglearn_simple_net_execution_time)