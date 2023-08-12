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

# # Tigramite Package
sys.path.insert(0, os.path.join(files_folder,'tigramite-master'))
from tigramite.independence_tests import ParCorr
import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI

## parameters and variables
alpha_level_pcmci = [1e-20,1e-18,1e-16,1e-14,1e-12,1e-10,1e-8,1e-6,1e-4,1e-2 ]
pc_alpha = 1

tau_max = 2
tau_min = 0

parcorr = ParCorr(significance='analytic')

num_simple_networks = 9
num_rep_simple_net = 60

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

## run pcmci
pcmci_link_matrix_all_simple_net = [[['' for t in range(0, len(alpha_level_pcmci))] for j in range(0,num_rep_simple_net)] for i in range(0,num_simple_networks )]
pcmci_pval_matrix_all_simple_net = [[['' for t in range(0, len(alpha_level_pcmci))] for j in range(0,num_rep_simple_net)] for i in range(0,num_simple_networks )]
pcmci_simple_net_execution_time = [[['' for t in range(0, len(alpha_level_pcmci))] for j in range(0,num_rep_simple_net)] for i in range(0,num_simple_networks )]

for n in range(0, num_simple_networks):

       current_simple_net = simple_net_all_data[n]
       nodes_simple_net = all_net_nodes_dict[str(n+1)]

       for r in range(0,num_rep_simple_net):
              
              dataframe  = pp.DataFrame(current_simple_net[r].to_numpy(), var_names = nodes_simple_net)
              pcmci = PCMCI(dataframe = dataframe, cond_ind_test = parcorr)

              for a in range(0, len(alpha_level_pcmci)):

                     start = time.time()
                     link_matrix = pcmci.run_pcmci(tau_min = tau_min, tau_max = tau_max, pc_alpha = pc_alpha, alpha_level = alpha_level_pcmci[a])
                     end = time.time()
                     
                     print('network %d'% (n+1), 'repetition %d'%(r+1), 'alpha level = %0.3f'%alpha_level_pcmci[a], 'execution time: %0.3f s'%(end-start) )
                     
                     pcmci_simple_net_execution_time[n][r][a] = end-start
                     pcmci_link_matrix_all_simple_net[n][r][a] = link_matrix['graph']
                     pcmci_pval_matrix_all_simple_net[n][r][a] = link_matrix['p_matrix']

                    # np.savez(os.path.join(results_folder,'PCMCI_all_simple_nets'),pcmci_link_matrix_all_simple_net = pcmci_link_matrix_all_simple_net, pcmci_simple_net_execution_time = pcmci_simple_net_execution_time, pcmci_pval_matrix_all_simple_net = pcmci_pval_matrix_all_simple_net)