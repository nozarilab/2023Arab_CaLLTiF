# General
import numpy as np
import time
import sys
import os
import datetime
import pandas as pd

sys.path.insert(0, "/home/fahimeh/Volitional_Control_Neurofeedback/Non-DecNef/causal_discovery_packages/tigramite-master-new")

# # Tigramite Package
from tigramite.independence_tests import ParCorr
import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI

## parameters and variables

pc_alpha = [0.01, 0.2, 0.4, 0.6, 0.8 , 1]

tau_max = 2
tau_min = 0

parcorr = ParCorr(significance='analytic')

num_simple_networks = 9
num_rep_simple_net = 60


## Load simulated data
all_net_nodes_dict = {}
simple_net_all_data = [0]*num_simple_networks

for n in range(0,num_simple_networks):
    path_to_Simple_Net = "/home/fahimeh/Volitional_Control_Neurofeedback/Non-DecNef/codes/Causality_Analysis/Dataset/DataSets_Feedbacks/1_Simple_Networks/Network"+str(n+1)+"_amp/data_fslfilter_concat"

    if n == 7 or n == 8:
       path_to_Simple_Net = "/home/fahimeh/Volitional_Control_Neurofeedback/Non-DecNef/codes/Causality_Analysis/Dataset/DataSets_Feedbacks/1_Simple_Networks/Network"+str(n+1)+"_amp_amp/data_fslfilter_concat"

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


## run pcmciplus

pcmciplus_link_matrix_all_simple_net = [[[0 for t in range(0, len(pc_alpha))] for j in range(0,num_rep_simple_net)] for i in range(0,num_simple_networks )]
pcmciplus_pval_matrix_all_simple_net = [[[0 for t in range(0, len(pc_alpha))] for j in range(0,num_rep_simple_net)] for i in range(0,num_simple_networks )]
pcmciplus_simple_net_execution_time = [[[0 for t in range(0, len(pc_alpha))] for j in range(0,num_rep_simple_net)] for i in range(0,num_simple_networks )]

for n in range(0, num_simple_networks):

       current_simple_net = simple_net_all_data[n]
       nodes_simple_net = all_net_nodes_dict[str(n+1)]

       for r in range(0,num_rep_simple_net):
              
              dataframe  = pp.DataFrame(current_simple_net[r].to_numpy(), var_names = nodes_simple_net)
              pcmci = PCMCI(dataframe = dataframe, cond_ind_test = parcorr)

              for a in range(0, len(pc_alpha)):

                    start = time.time()
                    link_matrix = pcmci.run_pcmciplus(tau_min = tau_min, tau_max = tau_max, pc_alpha = pc_alpha[a])
                    end = time.time()
                     
                    print('network %d'% (n+1), 'repetition %d'%(r+1),'PC Alpha:%0.2f'%pc_alpha[a], 'execution time: %0.3f s'%(end-start) )
                     
                    pcmciplus_simple_net_execution_time[n][r][a] = end-start
                    pcmciplus_link_matrix_all_simple_net[n][r][a] = link_matrix['graph']
                    pcmciplus_pval_matrix_all_simple_net[n][r][a] = link_matrix['p_matrix']

                    # np.savez('PCMCIPlus_all_simple_nets',pcmciplus_link_matrix_all_simple_net = pcmciplus_link_matrix_all_simple_net, pcmciplus_simple_net_execution_time = pcmciplus_simple_net_execution_time, pcmciplus_pval_matrix_all_simple_net = pcmciplus_pval_matrix_all_simple_net)