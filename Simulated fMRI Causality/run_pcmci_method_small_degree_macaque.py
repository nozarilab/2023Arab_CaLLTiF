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

# alpha_level_pcmci = [1e-20,1e-18,1e-16,1e-14,1e-12,1e-10,1e-8,1e-6,1e-4,1e-2]
# alpha_level_pcmci = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6,1e-5, 1e-4]
# alpha_level_pcmci = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
# pc_alpha = [1]

alpha_level_pcmci = 1e-5
pc_alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

tau_max = 3
tau_min = 0

num_rep_macaque = 60
# num_rep_macaque = 10

parcorr = ParCorr(significance='analytic')

## Load simulated data
path_to_Macaque_small_degree  = os.path.join(data_folder,"DataSets_Feedbacks/2_Macaque_Networks/SmallDegree/data_fslfilter_concat") 
# path_to_Macaque_long_range = os.path.join(data_folder,"DataSets_Feedbacks/2_Macaque_Networks/LongRange/data_fslfilter_concat") 
# path_to_Macaque_full = os.path.join(data_folder,"DataSets_Feedbacks/2_Macaque_Networks/Full/data_fslfilter_concat") 


num_nodes_macaque_small_degree = 28
# num_nodes_macaque_long_range = 67
# num_nodes_macaque_full = 91

nodes_macaque_small_degree = ['X%d'%i for i in range(1,num_nodes_macaque_small_degree+1)]
# nodes_macaque_long_range = ['X%d'%i for i in range(1,num_nodes_macaque_long_range+1)]
# nodes_macaque_full = ['X%d'%i for i in range(1,num_nodes_macaque_full+1)]


# Lets load data 

macaque_all_data_small_degree = [0]*num_rep_macaque

for i in range (1,num_rep_macaque + 1):
    if i <= 9:
        current_str = "0"+str(i)
    elif 10 <= i:
        current_str = str(i)

    df_temp  = pd.read_csv(path_to_Macaque_small_degree+"/concat_BOLDfslfilter_"+current_str+".txt", delimiter = '\t')
    macaque_all_data_small_degree[i-1] = df_temp.to_numpy()

## run pcmci
def run_pcmci_single(dataframe,parcorr, tau_min, tau_max,pc_alpha, alpha_level ):

    pcmci = PCMCI(dataframe = dataframe, cond_ind_test = parcorr)
    link_matrix = pcmci.run_pcmci(tau_min = tau_min, tau_max = tau_max, pc_alpha = pc_alpha, alpha_level = alpha_level)
                
    return link_matrix

pcmci_link_matrix_all_macaque = [['' for k in range(0,len(pc_alpha))] for j in range(0,num_rep_macaque)]
pcmci_pval_matrix_all_macaque = [['' for k in range(0,len(pc_alpha))] for j in range(0,num_rep_macaque)]
pcmci_macaque_execution_time = [[0 for k in range(0,len(pc_alpha))] for j in range(0,num_rep_macaque)]

for r in range(0,num_rep_macaque):
    
    dataframe  = pp.DataFrame(np.array(macaque_all_data_small_degree[r]), var_names = num_nodes_macaque_small_degree)
    for a1 in range(0, len(pc_alpha)):

                start = time.time()

                link_matrix = run_pcmci_single(dataframe,parcorr, tau_min, tau_max,pc_alpha[a1], alpha_level_pcmci)
                
                end = time.time()
                print('repetition %d'%(r+1),'pc alpha = %0.2f'%pc_alpha[a1], 'execution time: %0.3f s'%(end-start) )
                
                pcmci_macaque_execution_time[r][a1] = end-start
                pcmci_link_matrix_all_macaque[r][a1] = link_matrix['graph']
                pcmci_pval_matrix_all_macaque[r][a1] = link_matrix['p_matrix']
                # np.savez(os.path.join(results_folder,'PCMCI_macaque_small_degree_sweep_both_alpha_lag_3_try_more_values_of_pc_alpha'),pcmci_link_matrix_all_macaque = pcmci_link_matrix_all_macaque, pcmci_macaque_execution_time = pcmci_macaque_execution_time, pcmci_pval_matrix_all_macaque = pcmci_pval_matrix_all_macaque)



