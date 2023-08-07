# General
import numpy as np
import time
import sys
import os
import datetime
import pandas as pd

sys.path.insert(0, "/home/fahimeh/Volitional_Control_Neurofeedback/Non-DecNef/causal_discovery_packages/causal_discovery_for_time_series_master")
sys.path.insert(0, "/home/fahimeh/Volitional_Control_Neurofeedback/Non-DecNef/causal_discovery_packages/tigramite-master-new")

# # Tigramite Package
from tigramite.independence_tests import ParCorr
import tigramite.data_processing as pp
import calltif

## parameters and variables

alpha_level= 1e-5
pc_alpha = 1

tau_max = 3
tau_min = 0

num_rep_macaque = 60
# num_rep_macaque = 10

parcorr = ParCorr(significance='analytic')

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


calltif_link_matrix_all_macaque = [''for j in range(0,num_rep_macaque)]
calltif_pval_matrix_all_macaque = ['' for j in range(0,num_rep_macaque)]
calltif_macaque_execution_time = ['' for j in range(0,num_rep_macaque)]

for r in range(0,num_rep_macaque):
    
    dataframe  = pp.DataFrame(np.array(macaque_all_data_full[r]), var_names = nodes_macaque_full)
    start = time.time()

    link_matrix = calltif.run_calltif(dataframe,parcorr, tau_min= tau_min, tau_max = tau_max,pc_alpha = pc_alpha, alpha_level = alpha_level)

    end = time.time()
    print('repetition %d'%(r+1), 'execution time: %0.3f s'%(end-start) )
    
    calltif_macaque_execution_time[r] = end-start
    calltif_link_matrix_all_macaque[r] = link_matrix['graph']
    calltif_pval_matrix_all_macaque[r] = link_matrix['p_matrix']
    # np.savez('results_all_method_mecaque/CaLLTiF_macaque_full_sweep_both_alpha_lag_3',calltif_link_matrix_all_macaque = calltif_link_matrix_all_macaque, calltif_macaque_execution_time = calltif_macaque_execution_time, calltif_pval_matrix_all_macaque = calltif_pval_matrix_all_macaque)



