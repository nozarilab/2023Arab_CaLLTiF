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
alpha_level = 0.1
pc_alpha = 1

tau_max = 3
tau_min = 0

NSR_all = [0.0, 0.001, 0.01]

num_rep_macaque = 60

# num_nodes_macaque_small_degree = 28
# num_nodes_macaque_long_range = 67
num_nodes_macaque_full = 91

# nodes_macaque_small_degree = ['X%d'%i for i in range(1,num_nodes_macaque_small_degree+1)]
# nodes_macaque_long_range = ['X%d'%i for i in range(1,num_nodes_macaque_long_range+1)]
nodes_macaque_full = ['X%d'%i for i in range(1,num_nodes_macaque_full+1)]

# functions
def import_npz(npz_file):
    Data = np.load(npz_file, allow_pickle= True)
    for varName in Data:
        globals()[varName] = Data[varName]  

# Lets load data 
import_npz('macaque_full_deconvolved_data.npz')

calltif_link_matrix_all_macaque_deconv = [['' for t in range(0, len(NSR_all))] for k in range(0,num_rep_macaque)]
calltif_pval_matrix_all_macaque_deconv = [['' for t in range(0, len(NSR_all))] for k in range(0,num_rep_macaque)]
calltif_macaque_execution_time_deconv = [[0 for t in range(0, len(NSR_all))] for k in range(0,num_rep_macaque)]

parcorr = ParCorr(significance='analytic')

for r in range(0,num_rep_macaque):

    for sn in range(0, len(NSR_all)):

            dataframe  = pp.DataFrame(np.array(data_concat_deconv_all[r][sn]), var_names = nodes_macaque_full)

            start = time.time()
            link_matrix = calltif.run_calltif(dataframe,parcorr, tau_min= tau_min, tau_max = tau_max,pc_alpha = pc_alpha, alpha_level = alpha_level)
            end = time.time()
            print('repetition %d'%(r+1),'NSR = %0.3f'%NSR_all[sn] , 'execution time: %0.3f s'%(end-start) )
            
            calltif_macaque_execution_time_deconv[r][sn] = end-start
            calltif_link_matrix_all_macaque_deconv[r][sn] = link_matrix['graph']
            calltif_pval_matrix_all_macaque_deconv[r][sn] = link_matrix['p_matrix']
            # np.savez('results_all_method_mecaque/CaLLTiF_macaque_full_deconvolved_lag_3',calltif_link_matrix_all_macaque_deconv = calltif_link_matrix_all_macaque_deconv, calltif_macaque_execution_time_deconv = calltif_macaque_execution_time_deconv, calltif_pval_matrix_all_macaque_deconv = calltif_pval_matrix_all_macaque_deconv)
