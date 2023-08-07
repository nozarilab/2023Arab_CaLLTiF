# General
import numpy as np
import time
import sys
import os
import datetime
import pandas as pd

from statsmodels.tsa.api import VAR
from scipy.stats import f
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/home/fahimeh/Volitional_Control_Neurofeedback/Non-DecNef/causal_discovery_packages/causal_discovery_for_time_series_master")
from baselines.scripts_python.granger_mv2 import Granger

## parameters and variables
mvgc_alpha = [0.4, 0.41 , 0.42,0.43, 0.44, 0.45, 0.46,0.47, 0.48,0.49, 0.5]
# mvgc_lags = [1, 2, 3, 4, 5]
mvgc_lags = [3]

num_rep_macaque = 60
# num_rep_macaque = 20

# functions
def granger_mv2(data, sig_level=0.05, maxlag=5, verbose=False):
    g = Granger(data, p=maxlag)
    res_df = g.fit(alpha=sig_level)
    return res_df

## Load simulated data
path_to_Macaque_small_degree = "/home/fahimeh/Volitional_Control_Neurofeedback/Non-DecNef/codes/Causality_Analysis/Dataset/DataSets_Feedbacks/2_Macaque_Networks/SmallDegree/data_fslfilter_concat"
# path_to_Macaque_long_range = "/home/fahimeh/Volitional_Control_Neurofeedback/Non-DecNef/codes/Causality_Analysis/Dataset/DataSets_Feedbacks/2_Macaque_Networks/LongRange/data_fslfilter_concat"
# path_to_Macaque_full = "/home/fahimeh/Volitional_Control_Neurofeedback/Non-DecNef/codes/Causality_Analysis/Dataset/DataSets_Feedbacks/2_Macaque_Networks/Full/data_fslfilter_concat"


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

## run mvgc
mvgc_link_matrix_all_macaque = [[['' for t in range(0, len(mvgc_alpha))] for k in range(0,len(mvgc_lags))] for j in range(0,num_rep_macaque)]
mvgc_pval_matrix_all_macaque = [[['' for t in range(0, len(mvgc_alpha))] for k in range(0,len(mvgc_lags))] for j in range(0,num_rep_macaque)]
mvgc_macaque_execution_time = [[[0 for t in range(0, len(mvgc_alpha))] for k in range(0,len(mvgc_lags))] for j in range(0,num_rep_macaque)]

for r in range(0,num_rep_macaque):
    
    dataframe  = pd.DataFrame(np.array(macaque_all_data_small_degree[r]), columns = nodes_macaque_small_degree)

    for l in range(0, len(mvgc_lags)):
        for a in range(0, len(mvgc_alpha)):

                start = time.time()
                res = granger_mv2(dataframe, sig_level = mvgc_alpha[a], maxlag= mvgc_lags[l], verbose=False)

                end = time.time()
                print('repetition %d'%(r+1),'lags = %0.2f'%mvgc_lags[l],  'alpha = %0.2f'%mvgc_alpha[a], 'execution time: %0.3f s'%(end-start))
                
                mvgc_macaque_execution_time[r][l][a] = end-start
                mvgc_link_matrix_all_macaque[r][l][a] = res
                # np.savez('results_all_method_mecaque/MVGC_macaque_full_sweep_both_parameters_all_reps',mvgc_link_matrix_all_macaque = mvgc_link_matrix_all_macaque, mvgc_macaque_execution_time = mvgc_macaque_execution_time)

