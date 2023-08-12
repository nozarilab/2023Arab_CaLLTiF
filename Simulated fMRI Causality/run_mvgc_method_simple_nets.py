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

code_folder = os.getcwd()
files_folder = os.path.join(code_folder, '..', 'External Packages and Files')
results_folder = os.path.join(code_folder, '..', 'Results')
data_folder = os.path.join(code_folder, '..', 'Data')
path_to_causal_discovery_for_time_series = os.path.join(files_folder,'causal_discovery_for_time_series_master')

sys.path.insert(0, path_to_causal_discovery_for_time_series)
from baselines.scripts_python.granger_mv2 import Granger

## parameters and variables

mvgc_alpha = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
# mvgc_lags = [1, 2, 3, 4, 5]
mvgc_lags = [2]

num_simple_networks = 9
num_rep_simple_net = 60

# functions
def granger_mv2(data, sig_level=0.05, maxlag=5, verbose=False):
    g = Granger(data, p=maxlag)
    res_df = g.fit(alpha=sig_level)
    return res_df
    
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


mvgc_link_matrix_all_simple_net = [[['' for t in range(0, len(mvgc_alpha))] for j in range(0,num_rep_simple_net)] for i in range(0,num_simple_networks )]
mvgc_simple_net_execution_time = [[[0 for t in range(0, len(mvgc_alpha))] for j in range(0,num_rep_simple_net)] for i in range(0,num_simple_networks )]

for n in range(0,num_simple_networks):
    
    current_simple_net = simple_net_all_data[n]
    nodes_simple_net = all_net_nodes_dict[str(n+1)]

    for i in range(0,num_rep_simple_net):

        dataframe  = pd.DataFrame(current_simple_net[i].to_numpy(), columns = nodes_simple_net)

        for a in range(0,len(mvgc_alpha)):

            start = time.time()
            res = granger_mv2(dataframe, sig_level = mvgc_alpha[a], maxlag= mvgc_lags[0], verbose=False)
            end = time.time()
            print('network %d'% (n+1), 'repetition %d'%(i+1), 'alpha = %0.3f'%mvgc_alpha[a], 'execution time: %0.3f s'%(end-start) )

            mvgc_simple_net_execution_time[n][i][a] = end-start
            mvgc_link_matrix_all_simple_net[n][i][a] = res

            # np.savez(os.path.join(results_folder,'mvgc_all_simple_nets'),mvgc_link_matrix_all_simple_net = mvgc_link_matrix_all_simple_net, mvgc_simple_net_execution_time = mvgc_simple_net_execution_time)


    


                

