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
mvgc_alpha = 0.01
mvgc_lag = 3
num_subjects = 700
Num_parcels = 116
nodes = ['X%d'%i for i in range(1 , Num_parcels+1)] 


# functions
def import_npz(npz_file):
    Data = np.load(npz_file, allow_pickle= True)
    for varName in Data:
        globals()[varName] = Data[varName]  

# data
import_npz(os.path.join(results_folder,'HCP_Rest_700_Subjects.npz'))

# functions
def granger_mv2(data, sig_level=0.05, maxlag=5, verbose=False):
    g = Granger(data, p=maxlag)
    res_df = g.fit(alpha=sig_level)
    return res_df


mvgc_hcp_rest_link_matrix = []
mvgc_hcp_rest_execution_time = []

num_concat = 4
num_samples = fmri_data_rest_1_RL.shape[1]
num_nodes = Num_parcels

for s in range(0, num_subjects):
    
    # prepare datat
    data = np.concatenate((fmri_data_rest_1_RL[s,:,0:Num_parcels], fmri_data_rest_1_LR[s,:,0:Num_parcels], fmri_data_rest_2_RL[s,:,0:Num_parcels],fmri_data_rest_2_LR[s,:,0:Num_parcels]))
    dataframe  = pd.DataFrame(data, columns = nodes)

    start = time.time()
    res = granger_mv2(dataframe, sig_level = mvgc_alpha, maxlag= mvgc_lag, verbose=False)

    end = time.time()

    # save results
    mvgc_hcp_rest_link_matrix.append(res)
    mvgc_hcp_rest_execution_time.append (end - start)

    print('Subject %d'%(s+1), 'Execution time: %0.3f s'%(end-start))

    # np.savez(os.path.join(results_folder,'mvgc_hcp_rest'),mvgc_hcp_rest_link_matrix = mvgc_hcp_rest_link_matrix, mvgc_hcp_rest_execution_time = mvgc_hcp_rest_execution_time)




    
