import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import nibabel as nib
import os
import datetime
import time
import sys
import warnings
import glob

code_folder = os.getcwd()
files_folder = os.path.join(code_folder, '..', 'External Packages and Files')
results_folder = os.path.join(code_folder, '..', 'Results')
path_to_calltif = os.path.join(code_folder, '..')

# Causal Inference
sys.path.insert(0, os.path.join(files_folder,'tigramite-master'))
sys.path.insert(0, path_to_calltif)

# Tigramite Package
from tigramite.independence_tests import ParCorr
import tigramite.data_processing as pp
import calltif

# parameters
Num_parcels = 116
nodes = ['X%d'%i for i in range(1 , Num_parcels+1)] 

tau_min = 0
tau_max = 3

pc_alpha = 1
alpha_level = 1e-2/32

num_subjects = 700

parcorr = ParCorr()

# functions
def import_npz(npz_file):
    Data = np.load(npz_file, allow_pickle= True)
    for varName in Data:
        globals()[varName] = Data[varName]  

# data
import_npz(os.path.join(results_folder,'HCP_Rest_700_Subjects.npz'))

multipledataset_calltif_hcp_rest_link_matrix = []
multipledataset_calltif_hcp_rest_execution_time = []

num_concat = 4
num_samples = fmri_data_rest_1_RL.shape[1]
num_nodes = Num_parcels

for s in range(0, num_subjects):
    
    # prepare datat
    data = np.concatenate((fmri_data_rest_1_RL[s,:,0:Num_parcels], fmri_data_rest_1_LR[s,:,0:Num_parcels], fmri_data_rest_2_RL[s,:,0:Num_parcels],fmri_data_rest_2_LR[s,:,0:Num_parcels]))
    data = np.reshape(np.array(data), (num_concat, num_samples, num_nodes ))

    start = time.time()

    dataframe  = pp.DataFrame(data, var_names = nodes, analysis_mode = 'multiple')
    link_matrix = calltif.run_calltif(dataframe,parcorr, tau_min= tau_min, tau_max = tau_max,pc_alpha = pc_alpha, alpha_level = alpha_level)

    end = time.time()

    # save results
    multipledataset_calltif_hcp_rest_link_matrix.append(link_matrix)
    multipledataset_calltif_hcp_rest_execution_time.append (end - start)

    print('Subject %d'%(s+1), 'Execution time: %0.3f s'%(end-start))

    # np.savez(os.path.join(results_folder,'multipledataset_calltif_hcp_rest'),multipledataset_calltif_hcp_rest_link_matrix = multipledataset_calltif_hcp_rest_link_matrix, multipledataset_calltif_hcp_rest_execution_time = multipledataset_calltif_hcp_rest_execution_time)




    
