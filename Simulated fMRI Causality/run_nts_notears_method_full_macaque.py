import csv
import math
import os
import warnings
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn import preprocessing
import sys

code_folder = os.getcwd()
files_folder = os.path.join(code_folder, '..', 'External Packages and Files')
results_folder = os.path.join(code_folder, '..', 'Results')
data_folder = os.path.join(code_folder, '..', 'Data')
path_to_nts_notears = os.path.join(files_folder, 'NTS-NOTEARS-main/notears')
sys.path.insert(0,path_to_nts_notears)

from locally_connected import LocallyConnected
from lbfgsb_scipy import LBFGSBScipy
from trace_expm import trace_expm
from utils import *
import utils as ut
import time
from  nts_notears import NTS_NOTEARS
from nts_notears import train_NTS_NOTEARS


# num_rep_macaque = 60
num_rep_macaque = 20

## Load simulated data
# path_to_Macaque_small_degree  = os.path.join(data_folder,"DataSets_Feedbacks/2_Macaque_Networks/SmallDegree/data_fslfilter_concat") 
# path_to_Macaque_long_range = os.path.join(data_folder,"DataSets_Feedbacks/2_Macaque_Networks/LongRange/data_fslfilter_concat") 
path_to_Macaque_full = os.path.join(data_folder,"DataSets_Feedbacks/2_Macaque_Networks/Full/data_fslfilter_concat") 

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
    
# set parameters
prior_knowledge = None
number_of_lags = 3

torch.set_default_dtype(torch.double)
np.set_printoptions(precision=3)

sequence_length = macaque_all_data_full[0].shape[0]
d = macaque_all_data_full[0].shape[1]
sem_type = 'AdditiveIndexModel'
n, s0, graph_type = sequence_length, d, 'ER'

variable_names_no_time = ['X{}'.format(j) for j in range(1, d + 1)]
variable_names = make_variable_names_with_time_steps(number_of_lags, variable_names_no_time)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nts_notears_link_matrix_all_macaque = [np.zeros((d,d,number_of_lags+1)) for j in range(0,num_rep_macaque)]
nts_notears_macaque_execution_time = [0 for j in range(0,num_rep_macaque)]
nts_notears_macaque_h = [0 for j in range(0,num_rep_macaque)]

for r in range(0, num_rep_macaque):

    X = macaque_all_data_full[r]

    scaler = preprocessing.StandardScaler().fit(X)
    normalized_X = scaler.transform(X)

    assert (normalized_X.std(axis=0).round(decimals=3) == 1).all()  # make sure all the variances are (very close to) 1

    start_time = time.time()
    model = NTS_NOTEARS(dims=[d, 10, 1], bias=True, number_of_lags=number_of_lags,
                        prior_knowledge=prior_knowledge, variable_names_no_time=variable_names_no_time)

    W_est_full,h = train_NTS_NOTEARS(model, normalized_X, device=device, lambda1= 0.005, lambda2=0.01,
                                    w_threshold=0, h_tol=1e-8, verbose=1)

    stop_time = time.time()

    # binary_estimated = W_est_full != 0
    total_d = d * (number_of_lags + 1)

    print('repetition %d'%(r+1), 'execution time: %0.3f s'%(stop_time-start_time) )

    nts_notears_macaque_execution_time[r] = stop_time-start_time
    nts_notears_link_matrix_all_macaque[r] = W_est_full
    nts_notears_macaque_h[r] = h

    # np.savez(os.path.join(results_folder,'NTS_notears_macaque_full'),nts_notears_link_matrix_all_macaque = nts_notears_link_matrix_all_macaque, nts_notears_macaque_execution_time = nts_notears_macaque_execution_time, nts_notears_macaque_h = nts_notears_macaque_h)