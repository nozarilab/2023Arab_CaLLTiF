
# Author: ّFahimeh Arab <farab002@ucr.edu>
# License: GNU General Public License v3.0

import sys
import os
import numpy as np

code_folder = os.getcwd()
files_folder = os.path.join(code_folder, '..', 'External Packages and Files')
results_folder = os.path.join(code_folder, '..', 'Results')
data_folder = os.path.join(code_folder, '..', 'Data')

# # Tigramite Package
sys.path.insert(0, os.path.join(files_folder,'tigramite-master'))
from tigramite.independence_tests import ParCorr
import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI


def run_calltif_oracle(dataframe, cond_ind_test, selected_links=None,
                  tau_min=0,
                  tau_max=1,
                  save_iterations=False,
                  pc_alpha=0.05,
                  max_conds_dim=None,
                  max_combinations=1,
                  max_conds_py=None,
                  max_conds_px=None,
                  alpha_level=0.05,
                  fdr_method='none',
                  ground_truth_parent_set = None):

    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)

    r""" Parameters
    ----------
    selected_links : dict or None
        Dictionary of form {0: [(3, -2), ...], 1:[], ...}
        specifying whether only selected links should be tested. If None is
        passed, all links are tested.
    tau_min : int, optional (default: 0)
        Minimum time lag to test. Note that zero-lags are undirected.
    tau_max : int, optional (default: 1)
        Maximum time lag. Must be larger or equal to tau_min.
    save_iterations : bool, optional (default: False)
        Whether to save iteration step results such as conditions used.
    pc_alpha : float, optional (default: 0.05)
        Significance level in algorithm.
    max_conds_dim : int, optional (default: None)
        Maximum number of conditions to test. If None is passed, this number
        is unrestricted.
    max_combinations : int, optional (default: 1)
        Maximum number of combinations of conditions of current cardinality
        to test. Defaults to 1 for PC_1 algorithm. For original PC algorithm
        a larger number, such as 10, can be used.
    max_conds_py : int, optional (default: None)
        Maximum number of conditions of Y to use. If None is passed, this
        number is unrestricted.
    max_conds_px : int, optional (default: None)
        Maximum number of conditions of Z to use. If None is passed, this
        number is unrestricted.
    alpha_level : float, optional (default: 0.05)
        Significance level at which the p_matrix is thresholded to 
        get graph.
    fdr_method : str, optional (default: 'fdr_bh')
        Correction method, currently implemented is Benjamini-Hochberg
        False Discovery Rate method. 

    Returns
    -------
    graph : array of shape [N, N, tau_max+1]
        Causal graph, see description above for interpretation.
    val_matrix : array of shape [N, N, tau_max+1]
        Estimated matrix of test statistic values.
    p_matrix : array of shape [N, N, tau_max+1]
        Estimated matrix of p-values, optionally adjusted if fdr_method is
        not 'none'.
    conf_matrix : array of shape [N, N, tau_max+1,2]
        Estimated matrix of confidence intervals of test statistic values.
        Only computed if set in cond_ind_test, where also the percentiles
        are set.

    """
# Oracle CaLLTiF starts from a complete lagged graph as the parent set
    # the following line needs to be changed based on the ground truth graph you have
            
    all_parents = dict()

    for j in range(0, pcmci.N):
        current_parents = []
        for i in range(0,pcmci.N):
            if ground_truth_parent_set[i,j] == 1:
                for k in range(1,tau_max+1):
                    current_parents.append((i,-k))

        all_parents[j] = current_parents

# Get the results from run_mci, using the parents as the input    
    results = pcmci.run_mci(selected_links=selected_links,
                            tau_min=tau_min,
                            tau_max=tau_max,
                            parents=all_parents,
                            max_conds_py=max_conds_py,
                            max_conds_px=max_conds_px,
                            alpha_level=alpha_level,
                            fdr_method=fdr_method)
    
    # Store the parents in the pcmci member
    pcmci.all_parents = all_parents
    
    return results
