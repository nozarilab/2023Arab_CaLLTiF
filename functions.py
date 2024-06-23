
import numpy as np
import datetime
import networkx as nx
import matplotlib.pyplot as plt


def compute_cohens_d(x,y):

    n1 = x.shape[0]
    n2 = y.shape[0]

    s1 = np.std(x, ddof = 1)
    s2 = np.std(y, ddof = 1)
    s = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
    cohens_d = (np.mean(x)-np.mean(y))/s
    return cohens_d

# functions needed for causal discovery analysis

# This function plot a directional graph from a binary matrix
def plot_graph(matrix, nodes, title):

    node_color = 'lightblue'
    node_size = 2500
    font_size = 28
    edge_width = 3

    G = nx.from_numpy_matrix(matrix, create_using = nx.MultiDiGraph())  

    pos=nx.circular_layout(G) 

    nx.draw_networkx(G, node_size = node_size, node_color=node_color, font_size = font_size, pos = pos, width = edge_width)

    # labels = nx.draw_networkx_labels(G, pos= pos)
    plt.title(title, fontsize = 40)


def is_strongly_connected(graph):

    n = graph.shape[0]
    S = graph
    k = 1

    Full = np.all((S != 0))
    
    S_binarized = np.zeros_like(S)
    while (k <= n):

        S = S + np.linalg.matrix_power(graph, k+1)
        k = k + 1

        Full = np.all((S != 0))

    disconnected = np.where (S == 0)
    
    S_binarized[S != 0] = 1

    return Full, disconnected, S_binarized,S

def import_npz(npz_file):
    Data = np.load(npz_file, allow_pickle= True)
    print(Data)
    for varName in Data:
        globals()[varName] = Data[varName]  

        

def network_map(graph,net_labels,network_dic,parcel_labels, num_cortical_parcels):


    num_networks = len(net_labels)
    num_parcels = len(parcel_labels)

    network_parcel_idx= eval(network_dic)

    for i in range (0 , num_parcels):

        if i < num_cortical_parcels:
            temp = parcel_labels[i]
            temp_first = temp[0:6]
            if temp_first in network_parcel_idx:
                network_parcel_idx[temp_first].append(i)
        elif i >= num_cortical_parcels  and i<num_cortical_parcels +8:
            network_parcel_idx['RH_Sub'].append(i)
        
        else:
            network_parcel_idx['LH_Sub'].append(i)

    all_network_normalized_num_edges_from_to = np.zeros((num_networks,num_networks))

    for i in range(0, num_networks):
        for j in range(0,num_networks):

            from_net = net_labels[i]
            to_net = net_labels[j]

            from_net_parcel_idx =  np.array(network_parcel_idx[from_net])
            to_net_parcel_idx =  np.array(network_parcel_idx[to_net])

            if len(to_net_parcel_idx) != 0 and len(from_net_parcel_idx) != 0:

                temp = graph[:,to_net_parcel_idx]
                temp2 = temp[from_net_parcel_idx,:]

                if from_net == to_net:
                    np.fill_diagonal(temp2, 0)

                num_edges_from_to = np.sum(temp2)
                all_network_normalized_num_edges_from_to[i,j] = num_edges_from_to/(len(from_net_parcel_idx)*len(to_net_parcel_idx))
            
    return all_network_normalized_num_edges_from_to


def network_map_v2(graph,net_labels,network_dic,parcel_labels, num_cortical_parcels):


    num_networks = len(net_labels)
    num_parcels = len(parcel_labels)

    network_parcel_idx= eval(network_dic)

    for i in range (0 , num_parcels):

        if i < num_cortical_parcels:
            temp = parcel_labels[i]
            temp_first = temp[3:6]
            if temp_first in network_parcel_idx:
                network_parcel_idx[temp_first].append(i)

            if i>=100:
                network_parcel_idx['Sub'].append(i)  
                

    all_network_normalized_num_edges_from_to = np.zeros((num_networks,num_networks))

    for i in range(0, num_networks):
        for j in range(0,num_networks):

            from_net = net_labels[i]
            to_net = net_labels[j]

            from_net_parcel_idx =  np.array(network_parcel_idx[from_net])
            to_net_parcel_idx =  np.array(network_parcel_idx[to_net])

            if len(to_net_parcel_idx) != 0 and len(from_net_parcel_idx) != 0:

                temp = graph[:,to_net_parcel_idx]
                temp2 = temp[from_net_parcel_idx,:]

                if from_net == to_net:
                    np.fill_diagonal(temp2, 0)

                num_edges_from_to = np.sum(temp2)
                all_network_normalized_num_edges_from_to[i,j] = num_edges_from_to/(len(from_net_parcel_idx)*len(to_net_parcel_idx))
            
    return all_network_normalized_num_edges_from_to
        

def convert_to_string_graph(graph_bool):
    
        graph = np.zeros(graph_bool.shape, dtype='<U3')
        graph[:] = ""
        graph[:,:,1:][graph_bool[:,:,1:]==1] = "-->"
        graph[:,:,0][np.logical_and(graph_bool[:,:,0]==1, 
                                    graph_bool[:,:,0].T==1)] = "o-o"
        graph[:,:,0][np.logical_and(graph_bool[:,:,0]==2, 
                                    graph_bool[:,:,0].T==2)] = "x-x"
        for (i,j) in zip(*np.where(
            np.logical_and(graph_bool[:,:,0]==1, graph_bool[:,:,0].T==0))):
            graph[i,j,0] = "-->"
            graph[j,i,0] = "<--"

        return graph

# this function reads the output txt file from causal-cmd software 
def monthToNum(shortMonth):
    return {
            'January': 1,
            'February': 2,
            'March': 3,
            'April': 4,
            'May': 5,
            'June': 6,
            'July': 7,
            'August': 8,
            'September': 9, 
            'October': 10,
            'November': 11,
            'December': 12
    }[shortMonth]
    
def tetrad_graph_parser(filename, ground_truth = 0):

    k = 0
    with open(filename) as f:
        for line in f:
            if line == "Graph Nodes:\n":
                nodes_line = k + 1
            if line == "Graph Edges:\n":
                edges_line = k + 1
            if line[0:5] == "Start":
                start_time_line = k
            if line[0:3] == "End":
                end_time_line = k
            k += 1 
    with open(filename) as f:
        all_lines = f.readlines()

    if ground_truth == 1:
        graph_nodes = all_lines[nodes_line].replace("\n","").split(",")
    else:
        graph_nodes = all_lines[nodes_line].replace("\n","").split(";")
        
    graph_edges = all_lines[edges_line:]

    num_nodes = len(graph_nodes)
    num_edges = len(graph_edges)


# calculate the execution time 
    execution_time = 0

    if ground_truth == 0:

        start_temp = all_lines[start_time_line][13:].replace("\n","").split(",")
        end_temp = all_lines[end_time_line][12:].replace("\n","").split(",")

        start_date = start_temp[1].split(" ")
        end_date = end_temp[1].split(" ")

        start_month = monthToNum(start_date[1])
        end_month = monthToNum(end_date[1])

        start_day = int(start_date[2])
        end_day = int(end_date[2])

        start_year = int(start_temp[2].split(" ")[1])
        end_year = int(end_temp[2].split(" ")[1])

        start_time = start_temp[2].split(" ")[2].split(":")
        end_time = end_temp[2].split(" ")[2].split(":")
        
        start_am_pm = start_temp[2].split(" ")[3]
        end_am_pm = end_temp[2].split(" ")[3]


        start_hour = int(start_time[0])
        start_min = int(start_time[1])
        start_second = int(start_time[2])

        end_hour = int(end_time[0])
        end_min = int(end_time[1])
        end_second = int(end_time[2])


        if (start_am_pm == 'PM') and (start_hour != 12):
            start_hour = start_hour + 12

        if (end_am_pm == 'PM') and (end_hour != 12):
            end_hour = end_hour + 12
        
        if (start_am_pm == 'AM') and (start_hour == 12):
            start_hour = 0

        if (end_am_pm == 'AM') and (end_hour == 12):
            end_hour = 0


        dt1 = datetime.datetime(start_year,start_month,start_day,start_hour,start_min,start_second) 
        dt2 = datetime.datetime(end_year,end_month,end_day,end_hour,end_min,end_second) 
        tdelta = dt2 - dt1 
        
        execution_time = tdelta.total_seconds()

        print(dt1, dt2,execution_time)


        # execution_time = end_total_sec - start_total_sec 
        
    graph_matrix = np.zeros((num_nodes,num_nodes))

    for e in range(0, num_edges):

        current_edge = graph_edges[e].replace(str(e+1)+".","").replace("\n","").replace(" ","").split("-->")
        i = int(current_edge[0].replace('X',''))-1
        j = int(current_edge[1].replace('X',''))-1

        graph_matrix[i,j] = 1

    return graph_nodes, graph_edges, graph_matrix, execution_time

# This function creats a time lag graph from the dictionary
def make_time_lag_graph_from_dict(graph_dict, tau_max):
    dict_vars = graph_dict.keys()
    num_vars = len(dict_vars)

    time_lag_graph = [[['' for t in range(0, tau_max+1)] for j in range(0,num_vars)] for i in range(0, num_vars)]

    for v1 in range(0,num_vars):
        var_name = 'X%d'%(v1+1)
        var_time_lags = graph_dict[var_name]

        for v2 in range(0,num_vars):
            for t in range(0,tau_max+1):
                if ('X%d'%(v2+1),-t) in var_time_lags:
                    time_lag_graph[v2][v1][t] = '-->'
                else:
                    time_lag_graph[v2][v1][t] = ''

    return time_lag_graph


# These set of functiona summarize a time lag graph to a summary graph by combining the edges accross the lags
# used for pcmci
def summarize_across_nonzero_lags(PAG):

        PAG_non_zero_lags = PAG[:,:,1:]
        m = PAG_non_zero_lags.shape[0]
        num_lags = PAG_non_zero_lags.shape[2]

        PAG_non_zero_lags_binary = np.zeros((m,m, num_lags))
        PAG_non_zero_lags_binary[PAG_non_zero_lags == '-->'] = 1

        binary_matrix = np.sum(PAG_non_zero_lags_binary, axis = 2)
        binary_matrix[binary_matrix >= 1] = 1

        return binary_matrix

def summarize_across_all_lags_v1(time_lag_graph):

    num_vars = time_lag_graph.shape[0]
    num_lags = time_lag_graph.shape[2]

    summary_graph = [['' for t in range(0, num_vars)] for i in range(0, num_vars)]

    adjacency_binary_matrix = np.zeros((num_vars, num_vars))
    orientation_binary_matrix = np.zeros((num_vars, num_vars))

    for i in range(0,num_vars):
        for j in range(0,num_vars):

            edges_all_lags  = time_lag_graph[i,j,:]

            if '-->' in edges_all_lags:
                summary_graph[i][j] = '-->'

                adjacency_binary_matrix[i,j] = 1
                orientation_binary_matrix[i,j] = 1

            elif ('-->' not in edges_all_lags and 'o-o' in edges_all_lags):
                summary_graph[i][j] = 'o-o'
                
                adjacency_binary_matrix[i,j] = 1
                orientation_binary_matrix[i,j] = 0

            else:
                summary_graph[i][j] = ''
                
                adjacency_binary_matrix[i,j] = 0
                orientation_binary_matrix[i,j] = 0

    
    return summary_graph, adjacency_binary_matrix, orientation_binary_matrix


def summarize_across_all_lags_v2(time_lag_graph):

    num_vars = time_lag_graph.shape[0]
    num_lags = time_lag_graph.shape[2]

    summary_graph = [['' for t in range(0, num_vars)] for i in range(0, num_vars)]

    adjacency_binary_matrix = np.zeros((num_vars, num_vars))
    orientation_binary_matrix = np.zeros((num_vars, num_vars))

    for i in range(0,num_vars):
        for j in range(0,num_vars):

            edges_all_lags  = time_lag_graph[i,j,:]

            if '-->' in edges_all_lags:
                summary_graph[i][j] = '-->'

                adjacency_binary_matrix[i,j] = 1
                orientation_binary_matrix[i,j] = 1

            elif ('-->' not in edges_all_lags and 'o-o' in edges_all_lags):
                summary_graph[i][j] = 'o-o'
                
                adjacency_binary_matrix[i,j] = 1
                orientation_binary_matrix[i,j] = 1

            else:
                summary_graph[i][j] = ''
                
                adjacency_binary_matrix[i,j] = 0
                orientation_binary_matrix[i,j] = 0
    
    return summary_graph, adjacency_binary_matrix, orientation_binary_matrix

def summarize_across_all_lags_v2_with_pvalues(time_lag_graph, p_matrix):

    num_vars = time_lag_graph.shape[0]
    num_lags = time_lag_graph.shape[2]

    summary_graph = [['' for t in range(0, num_vars)] for i in range(0, num_vars)]

    binary_matrix = np.zeros((num_vars, num_vars))
    binary_graph_p_matrix = np.zeros((num_vars, num_vars))

    for i in range(0,num_vars):
        for j in range(0,num_vars):

            edges_all_lags  = time_lag_graph[i,j,:]

            if '-->' in edges_all_lags:
                summary_graph[i][j] = '-->'
                binary_matrix[i,j] = 1

                which_lags = np.where(edges_all_lags =='-->')[0]
                binary_graph_p_matrix[i,j] = np.min(p_matrix[i,j,which_lags])

            elif ('-->' not in edges_all_lags and 'o-o' in edges_all_lags):
                summary_graph[i][j] = 'o-o'
                binary_matrix[i,j] = 1

                binary_graph_p_matrix[i,j]= p_matrix[i,j,0]

            else:
                summary_graph[i][j] = ''
                binary_matrix[i,j] = 0
                binary_graph_p_matrix[i,j] = np.min(p_matrix[i,j,:])
    
    return summary_graph, binary_matrix, binary_graph_p_matrix


# The following function computes TPR,FPR, Recall, Precision, and F1 score for an estimated graph
def find_tpr_fpr_from_binary_matrix(true_graph_matrix, estimated_graph_matrix):

    adj_TP = 0
    adj_FN = 0
    adj_FP = 0
    adj_TN = 0

    n, m = np.shape(true_graph_matrix)

    for i in range(n):
        for j in range(i):

            if (true_graph_matrix[i,j] or  true_graph_matrix[j,i] ) == 1 and (estimated_graph_matrix[i,j] or  estimated_graph_matrix[j,i] ) == 1:
                adj_TP += 1
            elif (true_graph_matrix[i,j] or  true_graph_matrix[j,i] ) == 0 and (estimated_graph_matrix[i,j] or  estimated_graph_matrix[j,i] ) == 0:
                adj_TN += 1 
            elif (true_graph_matrix[i,j] or  true_graph_matrix[j,i] ) == 1 and (estimated_graph_matrix[i,j] or  estimated_graph_matrix[j,i] ) == 0:
                adj_FN += 1 
            elif (true_graph_matrix[i,j] or  true_graph_matrix[j,i] ) == 0 and (estimated_graph_matrix[i,j] or  estimated_graph_matrix[j,i] ) == 1:
                adj_FP += 1 

    if adj_FN == 0:
        adjacency_TPR = 1
    else:
        adjacency_TPR = adj_TP/(adj_TP + adj_FN)

    if adj_TN == 0:
        adjacency_FPR = 1
    else:
         adjacency_FPR = adj_FP/(adj_FP + adj_TN)

    if adj_FN == 0:
        adjacency_Recall = 1
    else:
        adjacency_Recall = adj_TP/(adj_TP + adj_FN)

    if adj_FP == 0:
        adjacency_Precision = 1
    else:
        adjacency_Precision = adj_TP/(adj_TP + adj_FP)

    if adjacency_Recall == 0 and adjacency_Precision == 0:
        adjacency_F1_score = 0
    else:
        adjacency_F1_score = 2*adjacency_Recall*adjacency_Precision/(adjacency_Precision + adjacency_Recall)


    ori_TP = 0
    ori_FN = 0
    ori_FP = 0
    ori_TN = 0

    n, m = np.shape(true_graph_matrix)

    for i in range(n):
        for j in range(m):

            if (true_graph_matrix[i,j]  == 1) and (estimated_graph_matrix[i,j] == 1):
                ori_TP += 1
            elif (true_graph_matrix[i,j]  == 0) and (estimated_graph_matrix[i,j] == 0):
                ori_TN += 1 
            elif (true_graph_matrix[i,j]  == 1) and (estimated_graph_matrix[i,j] == 0):
                ori_FN += 1 
            elif (true_graph_matrix[i,j]  == 0) and (estimated_graph_matrix[i,j] == 1):
                ori_FP += 1 

    if ori_FN == 0:
        orientation_TPR = 1
    else:
         orientation_TPR = ori_TP/(ori_TP + ori_FN)

    if ori_TN == 0:
        orientation_FPR = 1
    else:
        orientation_FPR = ori_FP/(ori_FP + ori_TN)

    if ori_FN == 0:
        orientation_Recall = 1
    else:
        orientation_Recall = ori_TP/(ori_TP + ori_FN)

    if ori_FP == 0:
        orientation_Precision = 1
    else:
        orientation_Precision = ori_TP/(ori_TP + ori_FP)

    if orientation_Recall == 0 and orientation_Precision == 0:
        orientation_F1_score = 0
    else:
        orientation_F1_score = 2*orientation_Recall*orientation_Precision/(orientation_Precision + orientation_Recall)

    return adjacency_TPR, adjacency_FPR, adjacency_Recall, adjacency_Precision, adjacency_F1_score, orientation_TPR, orientation_FPR, orientation_Recall,orientation_Precision,orientation_F1_score


def find_tpr_fpr_from_orientation_and_adjacency_matrices(true_graph_matrix, estimated_adjacency_binary_matrix, estimated_orientation_binary_matrix):

    adj_TP = 0
    adj_FN = 0
    adj_FP = 0
    adj_TN = 0

    n, m = np.shape(true_graph_matrix)

    for i in range(n):
        for j in range(i):

            if (true_graph_matrix[i,j] or  true_graph_matrix[j,i] ) == 1 and (estimated_adjacency_binary_matrix[i,j] or  estimated_adjacency_binary_matrix[j,i] ) == 1:
                adj_TP += 1
            elif (true_graph_matrix[i,j] or  true_graph_matrix[j,i] ) == 0 and (estimated_adjacency_binary_matrix[i,j] or  estimated_adjacency_binary_matrix[j,i] ) == 0:
                adj_TN += 1 
            elif (true_graph_matrix[i,j] or  true_graph_matrix[j,i] ) == 1 and (estimated_adjacency_binary_matrix[i,j] or  estimated_adjacency_binary_matrix[j,i] ) == 0:
                adj_FN += 1 
            elif (true_graph_matrix[i,j] or  true_graph_matrix[j,i] ) == 0 and (estimated_adjacency_binary_matrix[i,j] or  estimated_adjacency_binary_matrix[j,i] ) == 1:
                adj_FP += 1 

    if adj_FN == 0:
        adjacency_TPR = 1
    else:
        adjacency_TPR = adj_TP/(adj_TP + adj_FN)

    if adj_TN == 0:
        adjacency_FPR = 1
    else:
         adjacency_FPR = adj_FP/(adj_FP + adj_TN)

    if adj_FN == 0:
        adjacency_Recall = 1
    else:
        adjacency_Recall = adj_TP/(adj_TP + adj_FN)

    if adj_FP == 0:
        adjacency_Precision = 1
    else:
        adjacency_Precision = adj_TP/(adj_TP + adj_FP)

    if adjacency_Recall == 0 and adjacency_Precision == 0:
        adjacency_F1_score = 0
    else:
        adjacency_F1_score = 2*adjacency_Recall*adjacency_Precision/(adjacency_Precision + adjacency_Recall)


    ori_TP = 0
    ori_FN = 0
    ori_FP = 0
    ori_TN = 0

    n, m = np.shape(true_graph_matrix)

    for i in range(n):
        for j in range(m):

            if (true_graph_matrix[i,j]  == 1) and (estimated_orientation_binary_matrix[i,j] == 1):
                ori_TP += 1
            elif (true_graph_matrix[i,j]  == 0) and (estimated_orientation_binary_matrix[i,j] == 0):
                ori_TN += 1 
            elif (true_graph_matrix[i,j]  == 1) and (estimated_orientation_binary_matrix[i,j] == 0):
                ori_FN += 1 
            elif (true_graph_matrix[i,j]  == 0) and (estimated_orientation_binary_matrix[i,j] == 1):
                ori_FP += 1 

    if ori_FN == 0:
        orientation_TPR = 1
    else:
         orientation_TPR = ori_TP/(ori_TP + ori_FN)

    if ori_TN == 0:
        orientation_FPR = 1
    else:
        orientation_FPR = ori_FP/(ori_FP + ori_TN)

    if ori_FN == 0:
        orientation_Recall = 1
    else:
        orientation_Recall = ori_TP/(ori_TP + ori_FN)

    if ori_FP == 0:
        orientation_Precision = 1
    else:
        orientation_Precision = ori_TP/(ori_TP + ori_FP)

    if orientation_Recall == 0 and orientation_Precision == 0:
        orientation_F1_score = 0
    else:
        orientation_F1_score = 2*orientation_Recall*orientation_Precision/(orientation_Precision + orientation_Recall)

    return adjacency_TPR, adjacency_FPR, adjacency_Recall, adjacency_Precision, adjacency_F1_score, orientation_TPR, orientation_FPR, orientation_Recall,orientation_Precision,orientation_F1_score
