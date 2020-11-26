"""
Created on Sun Sep 13 15:13:33 2020

@author: iseabrook1
"""

#Methods for assessment of performance of l_e approximation. 
#Specifically, methods to change a single edge, and then two edges, to observe how well the value of l_e
#captures the resultant change in lambda for that change in A. This is done for all the edges in the graph
#plot includes network to highlight how the value of l_e depends on network structure

#Isobel Seabrook, ucabeas@ucl.ac.uk
#MIT License. Please reference below publication if used for research purposes.
#Reference: Seabrook et al., Evaluating Structural Edge Importance in Financial Networks
#
################################################################################
#   Instructions for use.
#
#   Generate a graph dataframe and assign it to a free variable:
#
#   G = chosen_graph(n, timestamp)
#
#   Graph dataframe must be passed with variables n and timestamp:
#
#   n: Number of nodes in the clique (if barbell) or number of nodes in graph (if Erdos Renyi). 
#   Do not specify n for ring network.
#   timestamp: integer value of timestamp for graph snapshot
#
#   run chosen perturbation method:
#   
#   snapshot_perturb perturbs edges one at a time and compare the resultant relationship between perturbation and eigenvalue change to the value of l_e
#   snapshot_perturb_double works as snapshot_perturb but for two edges perturbed simultaneously.
#
#   perturbaiton functions must be initialised with a graph dataframe
#
###############################################################################

import networkx as nx
import pandas as pd
import random
import numpy as np

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from model_evaluation import structural_importance_model as sim
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
def static_bb_gen(n, timestamp):
    """ Function to produce an unweighted static barbell graph, with two nodes in the bridge and n nodes in the cliques. 
    
    Parameters:
        n: number of nodes to make up the clique
        timestamp: if generating a static graph as a time snapshot instance of a temporal graph, specify the timestamp for the snapshot.
        
    Returns:
        pandas edgelist dataframe. 
    """
    G = nx.barbell_graph(n, 2)
    
    for (u,v,w) in G.edges(data=True):
        w['weight'] = 1
    graph_data = nx.to_pandas_edgelist(G, source="seller id", target="buyer id")
    graph_data.columns = ["seller id","buyer id", "total_value"]
    graph_data["tuple_id"] = list(G.edges())
    graph_data['trade date time'] = timestamp
    return(graph_data)

def static_bb_gen_w(n, timestamp):
    """ Function to produce a static weighted barbell graph, with two nodes in the bridge and n nodes in the cliques. 
    
    Parameters:
        n: number of nodes to make up the clique
        timestamp: if generating a static graph as a time snapshot instance of a temporal graph, specify the timestamp for the snapshot.
        
    Returns:
        pandas edgelist dataframe. 
    """
   
    G = nx.barbell_graph(n, 2)
    
    for (u,v,w) in G.edges(data=True):
        w['weight'] = random.randint(0,10)
    graph_data = nx.to_pandas_edgelist(G, source="seller id", target="buyer id")
    graph_data.columns = ["seller id","buyer id", "total_value"]
    graph_data["tuple_id"] = list(G.edges())
    graph_data['trade date time'] = timestamp
    return(graph_data)
def static_er_gen(n, timestamp):
    """ Function to produce a static unwieghted erdos renyi graph, with n nodes and connection probability 0.5.  
    
    Parameters:
        n: number of nodes 
        timestamp: if generating a static graph as a time snapshot instance of a temporal graph, specify the timestamp for the snapshot.
        
    Returns:
        pandas edgelist dataframe. 
    """
   
    G = nx.erdos_renyi_graph(n, 0.5)
    
    for (u,v,w) in G.edges(data=True):
        w['weight'] = 1
    G=G.to_undirected()
    graph_data = nx.to_pandas_edgelist(G, source="seller id", target="buyer id")
    graph_data.columns = ["seller id","buyer id", "total_value"]
    graph_data["tuple_id"] = list(G.edges())
    graph_data['trade date time'] = timestamp
    return(graph_data)


def static_er_gen_w(n, timestamp):
    """ Function to produce a static weighted erdos renyi graph, with two nodes in the bridge and n nodes in the cliques. 
    
    Parameters:
        n: number of nodes to make up the clique
        timestamp: if generating a static graph as a time snapshot instance of a temporal graph, specify the timestamp for the snapshot.
        
    Returns:
        pandas edgelist dataframe. 
    """
   
    G = nx.erdos_renyi_graph(n, 0.5)
    for (u,v,w) in G.edges(data=True):
        w['weight'] = random.randint(0,10)
    G=G.to_undirected()
    graph_data = nx.to_pandas_edgelist(G, source="seller id", target="buyer id")
    graph_data.columns = ["seller id","buyer id", "total_value"]
    graph_data["tuple_id"] = list(G.edges())
    graph_data['trade date time'] = timestamp
    return(graph_data)



def static_rg_gen(timestamp):
    """ Function to produce a static unweighted ring network with prespecified nodes and edges. 
    
    Parameters:
        timestamp: if generating a static graph as a time snapshot instance of a temporal graph, specify the timestamp for the snapshot.
        
    Returns:
        pandas edgelist dataframe. 
    """
    H = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
    H.add_weighted_edges_from([(0,1,1),(1,2,1),(2,3,1),(3,4,1),(4,5,1), (5,6,1),(6,7,1), (7,8,1),(8,9,1),(9,10,1),(10,11,1),(11,12,1),(12,13,1),(13,14,1),(14,0,1)])    
    H_data = nx.to_pandas_edgelist(H, source="seller id", target="buyer id")
    H_data.columns = ["seller id","buyer id", "total_value"]
    H_data["tuple_id"] = list(H.edges())
    H_data['trade date time'] = timestamp
    return(H_data)

def static_rg_gen_w(timestamp):
    """ Function to produce a static unweighted ring network with prespecified nodes and edges. 
    
    Parameters:
        timestamp: if generating a static graph as a time snapshot instance of a temporal graph, specify the timestamp for the snapshot.
        
    Returns:
        pandas edgelist dataframe. 
    """    
    H = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
    H.add_weighted_edges_from([(0,1,3.0),(1,2,7.5),(2,3,1),(3,4,6),(4,5,3), (5,6,1.4),(6,7,5.5), (7,8,2.9),(8,9,1.1),(9,10,4),(10,11,5.5),(11,12,4.2),(12,13,3),(13,14,0.4),(14,0,1)])     
    H_data = nx.to_pandas_edgelist(H, source="seller id", target="buyer id")
    H_data.columns = ["seller id","buyer id", "total_value"]
    H_data["tuple_id"] = list(H.edges())
    H_data['trade date time'] = timestamp
    return(H_data)


def edge_perturb(data, edge, pert, return_type):
    """ Function to apply a perturbation to a single edge in a network. Options to return the resultant change in eigenvalue, adjacency matrix and change in specific edge wieght
    
    Parameters:
            data: pandas edgelist dataframe, with columns seller id, buyer id total value and trade date time
            edge: single edge tuple in format (seller id, buyer id)
            pert: percentage change to apply to edge as a decimal (e.g. 10% =0.1)
            return_type: choose from
                - dlambda: return the change in the network leading eigenvalue
                - dA: return the change in the adjecency matrix entry for the specified edge
                - adj: return the resultant perturbed matrix.
    Returns:
        see return_type.
    """
    G = nx.from_pandas_edgelist(data, source="seller id", target = "buyer id",edge_attr = ['total_value'],create_using=nx.Graph())
    print("pert",pert)
    if return_type== "edgelist":
        data_return=data.copy()
        data_return.loc[data_return.tuple_id==edge, 'total_value'] = (1+pert)*data_return[data_return.tuple_id==edge].total_value

        return(data_return)
    if edge not in G.edges():
        print("help")
    A = nx.to_pandas_adjacency(G,weight='total_value')
    A_pert = A.copy()
    print("AP",A_pert)
    A_comp=A.loc[edge[0], edge[1]]
    A_pert.loc[edge[0], edge[1]] = (1+pert)*A.loc[edge[0], edge[1]]
    A_pert.loc[edge[1], edge[0]] = (1+pert)*A.loc[edge[1], edge[0]]
    print(A_pert)
    A_comp_pert=A_pert.loc[edge[1], edge[0]]
    #for the eigenvalues we need to use the full symmetric network.
    eigenvalues, eigvecs = np.linalg.eigh(A)
    eigenvalues = max(eigenvalues)
    eigenvalues_pert, eigvecs = np.linalg.eigh(A_pert)
    eigenvalues_pert = max(eigenvalues_pert)
    if return_type == "dlambda":
        dlambda = eigenvalues_pert-eigenvalues
        return(dlambda)
    if return_type == "dA":
        dA = A_comp_pert-A_comp
        return(dA)
    if return_type == "adj": 
        return(A_pert)
    
def edge_perturb_double(data, edge1,edge2, pert, return_type):
    """ Function to apply a perturbation to a two edges in a network. Options to return the resultant change in eigenvalue, adjacency matrix and change in specific edge wieght
    
    Parameters:
            data: pandas edgelist dataframe, with columns seller id, buyer id total value and trade date time
            edge1: single edge tuple in format (seller id, buyer id)
            edge2: see edge 1
            pert: percentage change to apply to edge as a decimal (e.g. 10% =0.1)
            return_type: choose from
                - dlambda: return the change in the network leading eigenvalue
                - dA: return the change in the adjecency matrix entry for the specified edge
                - adj: return the resultant perturbed matrix.
    Returns:
        see return_type.
    """
    G = nx.from_pandas_edgelist(data, source="seller id", target = "buyer id",edge_attr = ['total_value'],create_using=nx.Graph())
    #G=G.to_undirected()
    A = nx.to_pandas_adjacency(G,weight='total_value')
    A_pert = A.copy()
    #print(A_pert)
    #currently looking just at the first edge
    A_comp=A.loc[edge1[0], edge1[1]]
    A_comp1=A.loc[edge2[0], edge2[1]]

    A_pert.loc[edge1[0], edge1[1]] = (1+pert)*A.loc[edge1[0], edge1[1]]
    A_pert.loc[edge1[1], edge1[0]] = (1+pert)*A.loc[edge1[1], edge1[0]]
    A_pert.loc[edge2[0], edge2[1]] = (1+pert)*A.loc[edge2[0], edge2[1]]
    A_pert.loc[edge2[1], edge2[0]] = (1+pert)*A.loc[edge2[1], edge2[0]]
    A_comp_pert=A_pert.loc[edge1[0], edge1[1]]
    A_comp_pert1=A_pert.loc[edge2[0], edge2[1]]
    #for the eigenvalues we need to use the full symmetric network.
    eigenvalues, eigvecs = np.linalg.eigh(A)
    eigenvalues = max(eigenvalues)
    eigenvalues_pert, eigvecs = np.linalg.eigh(A_pert)
    eigenvalues_pert = max(eigenvalues_pert)
    if return_type == "dlambda":
        dlambda = eigenvalues_pert-eigenvalues
        return(dlambda)
    if return_type == "dA":
        dA = A_comp_pert-A_comp
        dA1=A_comp_pert1-A_comp1
        return(dA, dA1)
    if return_type == "adj": 
        return(A_pert)

def snapshot_perturb(G_df):       
    """ Function to apply single edge perturbations to all edges, and compare the resultant relationship between edge weight change and eigenvalue change.    
    Parameters:
            G_df: pandas edgelist dataframe for single timesnapshot. Columns seller id, buyer id, total_value and trade date time. 
    Returns:
        plot of dA vs. dlambda, overlaid with line of constant l_e estimated for snapshot given by G_df.
    """
    perturbations = np.arange(-0.1, 0.120, 0.02)

    G_2 = nx.from_pandas_edgelist(G_df, source="seller id", target = "buyer id",edge_attr = ['total_value'],create_using=nx.Graph())
    G_2_df = nx.to_pandas_edgelist(G_2.to_undirected())
    G_2_df.columns=["seller id", "buyer id","total_value"]

    fig, ax_list = plt.subplots(4, 4, figsize = (10,10))
    ax_list = ax_list.flatten()
    i=0
    for j,n1 in enumerate(reversed(list(G_2.edges()))):
        print(n1)
        m_i_val = sim.le_generator_symm_evc(G_2_df, n1)
        
        nodebt2_pert_S= [edge_perturb(G_2_df, n1, pert = x,return_type = "dA") for x in perturbations]
        #print(nodebt2_pert_S)
        nodebt2_pert_lambda = [edge_perturb(G_2_df, n1, pert = x,return_type = "dlambda") for x in perturbations]
        print(nodebt2_pert_lambda)
        try:
            slope, intercept = np.polyfit(nodebt2_pert_S, nodebt2_pert_lambda, 1)
        except:
            continue
        print("slope",slope)
        print("l_e",m_i_val)
        print(m_i_val/slope)
        x,y = abline(m_i_val,0, min(nodebt2_pert_S)-0.01,max(nodebt2_pert_S)+0.01)
        print(x,y)
        ax_list[i].scatter(nodebt2_pert_S,nodebt2_pert_lambda, color="r", marker="+",label = "perturbation results")
        ax_list[i].plot(x,y, label = "$l_e$")
        #ax_list[i].legend().set_text("edge %s" %str(n1) +"\n l_e %s" %str(round(le_val, 3)))
        ax_list[i].text(0.9,0.05,"edge %s" %str(n1) +"\n l_e %s" %str(round(sim.le_generator_symm_evc(G_2_df, n1), 5))+"\n weight %s" %str(G_2.get_edge_data(n1[0],n1[1]).get("total_value")),fontsize=10, ha="right", 
     transform=ax_list[i].transAxes)
        ax_list[i].set_xlabel("$\Delta A(\epsilon)$", fontsize=8, labelpad=-2)
        ax_list[i].set_ylabel("$\Delta \lambda$", fontsize=8, labelpad=-5)
        ax_list[i].set_xlim((min(nodebt2_pert_S)-0.01, max(nodebt2_pert_S)+0.01))
        ax_list[i].set_ylim(-max(nodebt2_pert_lambda),max(nodebt2_pert_lambda))
        i+=1
    ax_list[-1]=nx.draw_networkx(G_2, node_size=100) 
    fig.tight_layout()
    plt.show()
    
def snapshot_perturb_double(G_df):    
    """ Function to apply double edge perturbations to all edges, and compare the resultant relationship between edge weight change and eigenvalue change.    
    Parameters:
            G_df: pandas edgelist dataframe for single timesnapshot. Columns seller id, buyer id, total_value and trade date time. 
    Returns:
        plot of dA vs. dlambda, overlaid with line of constant l_e estimated for snapshot given by G_df.
    """
    perturbations = np.arange(-0.1, 0.120, 0.02)

    G_2 = nx.from_pandas_edgelist(G_df, source="seller id", target = "buyer id",edge_attr = ['total_value'],create_using=nx.Graph())
    G_2_df = nx.to_pandas_edgelist(G_2.to_undirected())
    G_2_df.columns=["seller id", "buyer id","total_value"]

    fig, ax_list = plt.subplots(9, 4, figsize = (15,50))
    ax_list = ax_list.flatten()
    i=0
    lambda_comp={}
    node1_df=[]
    node2_df=[]
    for j,n1 in enumerate(reversed(list(G_2.edges()))):
        #if n1prev==n1:
        #    continue
        if i==35:
                print("break")
                break
        for k,n2 in enumerate(list(G_2.edges())):
            #if str(n2) in n2list:
            #    continue
            if n1==n2:
                continue
            print(n1)
            print(n2)
            m_i_val = sim.le_generator_symm_evc(G_2_df, n1)
            m_i_val1 = sim.le_generator_symm_evc(G_2_df, n2)
            nodebt2_pert_S1 = [edge_perturb_double(G_2_df, n1,n2, pert = x,return_type = "dA")[0] for x in perturbations]
            nodebt2_pert_S2 = [edge_perturb_double(G_2_df, n1,n2, pert = x,return_type = "dA")[1] for x in perturbations]
            nodebt2_pert_S = [edge_perturb_double(G_2_df, n1,n2, pert = x,return_type = "dA")[0]+edge_perturb_double(G_2_df, n1,n2, pert = x,return_type = "dA")[1] for x in perturbations]
            approx_lambda = [edge_perturb_double(G_2_df, n1,n2, pert = x,return_type = "dA")[0]*m_i_val+edge_perturb_double(G_2_df, n1,n2, pert = x,return_type = "dA")[1]*m_i_val1 for x in perturbations]
            #print(nodebt2_pert_S)
            nodebt2_pert_lambda = [edge_perturb_double(G_2_df, n1,n2, pert = x,return_type = "dlambda") for x in perturbations]
            print("lambda",nodebt2_pert_lambda)
            print("approx lambda", approx_lambda)
            lambda_comp[i] = pd.DataFrame([[i,j] for i,j in zip(nodebt2_pert_lambda, approx_lambda)], columns = ["Lambda", "Approx_lambda"])
            reg = LinearRegression().fit(np.column_stack([nodebt2_pert_S1,nodebt2_pert_S2]),nodebt2_pert_lambda)
            slope1=reg.coef_[0]
            slope2=reg.coef_[1]
            if m_i_val==0:
                continue
            print("slope",slope1,slope2)
            print("l_e",m_i_val,m_i_val1)
            node1_df.append(pd.Series([slope1, m_i_val]))
            node2_df.append(pd.Series([slope2, m_i_val1]))
            x,y = abline(m_i_val,0, min(nodebt2_pert_S)-0.01,max(nodebt2_pert_S)+0.01)
            print(x,y)
            ax_list[i].scatter(nodebt2_pert_S,nodebt2_pert_lambda, color="r", marker="+",label = "perturbation results")
            ax_list[i].plot(x,y, label = "$l_e$")
            #ax_list[i].legend().set_text("edge %s" %str(n1) +"\n l_e %s" %str(round(le_val, 3)))
            ax_list[i].text(0.9,0.05,"edge %s" %str(n1) + "\n 2nd edge %s" %str(n2),fontsize=10, ha="right", 
         transform=ax_list[i].transAxes)
            ax_list[i].set_xlabel("$\Delta A(\epsilon)$", fontsize=8, labelpad=-2)
            ax_list[i].set_ylabel("$\Delta \lambda$", fontsize=8, labelpad=-5)
            ax_list[i].set_xlim((min(nodebt2_pert_S)-0.01, max(nodebt2_pert_S)+0.01))
            ax_list[i].set_ylim(-max(nodebt2_pert_lambda),max(nodebt2_pert_lambda))
            i+=1
            if i==35:
                print("break")
                break
            
    ax_list[-1]=nx.draw_networkx(G_2, node_size=100) 
    fig.tight_layout()
    plt.show()
    return(lambda_comp, pd.DataFrame(node1_df), pd.DataFrame(node2_df))

def abline(slope, intercept, xmin, xmax):
    """Produce x,y required to plot a line from slope and intercept
    Parameters:
        slope: gradient (slope) of line 
        intercept: intercept of line
        xmin: minimum x value
        xmax: maximum x value
    Returns: x,y values for line.
    """
    x_vals = np.array((xmin, xmax))
    y_vals = intercept + slope * x_vals
    return(x_vals, y_vals)

if __name__ == "__main__":
    snapshot_perturb(static_er_gen(8,1))
    er_lambda_comp, er_node1, er_node2 = snapshot_perturb_double(static_er_gen(8,1))
    
    snapshot_perturb(static_er_gen_w(8,1))
    er_lambda_comp_w, er_node1_w, er_node2_w = snapshot_perturb_double(static_er_gen_w(8,1))

    snapshot_perturb(static_bb_gen_w(4,1))
    bb_lambda_comp, bb_node1, bb_node2 = snapshot_perturb_double(static_bb_gen_w(4,1))
    
    snapshot_perturb(static_bb_gen_w(4,1))
    bb_lambda_comp_w, bb_node1_w, bb_node2_w = snapshot_perturb_double(static_bb_gen_w(4,1))
    
    snapshot_perturb(static_rg_gen(1))
    r_lambda_comp, r_node1,r_node2=snapshot_perturb_double(static_rg_gen(1))
    
    snapshot_perturb(static_rg_gen_w(1))
    r_lambda_comp, r_node1,r_node2=snapshot_perturb_double(static_rg_gen_w(1))
    
    