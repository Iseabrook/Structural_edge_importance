"""
Created on Sun Sep 13 15:13:33 2020

@author: iseabrook1
"""

#Exploratory data analysis of networks of bilateral trade and college messaging
#https://correlatesofwar.org/data-sets/bilateral-trade
#https://snap.stanford.edu/data/CollegeMsg.html
#This script includes the code required to produce the network summary statistics 
#presented in the paper referenced below. It also contains the functions required
#to produce dataframes of l_e values paired with the subsequent edge changes dA, which is
#required for much of the further analysis of this paper.

#Isobel Seabrook, ucabeas@ucl.ac.uk
#MIT License. Please reference below publication if used for research purposes.
#Reference: Seabrook et al., Evaluating Structural Edge Importance in Financial Networks
#
################################################################################
#   Instructions for use.
#
#   User is required to provide paths to the source files for the above datasets,
#   and the desired path at which to save these.. 
#   
#   This script then produces plots and summary statistics for these two datasets. These
#   are calculated for each time snapshot and across all time:
#   Number of nodes 
#   Number of edges 
#   density of network 
#   reciprocity
#   correlation coefficient
#   sum of all weights in network
#   maximum eigenvalue
#   
#   It also provides the functionality to calculate the spearman rank correlation
#   between l_e and other network quantities.

###############################################################################

import ast
import pandas as pd
import numpy as np

import itertools
from scipy import stats
import networkx as nx
from sklearn.linear_model import LinearRegression
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from model_evaluation import structural_importance_model as sim
import datetime
import matplotlib.pyplot as plt

def conn_stats(data):
    """ Function to calculate temporal network statistics
    
    Parameters:
        data: pandas edgelist dataframe for single graph snapshot, with columns seller id, buyer id total value and trade date time
        
    Returns:
        node_count: number of nodes for graph snapshot
        edge_count: number of edges for graph snapshot
        density: density of graph snapshot
        reciprocity: reciprocity of graph snapshot
        corr_coefficient: correlataion coefficient for graph snapshot
        value_sum: sum of weights for graph snapshot
        max(eigenvalues): maximum eigenvalue of graph.
    """
    M = nx.from_pandas_edgelist(data, source="seller id", target = "buyer id",edge_attr = ['total_value'],create_using=nx.MultiDiGraph())
    G = nx.DiGraph()
    for u,v,data1 in M.edges(data=True):
        w = data1['total_value'] if 'total_value' in data else 1.0
        if G.has_edge(u,v):
            G[u][v]['weight'] += w
        else:
            G.add_edge(u, v, weight=w)
    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    density = nx.density(G)
    reciprocity = nx.reciprocity(G)
    #connectivity = nx.average_node_connectivity(G)
    corr_coefficient = (nx.reciprocity(G)-nx.density(G))/max((1-nx.density(G), 0.00001))
    value_sum = data["total_value"].sum()
    eigenvalues, eigenvectors = np.linalg.eigh(nx.to_pandas_adjacency(G))
    print(eigenvalues)
    return(node_count,edge_count,density,reciprocity,corr_coefficient, value_sum, max(eigenvalues))

def strength_prods(g_df):
    """ Function to calculate the product of node strengths given a graph dataframe
    
    Parameters:
            data: pandas edgelist dataframe, with columns seller id, buyer id total value and trade date time
    
    Returns:
        dataframe of product of strengths, and individual node strengths per edge.
    """    
    whole_graph = nx.from_pandas_edgelist(g_df, source="seller id", target = "buyer id",edge_attr = ['total_value'],create_using=nx.Graph())
    whole_graph_uni = sim.multi_edge_to_uni(whole_graph)         
    deg_dict=dict(whole_graph_uni.degree(weight='total_value'))
    deg_prod = pd.DataFrame([i*j for i,j in itertools.product(deg_dict.values(),deg_dict.values())], 
          index=[(i,j) for i,j in itertools.product(deg_dict.keys(),deg_dict.keys())])
    deg_prod["node1_ec"]=[deg_dict.get(i) for (i,j) in deg_prod.index]
    deg_prod["node2_ec"]=[deg_dict.get(j) for (i,j) in deg_prod.index]                             
    deg_prod.index.name="variable"
    deg_prod.reset_index(inplace=True)
    deg_prod["variable"] = deg_prod.variable.apply(lambda x: ast.literal_eval(str(x)))

    deg_prod.set_index('variable', inplace=True)   
    deg_prod.columns=["deg_prod", "node1_deg", "node2_deg"]
    return(deg_prod)

def degree_prods(g_df):
    """ Function to calculate the product of node degreees given a graph dataframe
    
    Parameters:
            data: pandas edgelist dataframe, with columns seller id, buyer id total value and trade date time
    
    Returns:
        dataframe of product of degrees, and individual node strengths per edge.
    """    
    whole_graph = nx.from_pandas_edgelist(g_df, source="seller id", target = "buyer id",edge_attr = ['total_value'],create_using=nx.Graph())
    whole_graph_uni = sim.multi_edge_to_uni(whole_graph)         
    deg_dict=dict(whole_graph_uni.degree())
    deg_prod = pd.DataFrame([i*j for i,j in itertools.product(deg_dict.values(),deg_dict.values())], 
          index=[(i,j) for i,j in itertools.product(deg_dict.keys(),deg_dict.keys())])
    deg_prod["node1_ec"]=[deg_dict.get(i) for (i,j) in deg_prod.index]
    deg_prod["node2_ec"]=[deg_dict.get(j) for (i,j) in deg_prod.index]                             
    deg_prod.index.name="variable"
    deg_prod.reset_index(inplace=True)
    deg_prod["variable"] = deg_prod.variable.apply(lambda x: ast.literal_eval(str(x)))

    deg_prod.set_index('variable', inplace=True)   
    deg_prod.columns=["deg_prod", "node1_deg", "node2_deg"]
    return(deg_prod)


def correlation_calculations(whole_graph, ds_le, raw_data):
    """ Function to calculate correlations of l_e with dA, edge betweenness, product of node weights and product of node strengths. 
    
    Parameters:
            raw_data: pandas edgelist dataframe, with columns seller id, buyer id total value and trade date time
            whole_graph: networkx graph object corresponding to raw_data
            ds_le: dataframe of dA le pairs
    
    Returns:
        prints correlations listed above.
        ebc_le: edge betweenness and le pairs
        da_le_mean: mean values of da and le across time
        node_degs_le: node degree product and le pairs 
        strength_degs_le: node strength product and le pairs
    """    
    ebc_df = pd.DataFrame.from_dict(nx.edge_betweenness_centrality(whole_graph), orient='index')
    ebc_df.columns=["edge_betweenness"]
    ebc_df.index.name="variable"
    ebc_df.reset_index(inplace=True)

    ebc_df["variable"] = ebc_df.variable.apply(lambda x: ast.literal_eval(str(x)))
    ds_le.l_e=ds_le.l_e.apply(pd.to_numeric)
    le_sum_df = pd.DataFrame(ds_le[ds_le.delta_A_rel1!=0].groupby("variable").l_e.agg(np.nansum))
    le_sum_df.reset_index(inplace=True)
    le_sum_df["variable"] = le_sum_df.variable.apply(lambda x: ast.literal_eval(str(x)))

    le_sum_df.set_index('variable', inplace=True)
    ebc_df.set_index('variable', inplace=True)

    ebc_le = ebc_df.join(le_sum_df)
    ebc_le.dropna(inplace=True)
    ebc_le["le_rank"]=ebc_le.l_e.rank()
    ebc_le["ebc_rank"]=ebc_le.edge_betweenness.rank()

    #2
    print(stats.spearmanr(ebc_le.le_rank, ebc_le.ebc_rank))

    da_sum_df= pd.DataFrame(ds_le[ds_le.delta_A_rel1!=0].groupby("variable").delta_A_rel1.agg(np.nansum))
    da_sum_df.reset_index(inplace=True)
    da_sum_df=da_sum_df[da_sum_df.delta_A_rel1!=0]
    da_sum_df["variable"] = da_sum_df.variable.apply(lambda x: ast.literal_eval(str(x)))

    da_sum_df.set_index('variable', inplace=True)

    da_le_sum = da_sum_df.join(le_sum_df)
    da_le_sum["le_rank"]=da_le_sum.l_e.rank()
    da_le_sum["da_rank"]=da_le_sum.delta_A_rel1.rank()

    le_mean_df = pd.DataFrame(ds_le[ds_le.delta_A_rel1!=0].groupby("variable").l_e.agg(np.nanmean))
    le_mean_df.reset_index(inplace=True)
    le_mean_df["variable"] = le_mean_df.variable.apply(lambda x: ast.literal_eval(str(x)))

    le_mean_df.set_index('variable', inplace=True)   
    da_mean_df= pd.DataFrame(ds_le[ds_le.delta_A_rel1!=0].groupby("variable").delta_A_rel1.agg(np.nanmean))
    da_mean_df.reset_index(inplace=True)
    da_mean_df=da_mean_df[da_mean_df.delta_A_rel1!=0]
    da_mean_df["variable"] = da_mean_df.variable.apply(lambda x: ast.literal_eval(str(x)))

    da_mean_df.set_index('variable', inplace=True)

    da_le_mean = da_mean_df.join(le_mean_df)
    da_le_mean["le_rank"]=da_le_mean.l_e.rank()
    da_le_mean["da_rank"]=da_le_mean.delta_A_rel1.rank()

    print(stats.spearmanr(da_le_mean.dropna().le_rank, da_le_mean.dropna().da_rank))


    node_degs_le = raw_data.groupby("trade date time").apply(degree_prods)
    node_degs_le.reset_index(inplace=True)
    node_degs_le["trade date time"]=node_degs_le["trade date time"].astype(str)
    ds_le["trade date time"]=ds_le["trade date time"].astype(str)

    node_degs_le=ds_le[["variable", "trade date time", "l_e"]].merge(node_degs_le, on=["variable", "trade date time"])
    node_degs_le.dropna(inplace=True)
    node_degs_le["le_rank"]=node_degs_le.l_e.rank(ascending=False,method='dense')
    node_degs_le["dp_rank"]=node_degs_le.deg_prod.rank(ascending=False,method='dense')
    node_degs_le["node1_rank"]=node_degs_le.node1_deg.rank(ascending=False,method='dense')
    node_degs_le["node2_rank"]=node_degs_le.node2_deg.rank(ascending=False,method='dense')

    print(stats.spearmanr(node_degs_le.dropna().le_rank, node_degs_le.dropna().dp_rank))

    strength_degs_le = raw_data.groupby("trade date time").apply(strength_prods)
    strength_degs_le.reset_index(inplace=True)
    strength_degs_le["trade date time"]=strength_degs_le["trade date time"].astype(str)

    strength_degs_le=ds_le[["variable", "trade date time", "l_e"]].merge(strength_degs_le, on=["variable", "trade date time"])
    strength_degs_le.dropna(inplace=True)
    strength_degs_le["le_rank"]=strength_degs_le.l_e.rank(ascending=False,method='dense')
    strength_degs_le["sp_rank"]=strength_degs_le.deg_prod.rank(ascending=False,method='dense')
    strength_degs_le["node1_rank"]=strength_degs_le.node1_deg.rank(ascending=False,method='dense')
    strength_degs_le["node2_rank"]=strength_degs_le.node2_deg.rank(ascending=False,method='dense')
    print(stats.spearmanr(strength_degs_le.dropna().le_rank, strength_degs_le.dropna().sp_rank))

    return(ebc_le, da_le_mean, node_degs_le, strength_degs_le)

def normalize(df):
    """ Function to apply min-max normalisation to a given variable
    
    Parameters:
            df: variable required to normalise
            
    Returns:
        result: normalised variable
    """ 
    max_value = df.max()
    min_value = df.min()
    result = (df - min_value) / (max_value - min_value)
    return result
if __name__ == "__main__":
    path_to_college="C:/Users/iseabrook1/OneDrive - Financial Conduct Authority/Network_analytics/PhD/Data"
    path_to_save_college="C:/Users/iseabrook1/OneDrive - Financial Conduct Authority/Network_analytics/PhD/Data"
    path_to_bilat_folder = "C:/Users/iseabrook1/OneDrive - Financial Conduct Authority/Network_analytics/PhD/Data"
    path_to_save_bilat="C:/Users/iseabrook1/OneDrive - Financial Conduct Authority/Network_analytics/PhD/Data"
    # #college_messaging
    raw_data = pd.read_csv(path_to_college+"/CollegeMsg.txt", sep = " ", header=None)
    
    raw_data.columns = ["seller id", "buyer id", "trade date time"]
    raw_data["day"] =raw_data["trade date time"].apply(lambda x: datetime.datetime.fromtimestamp(x).date())
    raw_data["trade date time"] = raw_data["trade date time"].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
    raw_data["week_start"] = pd.to_datetime(raw_data["trade date time"].str[:10], format='%Y/%m/%d').dt.to_period('W').apply(lambda r: r.start_time)

    raw_data["total_value"] = 1    
    
    raw_data_day = raw_data.groupby(["buyer id", "seller id", "day"]).total_value.sum()
    raw_data_day = pd.DataFrame(raw_data_day).reset_index()
    raw_data_day["trade date time"] = raw_data_day.day

    raw_data_day = raw_data_day[(raw_data_day.day<datetime.date(2004,6,16)) & (raw_data_day.day>datetime.date(2004,4,25))]
    raw_data_day.drop("day", axis=1, inplace=True)
                  
    temporal_stats = raw_data_day.groupby(raw_data_day["trade date time"], axis=0).apply(conn_stats)
    temporal_stats=temporal_stats.apply(pd.Series)
    temporal_stats.columns = ["node_count", "edge_count", "density", "reciprocity","corr_coefficient", "value_sum", "eigval"]
    f, axs = plt.subplots(3,3,figsize=(15,15))
    k=0
    for i in range(3):
        for j in range(3):
            if k<7:
                ax = axs[i,j]
                #ax.xaxis.set_major_locator(plt.MaxNLocator(10))
                ax.tick_params(labelrotation=45)
                ax.plot(temporal_stats.iloc[:,k])
                ax.title.set_text(temporal_stats.columns[k])
                k+=1
                print(k)
    plt.show()

    whole_graph = nx.from_pandas_edgelist(raw_data_day, source="seller id", target = "buyer id",edge_attr = ['total_value'],create_using=nx.Graph())
    
    whole_graph_uni = sim.multi_edge_to_uni(whole_graph) 
                  
    print("Number of nodes",whole_graph.number_of_nodes())
    print(len(pd.unique(raw_data_day[['seller id', 'buyer id']].values.ravel('K'))))
    #edges
    print("Number of edges", whole_graph.number_of_edges())
    #connectivity - number of links/number of nodes
    print("Connectivity", whole_graph.number_of_edges()/whole_graph.number_of_nodes())
    #print("Nx connectivity", nx.average_node_connectivity(G_brown))
    #degree of completeness (number of links relative to number of possible links) aka denisty
    print("Density", nx.density(whole_graph)*100,"%")
    #Reciprocity - fraction of links where the is a link in the opposite direction
    print("Reciprocity", nx.overall_reciprocity(whole_graph)*100,"%")
    #Correlation coefficient of adjacency matrix, which takes into account that networks with higher hegree of completeness are more reciprocal
    print("Correlation coefficient of adjacency matrix", (nx.reciprocity(whole_graph)-nx.density(whole_graph))/(1-nx.density(whole_graph))*100,"%")
    recip = []
    price_sum=[]
    quant_sum=[]
    total_value=[]
    
    edge_pairs=[]
    for x,y in list(nx.Graph(whole_graph).edges()):
        edge_pairs.append([x,y])
        G_sub = whole_graph.subgraph([x,y])
        recip.append(nx.reciprocity(G_sub)) 
        price_sum.append(sum(nx.get_edge_attributes(G_sub,'price').values()))
        quant_sum.append(sum(nx.get_edge_attributes(G_sub,'quantity').values()))
        total_value.append(sum(nx.get_edge_attributes(G_sub,'total_value').values()))
    
    recip_price = pd.concat([pd.DataFrame(edge_pairs),pd.Series(recip), pd.Series(price_sum), pd.Series(quant_sum), pd.Series(total_value)], axis=1)
    recip_price.columns = ["node1", "node2","reciprocity", "price", "quantity", "total_value"]
    recip_price = recip_price.loc[recip_price['total_value']!=recip_price['total_value'].max()]
    recip_price = recip_price.loc[recip_price['total_value']!=recip_price['total_value'].max()]
    recip_price = recip_price.loc[recip_price['total_value']!=recip_price['total_value'].max()]
    recip_price = recip_price.loc[recip_price['total_value']!=recip_price['total_value'].max()]

    plt.figure()
    plt.scatter(recip_price.reciprocity,(recip_price.total_value))
    plt.xlabel("Reciprocity")
    plt.ylabel("Transaction value (Price*quantity)")
    plt.show()
    recip_price.dropna(inplace=True)
    model = LinearRegression().fit(recip_price.reciprocity.values.reshape(-1, 1), recip_price.price.values.reshape(-1, 1))
    r_sq = model.score(recip_price.reciprocity.values.reshape(-1, 1), recip_price.price.values.reshape(-1, 1))
    print('coefficient of determination:', r_sq)
    #recip_price.index=recip_price[["node1", "node2"]]
    print(np.corrcoef(recip_price[["reciprocity", "price", "quantity", "total_value"]], rowvar=False))
  
    ds_le_college=sim.da_le_pairs(raw_data_day)
    ds_le_college.to_csv(path_to_save_college+"/ds_le_college.csv")
    #Individual edge exploration
    ebc_le_college, da_le_mean_college, node_degs_le_college, strength_degs_le_college = correlation_calculations(whole_graph_uni, ds_le_college, raw_data_day)    
    
                  
                  
    #bilateral trade - flow 1 refers to flows from A-->B and flow 2 from B-->A.
    raw_data = pd.read_csv(path_to_bilat_folder+"/Dyadic_COW_4.0.csv")
    raw_data1 = raw_data.copy()
    raw_data.columns = ['ccode1', 'ccode2', 'trade date time', 'buyer id', 'seller id', 'total_value', 'flow2',
            'smoothflow1', 'smoothflow2', 'smoothtotrade', 'spike1', 'spike2',
            'dip1', 'dip2', 'trdspike', 'tradedip', 'bel_lux_alt_flow1',
            'bel_lux_alt_flow2', 'china_alt_flow1', 'china_alt_flow2', 'source1',
            'source2', 'version']
    
     
    raw_data1.columns = ['ccode1', 'ccode2', 'trade date time', 'seller id', 'buyer id', 'flow1', 'total_value',
            'smoothflow1', 'smoothflow2', 'smoothtotrade', 'spike1', 'spike2',
            'dip1', 'dip2', 'trdspike', 'tradedip', 'bel_lux_alt_flow1',
            'bel_lux_alt_flow2', 'china_alt_flow1', 'china_alt_flow2', 'source1',
            'source2', 'version']
    raw_data=raw_data[['trade date time','buyer id', 'seller id', 'total_value']]
    raw_data1 = raw_data1[['trade date time','buyer id', 'seller id', 'total_value']] 
    raw_data=pd.concat([raw_data, raw_data1], ignore_index=True)
    raw_data = raw_data[raw_data.total_value!=-9]
    # #filter the data to a manageable size
    selected_countries = raw_data[(raw_data.total_value.groupby(raw_data["buyer id"]).transform('sum') < 1e7) &(raw_data.total_value.groupby(raw_data["buyer id"]).transform('sum') > 1e6)]["buyer id"].unique()
    
    raw_data_filt = raw_data[(raw_data["buyer id"].isin(selected_countries))]
    raw_data_filt = raw_data_filt[raw_data_filt["seller id"].isin(selected_countries)]
    raw_data_filt=raw_data[raw_data["trade date time"]<1960]
    low, high = raw_data_filt.total_value.quantile([0.25,0.75])
    raw_data_filt =raw_data_filt.query('{low}<total_value<{high}'.format(low=low,high=high))
    
    raw_data_filt = raw_data_filt[raw_data_filt.total_value!=0]
    raw_data_filt["total_value"] = normalize(raw_data_filt.total_value)
                  
    temporal_stats = raw_data_filt.groupby(raw_data_filt["trade date time"], axis=0).apply(conn_stats)
    temporal_stats=temporal_stats.apply(pd.Series)
    temporal_stats.columns = ["node_count", "edge_count", "density", "reciprocity","corr_coefficient", "value_sum", "eigval"]
    f, axs = plt.subplots(3,3,figsize=(15,15))
    k=0
    for i in range(3):
        for j in range(3):
            if k<7:
                ax = axs[i,j]
                #ax.xaxis.set_major_locator(plt.MaxNLocator(10))
                ax.tick_params(labelrotation=45)
                ax.plot(temporal_stats.iloc[:,k])
                ax.title.set_text(temporal_stats.columns[k])
                k+=1
    plt.show()
    whole_graph = nx.from_pandas_edgelist(raw_data_filt, source="seller id", target = "buyer id",edge_attr = ['total_value'],create_using=nx.DiGraph())
    whole_graph_uni = sim.multi_edge_to_uni(whole_graph) 


    print("Number of nodes",whole_graph.number_of_nodes())
    print(len(pd.unique(raw_data_filt[['seller id', 'buyer id']].values.ravel('K'))))
    #edges
    print("Number of edges", whole_graph.number_of_edges())
    #connectivity - number of links/number of nodes
    print("Connectivity", whole_graph.number_of_edges()/whole_graph.number_of_nodes())
    #print("Nx connectivity", nx.average_node_connectivity(G_brown))
    #degree of completeness (number of links relative to number of possible links) aka denisty
    print("Density", nx.density(whole_graph)*100,"%")
    #Reciprocity - fraction of links where the is a link in the opposite direction
    print("Reciprocity", nx.overall_reciprocity(whole_graph)*100,"%")
    #Correlation coefficient of adjacency matrix, which takes into account that networks with higher hegree of completeness are more reciprocal
    print("Correlation coefficient of adjacency matrix", (nx.reciprocity(whole_graph)-nx.density(whole_graph))/(1-nx.density(whole_graph))*100,"%")
    recip = []
    price_sum=[]
    quant_sum=[]
    total_value=[]
    
    edge_pairs=[]
    for x,y in list(nx.Graph(whole_graph).edges()):
        edge_pairs.append([x,y])
        G_sub = whole_graph.subgraph([x,y])
        recip.append(nx.reciprocity(G_sub)) 
        price_sum.append(sum(nx.get_edge_attributes(G_sub,'price').values()))
        quant_sum.append(sum(nx.get_edge_attributes(G_sub,'quantity').values()))
        total_value.append(sum(nx.get_edge_attributes(G_sub,'total_value').values()))
    
    recip_price = pd.concat([pd.DataFrame(edge_pairs),pd.Series(recip), pd.Series(price_sum), pd.Series(quant_sum), pd.Series(total_value)], axis=1)
    recip_price.columns = ["node1", "node2","reciprocity", "price", "quantity", "total_value"]
    recip_price = recip_price.loc[recip_price['total_value']!=recip_price['total_value'].max()]
    recip_price = recip_price.loc[recip_price['total_value']!=recip_price['total_value'].max()]
    recip_price = recip_price.loc[recip_price['total_value']!=recip_price['total_value'].max()]
    recip_price = recip_price.loc[recip_price['total_value']!=recip_price['total_value'].max()]

    plt.figure()
    plt.scatter(recip_price.reciprocity,(recip_price.total_value))
    plt.xlabel("Reciprocity")
    plt.ylabel("Transaction value (Price*quantity)")
    plt.show()
    recip_price.dropna(inplace=True)
    model = LinearRegression().fit(recip_price.reciprocity.values.reshape(-1, 1), recip_price.price.values.reshape(-1, 1))
    r_sq = model.score(recip_price.reciprocity.values.reshape(-1, 1), recip_price.price.values.reshape(-1, 1))
    print('coefficient of determination:', r_sq)
    recip_price.index=recip_price[["node1", "node2"]]
    print(np.corrcoef(recip_price[["reciprocity", "price", "quantity", "total_value"]], rowvar=False))
                    
    ds_le_bilat=sim.da_le_pairs(raw_data_filt)
    ds_le_bilat.to_csv(path_to_save_bilat+"/ds_le_bilat.csv")    
    #Individual edge exploration
    ebc_le_bilat, da_le_mean_bilat, node_degs_le_bilat, strength_degs_le_bilat = correlation_calculations(whole_graph_uni, ds_le_bilat, raw_data_filt)    
