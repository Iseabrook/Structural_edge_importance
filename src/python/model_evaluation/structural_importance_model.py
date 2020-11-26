"""
Created on Sun Sep 13 15:13:33 2020

@author: iseabrook1
"""

#Methods for generating temporal networks given an initial snapshot according to the structural importance model. 
#This script contains the function to calculate the approximation to edge structural importance, l_e. 
#It then contains methods for generating networks according to the structural importance model, for different parameter values.
#The plots produced demonstrate the behaviour of the model according to the parameters, and how these parameters related to 
#predictability.
#Isobel Seabrook, ucabeas@ucl.ac.uk
#MIT License. Please reference below publication if used for research purposes.
#Reference: Seabrook et al., Evaluating Structural Edge Importance in Financial Networks
#
################################################################################
#   Instructions for use (generation from model).
#
#   generate lists of the chosen parameter ranges, and le values, to trial. Then generate a list of dataframes for these parameter values using
#
#   generate_temporal(le_list, alpha_list, rho_list)
#
#   generate change distribution plots of P(dA=0|l_e) and boxplots of values of l_e for change vs. no change using:
#   change_dist_plots(df_list)
#   change_boxplots(df_list)
#
#   The methods can also be applied to l_e lists for generated networks, as in le_validation, or other initial snapshots.
#
#   Compare predictive capability for different parameters using the output from model_change_predict. This function also produces results for observing the performance of the MLE numerical parameter estimation method. 
#   
#   This script also includes a function to generate distributions for varying of parameters beta and gamma, which has not been used in the above paper.
#
# Instructions for application on real datasets, to generate dA,l_e pairs
#    Use function da_le_pairs on a pandas edgelist dataframe with columns 'buyer id', 'seller id', as source/target,  'total_value' as weight, 'trade date time' as timestamp. This will produce a dataframe of l_e for each edge at each timestamp, along with the subsequent change in edge weight dA. 
#
###############################################################################
import networkx as nx
import pandas as pd
import numpy as np
import ast
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
from numpy.random import choice
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from model_evaluation import le_validation as lval
from data_evaluation import parameter_estimation as est

def le_generator_symm_evc(data, edge):
    """ Function to generate the value of l_e for an individual edge
    
    Parameters:
        data: pandas edgelist dataframe for network snapshot - columns seller id, buyer id, total_value and trade date time.
        edge: edge tuple (seller id, buyer id) 
        
    Returns:
        value of l_e calculated. Where nodes are not found in the giant component, edge tuple is returned.
    """
    M = nx.from_pandas_edgelist(data, source="seller id", target = "buyer id",
                                edge_attr = ['total_value'],
                                create_using=nx.MultiDiGraph())
    Mc_list =  max(nx.connected_components(M.to_undirected()), key=len)
    Mc = M.subgraph(Mc_list)
    if edge[0] in Mc.nodes() and edge[1] in Mc.nodes():
        #print(edge)
        G = nx.Graph()#DiGraph if directed
        G.add_nodes_from(Mc)
        for u,v,data1 in Mc.edges(data=True):
            w = data1['total_value'] if 'total_value' in data else 1.0
            if G.has_edge(u,v):
                G[u][v]['weight'] += w
            else:
                G.add_edge(u, v, weight=w)
        try:
            eigvect = nx.eigenvector_centrality(G, weight='weight',max_iter=500)
        except:
            eigvect = nx.eigenvector_centrality_numpy(G, weight='weight',max_iter=500)
        l_e = 2*eigvect.get(edge[0])*eigvect.get(edge[1])
        return(l_e)
    else:
        return(edge)
def da_le_pairs(g_df):
    """ Function to generate dataframe of dA l_e pairs. 
    
    Parameters:
        g_df: pandas edgelist dataframe for network snapshot - columns seller id, buyer id, total_value and trade date time.
        
    Returns: dataframe, columns:
        trade_date_time: timestamp of edge 
        l_e: value of l_e calculated for each edge
        delta_A_act: subsequent observed dA value. 
        A_init: initial weight
        A_fin: final weight
        delta_A_rel1: relative change in edge weight (A_fin-A_init/A_init)
        variable: edge tuple
        log_l_e: natural log of l_e
        log_delta_A_rel1: natural log of delta_A_rel1
        
    """
    G = nx.from_pandas_edgelist(g_df, source="seller id", target = "buyer id",edge_attr = ['total_value'],create_using=nx.MultiDiGraph())
    monthly_values_list = [[] for i in range(3)]
    counter = 0
    whole_graph_uni = multi_edge_to_uni(G)

    for edge in whole_graph_uni.edges():
        counter+=1
        monthly_values_list[0].append(edge)
        price_series = pd.Series(g_df[(g_df["buyer id"]==edge[1]) & (g_df["seller id"]==edge[0])].sort_values(by="trade date time").groupby("trade date time").total_value.sum())

        monthly_values_list[1].append(price_series)
        x = g_df.groupby(g_df["trade date time"], axis=0).apply(lambda x: le_generator_symm_evc(x, edge))
        monthly_values_list[2].append(x)

    # create a dataframe with columns for the value of l_e for a given edge for a given time, including both relative and absolute change and timestamp
    A= pd.DataFrame.from_records(monthly_values_list[1] , index=monthly_values_list[0])
    A = A.apply(lambda series: series.loc[:series.last_valid_index()].ffill(), axis=1)
    A.drop_duplicates(inplace=True)
    A_T=A.T
    A_T_shift = A.shift(-1,axis=1).T
    A_T.index.name = "trade date time"
    A_T_shift.index.name = "trade date time"
    
    l = pd.DataFrame.from_records(monthly_values_list[2] , index=monthly_values_list[0])
    A_T.drop_duplicates(inplace=True)
    A_T_shift.drop_duplicates(inplace=True)
    l.drop_duplicates(inplace=True)
    da_le = pd.concat([A_T.reset_index().melt(id_vars="trade date time").set_index(["trade date time","variable"]),\
                       A_T_shift.reset_index().melt(id_vars="trade date time").set_index(["trade date time","variable"]),\
                           l.T.reset_index().melt(id_vars="trade date time").set_index(["trade date time","variable"])], \
                          join = 'inner', axis=1, sort=True)
    da_le.reset_index(level=0, inplace=True)
    da_le.columns = ["trade date time","A_init","A_fin", "l_e"]
    da_le["delta_A_act"] = da_le.A_fin - da_le.A_init

    da_le =da_le[[type(x)!=tuple for x in da_le.l_e]] 
    da_le.reset_index(inplace=True)
    da_le["variable"] = da_le.variable.apply(lambda x: ast.literal_eval(str(x)))
    da_le["delta_A_rel1"] = da_le.delta_A_act/da_le["A_init"]
    da_le = da_le[da_le.delta_A_rel1!=np.inf]
    da_le["log_l_e"]=np.log(da_le.l_e.astype(np.float64))
    
    da_le["log_delta_A_rel1"] = np.log(1+da_le.delta_A_rel1)
    da_le=da_le[da_le.log_l_e!=-np.inf]
    da_le=da_le[da_le.log_delta_A_rel1!=-np.inf]
    
    da_le.dropna(inplace=True)    
    return(da_le)

def multi_edge_to_uni(G):
    """This function takes in a graph which has multiple edges between two nodes,
    and sums them to a single edge.
    
    Parameters:
        G: networkx graph with potentially multi-edges
        
    Returns:
        G_uni: graph with multi-edge weights summed to give single total weight.
    """
    G_uni = nx.Graph()
    for u,v,data in G.edges(data=True):
        w = data['total_value']

        if G_uni.has_edge(u,v):
            G_uni[u][v]['total_value'] += w
        else:
            G_uni.add_edge(u, v, total_value=w)
    return(G_uni)

def generate_temporal(le_list, alpha_list, rho_list):
    """ Function generate a list of two-snapshot network attributes (le, change_bool), for given lists of parameter values.
    
    Parameters:
        le_list: values of l_e for intial network snapshot
        alpha_list: list of alpha parameters for trial
        rho_list: list of rho parameters for trial
        
    Returns:
        df_list: list of pandas dataframes with columns l_e and change_bool (change_bool is 1 if edge changes, 0 otherwise)
        alpha_list: alpha_list used
        rho_list: rho_list used
    """

    df_list=[]
    try:
        for alpha in np.unique(alpha_list):
            for rho in np.unique(rho_list):
                le_interp=pd.Series([alpha*pow(i,rho) for i in le_list], index=le_list.index).fillna(0)
                le_interp[le_interp>1]=1
                print(le_interp)
                change_bool = pd.Series([np.random.binomial(1, i, 1)[0] for i in le_interp], index=le_interp.index)
                res_df = pd.DataFrame({"l_e": le_list, "change_bool": change_bool})
                df_list.append(res_df)
    except:
        try:
            alpha = np.unique(alpha_list)
            for rho in np.unique(rho_list):
                le_interp=pd.Series([alpha*pow(i,rho) for i in le_list], index=le_list.index).fillna(0)
                le_interp[le_interp>1]=1
                print(le_interp)
                change_bool = pd.Series([np.random.binomial(1, i, 1)[0] for i in le_interp], index=le_interp.index)
                res_df = pd.DataFrame({"l_e": le_list, "change_bool": change_bool})
                df_list.append(res_df)
        except:
            rho = np.unique(rho_list)
            for alpha in np.unique(alpha_list):
                print(alpha, rho)
                le_interp=pd.Series([alpha*pow(i,rho) for i in le_list], index=le_list.index).fillna(0)
                le_interp[le_interp>1]=1
                print(le_interp)
                change_bool = pd.Series([np.random.binomial(1, i, 1)[0] for i in le_interp], index=le_interp.index)
                res_df = pd.DataFrame({"l_e": le_list, "change_bool": change_bool})
                df_list.append(res_df)
        
    return(df_list,alpha_list, rho_list)

def generate_temporal_ind(le_list, alpha, rho):
    """ Function generate a two-snapshot network attributes (le, change_bool), for given lists of parameter values.
    
    Parameters:
        le_list: values of l_e for intial network snapshot
        alpha:  alpha parameter for trial
        rho_list:  rho parameter for trial
        
    Returns:
        res_df: pandas dataframe with columns l_e and change_bool (change_bool is 1 if edge changes, 0 otherwise)
        
    """
    le_interp=pd.Series([alpha*pow(i,rho) for i in le_list], index=le_list.index)#.fillna(1)
    le_interp[le_interp>1]=1
    change_bool = pd.Series([np.random.binomial(1, i, 1)[0] for i in le_interp], index=le_interp.index)
    res_df = pd.DataFrame({"l_e": le_list, "change_bool": change_bool})
    return(res_df)

def change_dist(data):
    """ Function to produce the values of the pdf for P(deltaA=0,1| l_e). 
    Parameters:
        data: dataset with columns l_e, change_bool
        
    Returns:
        pdf_vals: pandas series of the values estimated for the pdf.
    """
    if len(data)>1:

        dens_c = sm.nonparametric.KDEMultivariateConditional(endog=[data.change_bool.values],exog=[data.l_e.astype(np.float64).values], dep_type='u', indep_type='c', bw='normal_reference')
        pdf_vals = dens_c.pdf(data.change_bool.values,data.l_e.values)
    else:
        pdf_vals = np.nan

    pdf_vals = pd.Series(pdf_vals, index = data.index)
    return(pdf_vals)

def change_dist_plots(ds_list, param_list, param_label):
    """ Function to produce plot of the distribution P(dA=0|l_e) for dataframes generated for different parameter values. 
    Parameters:
        ds_list: list of dataframes with columns l_e, change_bool
        
    Returns:
        plot of the estimated distributions for the differently parameterised datasets.
    """
    plt.figure()
    for idx, ds in enumerate(ds_list):
        
        change_pdf_vals=change_dist(ds)
            
        ds_le_change= ds.merge(change_pdf_vals.to_frame().reset_index(), left_index=True, right_index=True)
        print(ds_le_change.columns)
        ds_le_change.columns = ['l_e', 'change_bool', 'log_l_e', "change_pdf_vals"]
        plt.scatter(y=ds_le_change[ds_le_change.change_bool==0]["change_pdf_vals"],x=ds_le_change[ds_le_change.change_bool==0]["l_e"], marker='+',label=param_label+str(round(param_list[idx], 3)))
        plt.xlabel("l_e")
        plt.ylabel("$P(\Delta A=0| l_e)$")
            
    plt.legend()
    plt.show()

def change_boxplots(ds_list, alpha_gens_list, param_label):
    """ Function to produce boxplots of l_e distributions segmented by changes vs. no changes
    Parameters:
        ds_list: list of dataframes with columns l_e, change_bool
        
    Returns:
        boxplots for the differently parameterised datasets
    """
    fig, axs=plt.subplots(nrows=2, ncols=5)
    axs=axs.flatten()
    all_t=[]
    all_p=[]
    for i, ds in enumerate(ds_list):
        ds["log_l_e"] = np.log(ds.l_e)
        g1 = ds[ds.change_bool==0]["log_l_e"].values
        g2 = ds[ds.change_bool==1]["log_l_e"].values
        t,p=stats.ttest_ind(g1,g2)
        print(t,p)
        all_t.append(t)
        all_p.append(p)
        sns.boxplot(x="change_bool", y="log_l_e",  data=ds, ax=axs[i])
        axs[i].set_title(param_label+str(round(alpha_gens_list[i],3)))
        axs[i].set_xlabel("Change label")
        axs[i].set_ylabel("$\ln(l_e)$")
    plt.show()


def model_change_predict(ds_list, alpha_list, rho_list):
    """ Function to estimate the parameters for generated datasets, and to assess the predictability for the different parameters
    Parameters:
        ds_list: list of dataframes with columns l_e, change_bool
        alpha_list: list of alpha values used to generate ds_list
        rho_list: list of rho values used to generate ds_list
        
    Returns:
        pr_auc_diff_list: difference from dummy model for precision-recall auc, for all parameter values
        roc_auc_diff_list: difference from dummy model for ROC auc, for all parameter values
        alpha_dl: difference in estimated and actual parameter values
        rho_dl: same as alpha_dl
    """
    pr_auc_diff_list = []
    roc_auc_diff_list= []
    alpha_dl = []
    rho_dl = []
    for idx, df in enumerate(ds_list):
        df["log_l_e"] = np.log(df.l_e)
        def f(params):
            alpha, rho = params
            res=-est.log_likelihood(df,alpha, rho)
            #print(res)
            return res
        
        cons = []
        for i in df.l_e:
            def constraint(params):
                alpha, rho = params
                return(1-(alpha*pow(i, rho)))
            def constraint1(params):
                alpha, rho = params
                return(alpha*pow(i, rho))
          
            cons.append({'type': 'ineq', 'fun': constraint})
            cons.append({'type': 'ineq', 'fun': constraint1})
    
    
        solution = minimize(f, x0=(0.45, 0.9), constraints = cons, options={'rhobeg': 0.01, 'maxiter': 100000, 'disp': True, 'catol': 0.0000002},method = "COBYLA")
        print(solution)
        alpha_diff = solution.x[0]-alpha_list[idx]
        rho_diff= solution.x[1]-rho_list[idx]
        alpha_dl.append(alpha_diff)
        rho_dl.append(rho_diff)
        dataset_prec = df.copy()
        
        print("PR")
        X, y = np.array(dataset_prec["log_l_e"]).reshape(-1,1), dataset_prec["change_bool"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify  = y)
        best_classifier = LogisticRegression(class_weight='balanced',C=1, max_iter=200, penalty='l2')
        
        pipeline = Pipeline(steps =  [ 
                                      ('sampler', RandomOverSampler()),        
        
                                      ('gb_classifier', best_classifier)
                                     ]) 
        pipeline.fit(X_train, y_train)
        y_prediction = pipeline.predict(X_test) 
        y_score = pipeline.predict_proba(X_test) 
        
        balanced_accuracy = balanced_accuracy_score(y_test, y_prediction)    
        print('Balanced accuracy %.3f' % balanced_accuracy)    
        #print(confusion_matrix(y_test, y_prediction)) 
        tn, fp, fn, tp = confusion_matrix(y_test, y_prediction).ravel()   
        #print("tn:",tn, "fp:", fp,"fn:", fn,"tp:",tp) 
        
        
        #look at PR AUC. Use this to justify choice of recall 
        #(if we were to consider e.g. F1 score)
        from sklearn.metrics import roc_curve, auc
        from sklearn.metrics import precision_recall_curve
        fpr, tpr, _ = roc_curve(y_test, y_score[:,1])
        
        roc_auc= auc(fpr, tpr)
        print('ROC AUC: %.3f' % roc_auc)
    
        prec, recall, _ = precision_recall_curve(y_test, y_score[:,1],
                                                  pos_label=pipeline.named_steps['gb_classifier'].classes_[1])
        auc_score = auc(recall, prec)
        print('PR AUC: %.3f' % auc_score)
    
        bal_acc_list = []
        tpr_dumm_list=[]
        prec_dumm_list=[]
        recall_dumm_list=[]
        auc_dumm_list=[]
        thresholds_list=[]
        base_fpr = np.linspace(0, 1, 101)
    
        for j in range(100):
            ds_le_dumm = df.copy()
            p = len(ds_le_dumm[ds_le_dumm.change_bool==1])/len(ds_le_dumm)
            ds_le_dumm.loc[:, "change_bool"] = np.random.binomial(1, p, len(ds_le_dumm))
            X_dumm, y_dumm = np.array(ds_le_dumm["log_l_e"]).reshape(-1,1), ds_le_dumm["change_bool"]
            if np.sum(y_dumm)<3:
                y_dumm[len(y_dumm)]=1
                y_dumm[len(y_dumm)-1]=1
            X_train_dumm, X_test_dumm, y_train_dumm, y_test_dumm = train_test_split(X_dumm, y_dumm, test_size=0.2, random_state=42, stratify  = y)
            
            pipeline_dumm = Pipeline(steps =  [ 
                                          ('sampler', RandomOverSampler()),        
            
                                          ('gb_classifier', best_classifier)
                                         ]) 
            pipeline_dumm.fit(X_train_dumm, y_train_dumm)
            y_prediction_dumm = pipeline_dumm.predict(X_test_dumm) 
            y_score_dumm = pipeline_dumm.predict_proba(X_test_dumm) 

            fpr_dumm, tpr_dumm, _ = roc_curve(y_test_dumm, y_score_dumm[:,1], drop_intermediate=False)
            
            roc_auc_dumm= auc(fpr_dumm, tpr_dumm)
            prec_dumm, recall_dumm, thresholds_dumm = precision_recall_curve(y_test_dumm, y_score_dumm[:,1],
                                                      pos_label=pipeline_dumm.named_steps['gb_classifier'].classes_[1])
            balanced_accuracy_dumm = balanced_accuracy_score(y_test_dumm, y_prediction_dumm)    
            
            tpr_dumm = np.interp(base_fpr, fpr_dumm, tpr_dumm)
            prec_dumm[0] = 0.0
    
            tpr_dumm[0] = 0.0
            #print(len([i for i in thresholds_dumm]))
            if len([i for i in thresholds_dumm])==1:
            
                tpr_dumm_list=tpr_dumm_list
            else:
                #print(len(thresholds_dumm))
                tpr_dumm_list.append(tpr_dumm)
                prec_dumm_list.append(prec_dumm)
                recall_dumm_list.append(recall_dumm)
                auc_dumm_list.append(roc_auc_dumm)
                thresholds_list.append(thresholds_dumm)
                bal_acc_list.append(balanced_accuracy_dumm)

        
        balanced_accuracy_dumm = np.mean(bal_acc_list)
        tpr_dumm_list = np.array(tpr_dumm_list)
        tpr_dumm_mean = tpr_dumm_list.mean(axis=0)
    
        mean_auc_dumm = auc(base_fpr, tpr_dumm_mean)
    
        
        prec_dumm_list = [i[0: min(map(len,thresholds_list))] for i in prec_dumm_list]
        prec_dumm_list = np.array(prec_dumm_list)
        prec_dumm_mean = prec_dumm_list.mean(axis=0)
    
    
        recall_dumm_list = [i[0:min(map(len,thresholds_list))] for i in recall_dumm_list]
        recall_dumm_list = np.array(recall_dumm_list)
        recall_dumm_mean = recall_dumm_list.mean(axis=0)
        mean_pr_auc_dumm = auc(recall_dumm_mean, prec_dumm_mean)
    
        print('Balanced accuracy dummy %.3f' % balanced_accuracy_dumm) 
        print('Dummy ROC AUC: %.3f' % mean_auc_dumm)
        print('Dummy PR AUC: %.3f' % mean_pr_auc_dumm)
    
        #take the values of l_e, take the distribution of 
        #dA and generate randomly dA. 
        pr_auc_diff_list.append(mean_pr_auc_dumm - auc_score)
        roc_auc_diff_list.append(mean_auc_dumm-roc_auc)
        
    return(pr_auc_diff_list, roc_auc_diff_list, alpha_dl, rho_dl)
 
def generate_temporal_allparams(G_df, alpha, rho, gamma, beta,t_range, allchange):
    """ Function generate a list of two-snapshot network attributes (le, change_bool), for given lists of parameter values, including the parameters sigma and beta
    
    Parameters:
        G_df: initial time snapshot edgelist dataframe
        alpha: alpha parameter for trial
        rho: rho parameter for trial
        gamma: gamma parameter for trial
        beta: beta parameter for trial
        t_range: number of time snapshots to generate
        allchange: bool for whether or not to change edges according to beta and gamma
        
    Returns:
        graph dataframe for all time
    """
    graph_list=[]
    graph_list.append(G_df)
    graph_data=G_df
    G_init = nx.from_pandas_edgelist(G_df,source="seller id", target = "buyer id", edge_attr=True, create_using = nx.Graph())
    for i in range(2,t_range):
        le_list=graph_data.tuple_id.apply(lambda x: le_generator_symm_evc(graph_data, x))
        total_value_list = pd.Series([G_init[u][v]['total_value'] for (u, v) in G_init.edges()], index=G_init.edges())
        le_interp=pd.Series([alpha*pow(i,rho) for i in le_list], index=total_value_list.index).fillna(1)
        le_interp=pd.Series([float(i)/np.nansum(le_interp.values) for i in le_interp.values], index=le_interp.index)
        change_bool =pd.Series([choice([0,1], p=[1-le_interp[i],le_interp[i]]) for i in G_init.edges()], index=G_init.edges())
        le_list=pd.Series([beta*(pow(i,gamma)) if i!=0 else 0 for i in le_list ], index=total_value_list.index).fillna(0)
        if allchange==True:
            new_total_value_list= dict(zip(G_init.edges(),[max((total_value_list[edge]*(1+np.random.normal(0,le_list[edge]))),0) for edge in G_init.edges()]))
        else:
            new_total_value_list= dict(zip(G_init.edges(),[(total_value_list[edge]*1.1) if change_bool[edge]==1 else total_value_list[edge] for edge in G_init.edges()]))

        nx.set_edge_attributes(G_init, new_total_value_list,'total_value')
        G_init=G_init.to_undirected()

        graph_data = nx.to_pandas_edgelist(G_init, source="seller id", target="buyer id")
        graph_data['trade date time'] = i
        graph_list.append(graph_data)  
    strength_graph_df_pos = pd.concat(graph_list, sort=True)
    return(strength_graph_df_pos)


def beta_gamma_trials(beta_list, gamma_list, gtype):
    """ Function to run trials across varying beta and gamma
    
    Parameters:
        beta_list: list of beta_parameters for trial
        gamma_list: list of gamma parameters for trials
        type of initial graph snapshot. Choose from 'er' (Erdos Renyi), 'bb' (Barbell), and 'rg' (ring)
    Returns:
        dataframe for all time of da le pairs and logged quantities derived from these
    """
    if gtype=="er":
        G_df = lval.static_er_gen(15,1)
    elif gtype=="bb":
        G_df = lval.static_bb_gen(8,1)
    elif gtype=="rg":
        G_df=lval.static_rg_gen(0)
    ds_le_dict={}

    for i, beta in enumerate(beta_list):
        strength_graph_df=generate_temporal_allparams(G_df, 1,gamma_list[i],beta, 20, 0.5, True)   
        ds_le = da_le_pairs(strength_graph_df, 0, len(strength_graph_df)+1)
        ds_le_dict[str("$\\beta$: "+str(beta)+", $\gamma$: "+str(gamma_list[i]))]=ds_le
    for vals, df in ds_le_dict.items():
        df['Values'] = vals
    ds_le_all = pd.concat(sorted(ds_le_dict.values(), key=lambda df: df['Values'][0]), ignore_index=True)
    ds_le_all_keys=[str("$\\beta$: "+str(beta)+", $\gamma$: "+str(gamma_list[i])) for i, beta in enumerate(beta_list)]
    print(ds_le_all_keys)
    fig, axs = plt.subplots(ncols=3, nrows=2)
    axs=axs.flatten()
    g=sns.jointplot(data=ds_le_all[(ds_le_all.delta_A_rel1!=0)&(ds_le_all.log_l_e>-15)&(abs(ds_le_all.log_delta_A_rel1)<10)], y="log_l_e", x="log_delta_A_rel1", hue='Values', hue_order=ds_le_all_keys,kind='scatter', alpha=0.1, ax=axs[0])
    g.plot_joint(sns.kdeplot, hue='dataset',ax=axs[0])
    
    a=sns.jointplot(data=ds_le_all[(ds_le_all.Values==ds_le_all_keys[0])&(ds_le_all.delta_A_rel1!=0)&(ds_le_all.log_l_e>-15)&(abs(ds_le_all.log_delta_A_rel1)<10)], y="log_l_e", x="log_delta_A_rel1", kind='scatter', alpha=0.1, ax=axs[1])
    a.plot_joint(sns.kdeplot,ax=axs[1])
    
    b=sns.jointplot(data=ds_le_all[(ds_le_all.Values==ds_le_all_keys[1])&(ds_le_all.delta_A_rel1!=0)&(ds_le_all.log_l_e>-15)&(abs(ds_le_all.log_delta_A_rel1)<10)], y="log_l_e", x="log_delta_A_rel1", kind='scatter', alpha=0.1, color='orange',ax=axs[2])
    b.plot_joint(sns.kdeplot,ax=axs[2],color='orange')
    
    c=sns.jointplot(data=ds_le_all[(ds_le_all.Values==ds_le_all_keys[2])&(ds_le_all.delta_A_rel1!=0)&(ds_le_all.log_l_e>-15)&(abs(ds_le_all.log_delta_A_rel1)<10)], y="log_l_e", x="log_delta_A_rel1", kind='scatter', alpha=0.1, color='g',ax=axs[3])
    c.plot_joint(sns.kdeplot,ax=axs[3],color='g')
    
    d=sns.jointplot(data=ds_le_all[(ds_le_all.Values==ds_le_all_keys[3])&(ds_le_all.delta_A_rel1!=0)&(ds_le_all.log_l_e>-15)&(abs(ds_le_all.log_delta_A_rel1)<10)], y="log_l_e", x="log_delta_A_rel1", kind='scatter', alpha=0.1,color='r', ax=axs[4])
    d.plot_joint(sns.kdeplot,ax=axs[4],color='r')
    
    e=sns.jointplot(data=ds_le_all[(ds_le_all.Values==ds_le_all_keys[4])&(ds_le_all.delta_A_rel1!=0)&(ds_le_all.log_l_e>-15)&(abs(ds_le_all.log_delta_A_rel1)<10)], y="log_l_e", x="log_delta_A_rel1", kind='scatter', alpha=0.1,color='purple', ax=axs[5])
    e.plot_joint(sns.kdeplot,ax=axs[5],color='purple')
    axs[0].set_xlabel("$\ln(1+\Delta A_{rel})$")
    axs[0].set_ylabel("$\ln(l_e)$")
    axs[1].set_xlabel("$\ln(1+\Delta A_{rel})$")
    axs[1].set_ylabel("$\ln(l_e)$")
    axs[2].set_xlabel("$\ln(1+\Delta A_{rel})$")
    axs[2].set_ylabel("$\ln(l_e)$")
    axs[3].set_xlabel("$\ln(1+\Delta A_{rel})$")
    axs[3].set_ylabel("$\ln(l_e)$")
    axs[4].set_xlabel("$\ln(1+\Delta A_{rel})$")
    axs[4].set_ylabel("$\ln(l_e)$")
    axs[5].set_xlabel("$\ln(1+\Delta A_{rel})$")
    axs[5].set_ylabel("$\ln(l_e)$")
    plt.show()
    return(ds_le_all)

if __name__ == "__main__":
    #Generate networks with varying alpha
    alpha_list = np.linspace(0.1,0.9,10)
    rho_list =  [0.5 for i in alpha_list]
    alpha_gens, alpha_gens_list, rho_gens_list= generate_temporal(pd.Series(np.linspace(0.2,0.9, 10000)), alpha_list, rho_list)

    change_dist_plots(alpha_gens, alpha_gens_list, "$\\alpha$: ")
    change_boxplots(alpha_gens, alpha_gens_list, "$\\alpha$: ")
    
    alpha_pr_auc_diffs, alpha_roc_auc_diffs, alpha_param_dl_alpha, alpha_param_dl_rho = model_change_predict(alpha_gens, alpha_list, rho_list)
    plt.figure()
    plt.scatter(alpha_gens_list, [abs(i) for i in alpha_pr_auc_diffs], marker='+')
    plt.xlabel("$\\alpha$")
    plt.ylabel("AUC improvement on dummy model")
    plt.show()    
    
    #Generate networks with varying rho
    rho_list1 = np.linspace(0,1, 10)
    alpha_list1 = [0.5 for i in rho_list]


    rho_gens, alpha_gens_list1, rho_gens_list1= generate_temporal(pd.Series(np.linspace(0.2,0.9, 10000)), alpha_list1, rho_list1)

    change_dist_plots(rho_gens, rho_gens_list1, "$\\rho$: ")
    change_boxplots(rho_gens, rho_gens_list1, "$\\rho$: ")

    
    rho_pr_auc_diffs, rho_roc_auc_diffs, rho_param_dl_alpha, rho_param_dl_rho = model_change_predict(rho_gens, alpha_list1, rho_list1)
    
    plt.figure()
    plt.scatter(rho_gens_list1, [abs(i) for i in rho_pr_auc_diffs], marker='+')
    plt.xlabel("$\\rho$")
    plt.ylabel("AUC improvement on dummy model")
    plt.show()
    
    #validate the reproduction of parameters using MLE estimation
    plt.figure()
    plt.scatter(rho_list1, rho_param_dl_alpha)
    plt.scatter(rho_list1, rho_param_dl_rho)
    plt.scatter(alpha_list, alpha_param_dl_alpha)
    plt.scatter(alpha_list, alpha_param_dl_rho)
    plt.xlabel("$\\rho$, $\\alpha$")
    plt.ylabel("MLE difference from true value")    
    plt.show()