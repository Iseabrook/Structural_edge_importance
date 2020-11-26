"""
Created on Sun Sep 13 15:13:33 2020

@author: iseabrook1
"""

#Methods for parameter estimation forthe structural importance model of network evolution. 

#Isobel Seabrook, ucabeas@ucl.ac.uk
#MIT License. Please reference below publication if used for research purposes.
#Reference: Seabrook et al., Evaluating Structural Edge Importance in Financial Networks
#
################################################################################
#   Instructions for use.
#
#   User is required to provide paths to the files containing l_e/dA pairs. 
#   These are generated for the college messaging dataset and bilateral trade dataset
#   in 'exploratory_data_analysis.py'.
#   This script then produces estimations of the parameters alpha and rho, using
#   COBYLA (https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyla.html)
#   numerical minimisation of the specified -log-likelihood function
#
#   The errors for the numerical estimations are calculated by approximating the 
#   inverse hessian matrix using numdifftools 
#   (https://numdifftools.readthedocs.io/en/latest/reference/numdifftools.html#numdifftools.core.Hessian)
#   
#   It then produces plots of the kernel density estimated plots P(Delta A=0|l_e), for the actual data
#   in comparison to data generated according to the structural important model for the estimated
#   parameters.

#   The script also includes the capability to plot the joint distributions of P(Delta A,l_e),
#   and uses the same method as for alpha and rho to estimate the parameters beta and gamma, 
#   and their associated errors.
###############################################################################
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import numdifftools as nd
import numpy.linalg as lin
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from model_evaluation import structural_importance_model as sim
import matplotlib.pyplot as plt
import seaborn as sns

def log_likelihood(ds_le, alpha, rho):
    """ log likelihood function for the parameters alpha and rho, for the structural
        importance model
        
        Parameters:
            ds_le: dataframe of dA le pairs
            alpha: alpha value
            rho: rho value
        
    Returns:
        value of log likelihood at alpha, rho
    """
    theta_e = pd.Series([alpha*pow(i, rho) for i in ds_le.l_e], index=ds_le.index)
    k_e = ds_le.change_bool
    L = pd.Series([(k_e[i] * np.log(theta_e[i])) + ((1-k_e[i])* (np.log(1-theta_e[i]))) for 
          i in theta_e.index])
    return(L.sum())  

def LL_beta_gamma(ds_le, beta, gamma):
    """ log likelihood function for the parameters beta and gamma, for the structural
        importance model
        
        Parameters:
            ds_le: dataframe of dA le pairs
            beta: beta value
            gamma: gamma value
        
    Returns:
        value of log likelihood at beta, gamma
    """
    theta_e = pd.Series([beta*pow(i, gamma) for i in ds_le.l_e], index=ds_le.index)
    x_e = ds_le.delta_A_act
    L = pd.Series([(np.divide(1, (2*np.pi*theta_e[i])) * np.exp(np.divide(-x_e[i]**2,2*theta_e[i]**2))) for 
          i in theta_e.index])
    #print(L.sum())
    return(L.sum()) 


if __name__ == "__main__":
    path_to_ds_le_bilat="C:/Users/iseabrook1/OneDrive - Financial Conduct Authority/Network_analytics/PhD/Data"
    path_to_ds_le_college="C:/Users/iseabrook1/OneDrive - Financial Conduct Authority/Network_analytics/PhD/Data"

    ds_le_bilat=pd.read_csv(path_to_ds_le_bilat+"/ds_le_bilat.csv")
    ds_le_college=pd.read_csv(path_to_ds_le_college+"/ds_le_college.csv")

    ds_le_bilat["change_bool"]=np.multiply(abs(ds_le_bilat.delta_A_act)>0,1)
    ds_le_college["change_bool"]=np.multiply(abs(ds_le_college.delta_A_act)>0,1)


    datasets=[ds_le_bilat,ds_le_college]
    ds_names = ["Bilateral Trade", "College Messaging"]

    # parameter estimation

    sol_list=[]
    var_list=[]
    for idx,ds in enumerate(datasets):
        def f(params):
            alpha, rho = params
            res=-log_likelihood(ds,alpha, rho)
            #print(res)
            return res

        cons = []
        for i in ds.l_e:
            def constraint(params):
                alpha, rho = params
                return(1-(alpha*pow(i, rho)))
            def constraint1(params):
                alpha, rho = params
                return(alpha*pow(i, rho))

            cons.append({'type': 'ineq', 'fun': constraint})
            cons.append({'type': 'ineq', 'fun': constraint1})

        solution = minimize(f, x0=(0.45, 0.9), constraints = cons, options={'rhobeg': 0.01, 'maxiter': 10000, 'disp': True, 'catol': 0.0000002},method = "COBYLA")
        print(ds_names[idx])
        print(solution)
        sol_list.append(solution)
        Hfun = nd.Hessian(f)
        h = Hfun(solution.x)
        print(h)
        var = lin.inv(h)
        var_list.append(var)
        print("errors", var)


    # generation from parameters

    ds_le_bilat_gen = sim.generate_temporal(pd.Series(np.linspace(ds_le_bilat.l_e.min(),ds_le_bilat.l_e.max(), 10000)), 1.16,0.430)
    ds_le_college_gen = sim.generate_temporal(pd.Series(np.linspace(ds_le_college.l_e.min(),ds_le_college.l_e.max(), 10000)), 0.545,0.034)

    fig, axs=plt.subplots(2,1, figsize=(10,5))
    ax1, ax2 = axs.flatten()
    namess = ["Real", "Generated"]
    for idx, ds in enumerate([ds_le_bilat, ds_le_bilat_gen]):

        change_pdf_vals=sim.change_dist(ds)
        change_pdf_vals.name = "change_pdf_vals" 
        ds_le_change= ds.merge(change_pdf_vals.to_frame().reset_index(), left_index=True, right_index=True)

        ax1.scatter(y=ds_le_change[ds_le_change.change_bool==0]["change_pdf_vals"],x=ds_le_change[ds_le_change.change_bool==0]["l_e"], marker='+', label=namess[idx])
        ax1.set_xlabel("l_e")
        ax1.set_ylabel("$P(\Delta A=0| l_e)$")
        ax1.set_title("Bilateral Trade")
    for idx, ds in enumerate([ds_le_college, ds_le_college_gen]):

        change_pdf_vals=sim.change_dist(ds)
        change_pdf_vals.name = "change_pdf_vals" 
        ds_le_change= ds.merge(change_pdf_vals.to_frame().reset_index(), left_index=True, right_index=True)

        ax2.scatter(y=ds_le_change[ds_le_change.change_bool==0]["change_pdf_vals"],x=ds_le_change[ds_le_change.change_bool==0]["l_e"], marker='+', label=namess[idx])
        ax2.set_xlabel("l_e")
        ax2.set_ylabel("$P(\Delta A=0| l_e)$")
        ax2.set_title("College Messaging")


    #gamma, beta parameters
    #distributions of weight changes
    fig, axs = plt.subplots(ncols=2, nrows=1)
    axs=axs.flatten()

    a=sns.jointplot(data=ds_le_bilat[(ds_le_bilat.delta_A_rel1!=0)&(ds_le_bilat.log_l_e>-15)&(abs(ds_le_bilat.log_delta_A_rel1)<10)], y="log_l_e", x="log_delta_A_rel1", kind='scatter', alpha=0.1, ax=axs[0])
    a.plot_joint(sns.kdeplot,ax=axs[0])
    
    b=sns.jointplot(data=ds_le_college[(ds_le_college.delta_A_rel1!=0)&(ds_le_college.log_l_e>-15)&(abs(ds_le_college.log_delta_A_rel1)<10)], y="log_l_e", x="log_delta_A_rel1", kind='scatter', alpha=0.1, color='orange',ax=axs[1])
    b.plot_joint(sns.kdeplot,ax=axs[1],color='orange')
    
 
    axs[0].set_xlabel("$\ln(1+\Delta A_{rel})$")
    axs[0].set_ylabel("$\ln(l_e)$")
    axs[1].set_xlabel("$\ln(1+\Delta A_{rel})$")
    axs[1].set_ylabel("$\ln(l_e)$")

    plt.show()
    
    sol_list_bg=[]
    var_list_bg=[]
    for idx,ds in enumerate(datasets):
        def f(params):
            beta, gamma = params
            res=-LL_beta_gamma(ds,beta, gamma)
            #print(res)
            return res
        
        cons = []
        for i in ds.l_e:
            def constraint1(params):
                beta, gamma = params
                return(beta*pow(i, gamma))
          
            cons.append({'type': 'ineq', 'fun': constraint1})
            
        
        solution = minimize(f, x0=(0.45, 0.9), constraints = cons, options={'rhobeg': 0.01, 'maxiter': 100000, 'disp': True, 'catol': 0.0000002},method = "COBYLA")
        print(ds_names[idx])
        print(solution)
        sol_list_bg.append(solution)
        Hfun = nd.Hessian(f)
        h = Hfun(solution.x)
        print(h)
        var = lin.inv(h)
        var_list_bg.append(var)
        print("errors", var)
    
