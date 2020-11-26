"""
Created on Sun Sep 13 15:13:33 2020

@author: iseabrook1
"""

#This script contains the code required to produce analyse the predictability of 
#binary edge changes given the value of l_e for each edge.

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
#   This script initially produces boxplots of the distribution of l_e for changing 
#   edges vs. unchanging edges, and uses kernel density estimation to estimate and plot
#   distributions of P(Delta A=0|l_e). These are presented in the paper above, and related 
#   to the observed prediction capability.
#   The script then uses the observed change labels to train a logistic regression 
#   classifier to predict which edges will change given the value of l_e. The code
#   compares the predictions to a monte carlo average of a dummy classifier which randomly
#   predicts edges to change with probability 1/no.of observed changes. The code then outputs
#   results for balanced accuracy, Receiver Operating Characteristic Area Under Curve, 
#   Precision Recall Area Under Curve, and the plots associated with these.
#
###############################################################################

import sys
sys.path.append("N:/documents/packages")
sys.path.append("N:/documents/phdenv")
#import generate_data as gd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import sys
sys.path.append("N:/documents/packages")
sys.path.append("N:/documents/phdenv")
sys.path.append("..") # Adds higher directory to python modules path.
from model_evaluation import structural_importance_model as sim
import seaborn as sns
from scipy import stats

from imblearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score



if __name__ == "__main__":


    path_to_college="C:/Users/iseabrook1/OneDrive - Financial Conduct Authority/Network_analytics/PhD/Data"
    path_to_bilat="C:/Users/iseabrook1/OneDrive - Financial Conduct Authority/Network_analytics/PhD/Data"
    
    ds_le_bilat=pd.read_csv(path_to_bilat+"/ds_le_bilat.csv")
    ds_le_college=pd.read_csv(path_to_college+"/ds_le_college.csv")
    
    ds_le_bilat["change_bool"]=np.multiply(abs(ds_le_bilat.delta_A_act)>0,1)
    ds_le_college["change_bool"]=np.multiply(abs(ds_le_college.delta_A_act)>0,1)
    
    
    #step 1 - boxplots
    datasets=[ds_le_bilat,ds_le_college]
    ds_names = ["Bilateral Trade", "College Messaging"]
    fig, axs=plt.subplots(1,2, figsize=(10,5))
    axs=axs.flatten()
    all_t=[]
    all_p=[]
    for i, ds in enumerate(datasets):
        g1 = ds[(ds.log_l_e>-15)&(ds.change_bool==0)]["log_l_e"].values
        g2 = ds[(ds.log_l_e>-15)&(ds.change_bool==1)]["log_l_e"].values
        t,p=stats.ttest_ind(g1,g2)
        print(t,p)
        all_t.append(t)
        all_p.append(p)
        sns.boxplot(x="change_bool", y="log_l_e",  data=ds[(ds.log_l_e>-15)], ax=axs[i])
        axs[i].set_title(ds_names[i] +  f': p-value = {p:.3e}')
        axs[i].set_xlabel("Change label")
        axs[i].set_ylabel("$\ln(l_e)$")
    plt.show()
    #step 2 - change distributions
       
    change_pdf_vals_bilat=sim.change_dist(ds_le_bilat.set_index(["variable", "trade date time"]))
        
    ds_le_change_bilat= ds_le_bilat.merge(change_pdf_vals_bilat.to_frame().reset_index(), on=["trade date time", "variable"])
    ds_le_change_bilat.columns = [   'index',   'variable',  'trade date time',
                          'A_init',            'A_fin',              'l_e',
                    'delta_A_act',        'delta_A_rel1',
                        'log_l_e', 'log_delta_A_rel1',
                    'change_bool',          "change_pdf_vals"]
    
    change_pdf_vals_college=sim.change_dist(ds_le_college.set_index(["variable", "trade date time"]))
      
    ds_le_change_college= ds_le_college.merge(change_pdf_vals_college.to_frame().reset_index(), on=["trade date time", "variable"])
    ds_le_change_college.columns = [   'index',         'variable',  'trade date time',
                      'A_init',            'A_fin',              'l_e',
                'delta_A_act',        'delta_A_rel1',
                    'log_l_e', 'log_delta_A_rel1',
                'change_bool',          "change_pdf_vals"]
    
    
    plt.figure()
    plt.scatter(y=ds_le_change_bilat[ds_le_change_bilat.change_bool==0]["change_pdf_vals"],x=ds_le_change_bilat[ds_le_change_bilat.change_bool==0]["l_e"], marker='+', label="Bilateral trade")
    
    plt.scatter(y=ds_le_change_college[ds_le_change_college.change_bool==0]["change_pdf_vals"],x=ds_le_change_college[ds_le_change_college.change_bool==0]["l_e"], marker='+', label="College Msg")
    
    plt.xlabel("l_e")
    plt.ylabel("$P(\Delta A=0| l_e)$")
        
    plt.legend()
    plt.show()
    
    # #step 3 - change prediction
    
    classifiers_dict = {'rf':RandomForestClassifier(),
                        'lr':LogisticRegression(class_weight='balanced', random_state = 42),
                        'gb':GaussianNB()}
    
    classifier_params_dict={'rf':
                            {'clf__bootstrap': [False, True],
                            'clf__n_estimators': [80,90, 100, 110, 130]},
                      'lr':
                            {'clf__C': [0.001,0.01,0.1,1,10, 100],
                            'clf__penalty': ('l2', 'l1'),
                            'clf__max_iter': [50, 100, 200]},
                      'gb':
                            {'clf__var_smoothing': [0.00000001, 0.000000001, 0.00000001]}}
        
        
    datasets=[ds_le_bilat,ds_le_college]
    ds_names = ["Bilateral Trade", "College Messaging"]
    colors = ["navy","orange","g","r", "purple"]
    classifiers = [LogisticRegression(class_weight='balanced',C=100, max_iter=50, penalty='l2'), #Bilat 
                    LogisticRegression(class_weight='balanced',C=10, max_iter=200, penalty='l2'), #College 
                  ]
    fig, ax=plt.subplots(3,1, figsize=(10,5))
    fig1, ax1=plt.subplots(3,1, figsize=(10,5))
    ax=ax.flatten()
    ax1=ax1.flatten()
    for i, ds in enumerate(datasets):
    
        dataset_prec = ds.copy()
        
        print("PR")
        X, y = np.array(dataset_prec["log_l_e"]).reshape(-1,1), dataset_prec["change_bool"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify  = y)
        best_classifier = classifiers[i]
        
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
        fpr, tpr, _ = roc_curve(y_test, y_score[:,1])
        
        roc_auc= auc(fpr, tpr)
        print('ROC AUC: %.3f' % roc_auc)
    
        prec, recall, _ = precision_recall_curve(y_test, y_score[:,1],
                                                  pos_label=pipeline.named_steps['gb_classifier'].classes_[1])
        auc_score = auc(recall, prec)
        print('PR AUC: %.3f' % auc_score)
    
        #binomial monte carlo generation, attempt to use log reg to predict. 
        pr_auc_dumm_list = []
        bal_acc_list = []
        fpr_dumm_list=[]
        tpr_dumm_list=[]
        prec_dumm_list=[]
        recall_dumm_list=[]
        auc_dumm_list=[]
        thresholds_list=[]
        base_fpr = np.linspace(0, 1, 101)
    
        for j in range(100):
            ds_le_dumm = ds.copy()
            p = len(ds_le_dumm[ds_le_dumm.change_bool==1])/len(ds_le_dumm)
            ds_le_dumm.loc[:, "change_bool"] = np.random.binomial(1, p, len(ds_le_dumm))
            X_dumm, y_dumm = np.array(ds_le_dumm["log_l_e"]).reshape(-1,1), ds_le_dumm["change_bool"]
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
            bal_acc_list.append(balanced_accuracy_dumm)
            
            tpr_dumm = np.interp(base_fpr, fpr_dumm, tpr_dumm)
            prec_dumm[0] = 0.0
    
            tpr_dumm[0] = 0.0
            tpr_dumm_list.append(tpr_dumm)
            prec_dumm_list.append(prec_dumm)
            recall_dumm_list.append(recall_dumm)
            auc_dumm_list.append(roc_auc_dumm)
            thresholds_list.append(thresholds_dumm)
        
        balanced_accuracy_dumm = np.mean(bal_acc_list)
        tpr_dumm_list = np.array(tpr_dumm_list)
        tpr_dumm_mean = tpr_dumm_list.mean(axis=0)
        tpr_dumm_std = tpr_dumm_list.std(axis=0)
    
        mean_auc_dumm = auc(base_fpr, tpr_dumm_mean)
        std_auc_dumm = np.std(auc_dumm_list)
    
        tprs_upper = np.minimum(tpr_dumm_mean + tpr_dumm_std, 1)
        tprs_lower = tpr_dumm_mean - tpr_dumm_std
        
        prec_dumm_list = [i[0: min(map(len,thresholds_list))] for i in prec_dumm_list]
        prec_dumm_list = np.array(prec_dumm_list)
        prec_dumm_mean = prec_dumm_list.mean(axis=0)
        prec_dumm_std = prec_dumm_list.std(axis=0)
    
        std_pr_auc_dumm = np.std(auc_dumm_list)
    
        precs_upper = np.minimum(prec_dumm_mean + prec_dumm_std, 1)
        precs_lower = prec_dumm_mean - prec_dumm_std
        recall_dumm_list = [i[0:min(map(len,thresholds_list))] for i in recall_dumm_list]
        recall_dumm_list = np.array(recall_dumm_list)
        recall_dumm_mean = recall_dumm_list.mean(axis=0)
        mean_pr_auc_dumm = auc(recall_dumm_mean, prec_dumm_mean)
    
        print('Balanced accuracy dummy %.3f' % balanced_accuracy_dumm) 
        print('Dummy ROC AUC: %.3f' % mean_auc_dumm)
        print('Dummy PR AUC: %.3f' % mean_pr_auc_dumm)
    
        #take the values of l_e, take the distribution of 
        #dA and generate randomly dA. 
        
        # plot roc curves
        
        lw = 2
        ax[0].plot(fpr, tpr,lw=lw, label=ds_names[i],color=colors[i])
        ax[0].plot(base_fpr, tpr_dumm_mean,lw=lw, color=colors[i], linestyle='--', alpha=0.5)
        ax[0].fill_between(base_fpr, tprs_lower, tprs_upper, color = colors[i], alpha = 0.2)
    
        ax1[0].plot(recall,prec,lw=lw, label=ds_names[i], color=colors[i])
        ax1[0].plot(recall_dumm_mean, prec_dumm_mean,lw=lw, color=colors[i], linestyle='--', alpha=0.5)
        ax1[0].fill_between(recall_dumm_mean, precs_lower, precs_upper, color = colors[i], alpha = 0.2)
        
        ax[i+1].plot(fpr, tpr,lw=lw, label=ds_names[i],color=colors[i])
        ax[i+1].plot(base_fpr, tpr_dumm_mean,lw=lw, color=colors[i], linestyle='--', alpha=0.5)
        ax[i+1].fill_between(base_fpr, tprs_lower, tprs_upper, color = colors[i], alpha = 0.2)
    
        ax1[i+1].plot(recall,prec,lw=lw, label=ds_names[i], color=colors[i])
        ax1[i+1].plot(recall_dumm_mean, prec_dumm_mean,lw=lw, color=colors[i], linestyle='--', alpha=0.5)
        ax1[i+1].fill_between(recall_dumm_mean, precs_lower, precs_upper, color = colors[i], alpha = 0.2)
       
        ax[i+1].set_xlim([0.0, 1.0])
        ax[i+1].set_ylim([0.0, 1.05])
        ax[i+1].set_xlabel('False Positive Rate')
        ax[i+1].set_ylabel('True Positive Rate')
        ax[i+1].set_title(ds_names[i])
        ax[i+1].legend(loc="lower right")
        ax1[i+1].set_xlim([0.0, 1.0])
        ax1[i+1].set_ylim([0.0, 1.05])
        ax1[i+1].set_xlabel('Recall')
        ax1[i+1].set_ylabel('Precision')
        ax1[i+1].set_title(ds_names[i])
        ax1[i+1].legend(loc="best")
        
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_ylim([0.0, 1.05])
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('ROC curves for edge change prediction')
    ax[0].legend(loc="lower right")
    ax1[0].set_xlim([0.0, 1.0])
    ax1[0].set_ylim([0.0, 1.05])
    ax1[0].set_xlabel('Recall')
    ax1[0].set_ylabel('Precision')
    ax1[0].set_title('Precision-Recall curves for edge change prediction')
    ax1[0].legend(loc="best")
    plt.show()  
    
    
