import time
from datetime import timedelta
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import classification_report,auc,cohen_kappa_score,confusion_matrix,accuracy_score,roc_auc_score,log_loss
from eegUtilities import NormInput, BalanceSamples,BalanceSamples2,catmean
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import json
from eegUtilities import BalanceSamples
from itertools import combinations
from multiprocessing import Pool
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.stats import zscore
cpus = 10

# from keras.utils import to_categorical
import warnings
warnings.filterwarnings('error', category=RuntimeWarning) # catch runtimewarning when calculate inverse matrix


# use weibull cdf to fit psychometric fun
def applyweibullfit(x,y,threshx = 0.7,lapse=0):
    def weibull_cdf(x, a, b,c,d):
        return c - d*np.exp(-(x/b)**a)  #a-slope b-threshold
        # return (1-lapse)*(c - d*np.exp(-(x/b)**a))+lapse #a-slope b-threshold
        
    if y[0]>=y[-1]: #slope is negative
        print('WARNING: weibull slope is negative!!')
        bounds_low = (-np.inf,0,0,0) # x value was shifted to positive, so threshold lowbound is 0
        bounds_hi = (0,np.inf,np.inf,np.inf)
    else:#slope is positive
        bounds_low = (0,0,0,0)
        bounds_hi = (np.inf,np.inf,np.inf,np.inf)       
        # bounds_low = (0,0,0.5,0.5)
        # bounds_hi = (np.inf,np.inf,1,1)

    # Use curve fitting to find the best values for the parameters
    try:
        popt, _ = curve_fit(weibull_cdf, x+np.abs(x.min())+2, y,bounds=(bounds_low,bounds_hi),method='trf',maxfev=30000)
    except RuntimeWarning:
        print('weibull curve_fit error:RuntimeWarning')
        popt = [np.nan,np.nan,np.nan,np.nan]
    except Exception as e:
        print('weibull curve_fit fail')
        print(e)
        popt = [np.nan,np.nan,np.nan,np.nan]        
    # Generate the fitted curve using the optimized parameters
    x_fit = np.linspace(x.min(), x.max(), 100)+np.abs(x.min())+2
    y_fit = weibull_cdf(x_fit, popt[0], popt[1],popt[2],popt[3]) # apply the optimized parameters : a_opt b_opt
    # print('a:'+str(popt[0])+'\nb:'+str(popt[1])+'\nc:'+str(popt[2])+'\nd:'+str(popt[3]))
    y_fit_discrete = weibull_cdf(x+np.abs(x.min())+2, popt[0], popt[1],popt[2], popt[3])
    # get threshold at threshx
    if y.min()>threshx:
        thresh = x.min()
    elif y.max()<threshx:
        thresh = x.max()
        print('fail to get threshold')
    else:
        thresh = x_fit[np.argmin(np.abs(y_fit-threshx))]-np.abs(x.min())-2
    popt[1] = thresh
    return x_fit-np.abs(x.min())-2,y_fit,popt,y_fit_discrete

# use logistic cdf to fit psychometric fun
def applylogisticfit(x,y): 
    if y[0]>=y[-1]: #slope is negative
        bounds_low = (-np.inf,0,x.min(),0)
        bounds_hi = (0,np.inf,x.max(),np.inf)
    else:#slope is positive
        bounds_low = (0,0,x.min(),0)
        bounds_hi = (np.inf,np.inf,x.max(),np.inf)        

    def logisticfun(x, a, b, x0, y0):
        return y0 + b/(1+np.exp(-(x-x0)/a)) #a-slope b-range
    # Use curve fitting to find the best values for the parameters
    try:
        popt, _ = curve_fit(logisticfun, x, y,bounds=(bounds_low,bounds_hi),method='trf',maxfev=15000)
    except Exception as e:
        print('neurMetric curve_fit error:')
        print(e)
        popt = [np.nan,np.nan,np.nan,np.nan]
    # Generate the fitted curve using the optimized parameters
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = logisticfun(x_fit, popt[0], popt[1], popt[2], popt[3]) # apply the optimized parameters : a_opt b_opt
    y_fit_discrete = logisticfun(x, popt[0], popt[1],popt[2], popt[3])
    # get threshold at dprime=1
    if y.min()>1:
        thresh = y.min()
    elif y.max()<1:
        thresh = np.nan
    else:
        thresh = x_fit[np.argmin(np.abs(y_fit-1))]
    popt[1] = thresh
    return x_fit,y_fit,popt,y_fit_discrete

def decoder_detection1(AllSUspk_df,method,figSavePath): # keep the same order of trials for each cluster in the session, use all trials without replacement  
    clsall = list(AllSUspk_df.sess_cls.unique())
    Nnlist = np.arange(1,len(clsall)+1,1)#
    clsSamp = 15

    extrcolstr='*'
    fitacc_nNueron_dict = {}

    spkdf_sig = pd.DataFrame()
    spkdf_noise_a = pd.DataFrame()
    spkdf_noise_v = pd.DataFrame()

    for cls in AllSUspk_df['sess_cls'].unique():  
        SUspk_df_temp =  AllSUspk_df[AllSUspk_df['sess_cls']==cls].sort_values(by=['trialNum'],kind='mergesort')
        SUspk_df_temp.reset_index(drop=True,inplace=True)     
        # kepp trials order with same snr,trialmod,resp to keep trial by trial correlations in the same session
        grouped = SUspk_df_temp.groupby(by=['trialMod','snr-shift','respLabel'])# the category group should appear the same order when loop across cls
        group_indices = {key: grouped.groups[key].tolist() for key in grouped.groups}
        spkdf_sig_cat_temp = pd.DataFrame()
        for gkey,glist in group_indices.items():
            ind_temp_sig = glist # save trials in each cluster in same order
            if any(substr in gkey[0] for substr in ['a','av']) and any(substr in gkey[2] for substr in ['hit','miss']):# a/av signal trial
                collist_sig=[extrcolstr+''.join([str(vv)+'_' for vv in gkey])+'sig'+extrcolstr+str(tt) for tt in ind_temp_sig]
                spkdf_sig_cat_temp = pd.concat([spkdf_sig_cat_temp,pd.DataFrame(SUspk_df_temp.loc[ind_temp_sig]['spkRateSig'].values.reshape(1,-1),columns=collist_sig)],axis=1) # rowxcol: cls X trials
            if gkey[0]=='v' and gkey[2]=='CR':# visual noise trial, code should only run one time for each cls
                # print(cls+' this text should only appear one time for each cls!')
                collist_sig=[extrcolstr+'v_CR_noise'+extrcolstr+str(tt) for tt in ind_temp_sig]                
                spkdf_noise_v_cr_temp = pd.DataFrame(SUspk_df_temp.loc[ind_temp_sig]['spkRateSig'].values.reshape(1,-1),columns=collist_sig)                    
        # visual noise trial (FA)
        ind_temp_noise = list(SUspk_df_temp[SUspk_df_temp['respLabel']=='FAv'].index) 
        collist_noise=[extrcolstr+'v_FA_noise'+extrcolstr+str(tt) for tt in ind_temp_noise]
        spkdf_noise_v_fa_temp = pd.DataFrame(SUspk_df_temp.loc[ind_temp_noise]['spkRateSig'].values.reshape(1,-1),columns=collist_noise) # rowxcol: cls X trials
                
        # audio noise trial (CR), same num of sample as visual noise trial, some trial the spkRateNoise could be NaN  
        ind_temp_noise = list(SUspk_df_temp.dropna(subset=['spkRateNoise']).index) 
        collist_noise=[extrcolstr+'a_CR_noise'+extrcolstr+str(tt) for tt in ind_temp_noise]
        spkdf_noise_a_cr_temp = pd.DataFrame(SUspk_df_temp.loc[ind_temp_noise]['spkRateNoise'].values.reshape(1,-1),columns=collist_noise) # rowxcol: cls X trials
        collist_noise=[extrcolstr+'v_CR_noise'+extrcolstr+str(tt) for tt in ind_temp_noise]
        spkdf_noise_v_cr_temp2 = pd.DataFrame(SUspk_df_temp.loc[ind_temp_noise]['spkRateNoise'].values.reshape(1,-1),columns=collist_noise) # rowxcol: cls X trials

        # audio noise trial (FA),
        ind_temp_noise = list(SUspk_df_temp[SUspk_df_temp['respLabel']=='FAa'].index) 
        collist_noise=[extrcolstr+'a_FA_noise'+extrcolstr+str(tt) for tt in ind_temp_noise]
        spkdf_noise_a_fa_temp = pd.DataFrame(SUspk_df_temp.loc[ind_temp_noise]['spkRateSig'].values.reshape(1,-1),columns=collist_noise) # rowxcol: cls X trials

        # assume cluster have same order of trials for each category, will report error if trial order doesn't matach across clusters
        spkdf_sig = pd.concat([spkdf_sig,spkdf_sig_cat_temp],axis=0)
        spkdf_noise_a = pd.concat([spkdf_noise_a,spkdf_noise_a_cr_temp],axis=0)
        # spkdf_noise_v = pd.concat((spkdf_noise_v,spkdf_noise_v_cr_temp),axis=0)
        spkdf_noise_v = pd.concat([spkdf_noise_v,spkdf_noise_v_cr_temp2],axis=0)

    spkdf_sig_a = spkdf_sig[[col for col in spkdf_sig.columns if '*a_' in col]]
    spkdf_sig_av = spkdf_sig[[col for col in spkdf_sig.columns if '*av_' in col]]
    print('total: spkdf_sig_a'+str(spkdf_sig_a.shape)+'  spkdf_noise_a'+str(spkdf_noise_a.shape)+' spkdf_sig_av'+str(spkdf_sig_av.shape)+'  spkdf_noise_v'+str(spkdf_noise_v.shape))       

    #need to balance NoiseSigTrials for bayesian theorom, randomly pick noise trials for each sig trial    
    noisepicka = np.sort(np.random.choice(range(spkdf_noise_a.shape[1]),size=spkdf_sig_a.shape[1],replace=False))
    noisepickv = np.sort(np.random.choice(range(spkdf_noise_v.shape[1]),size=spkdf_sig_av.shape[1],replace=False))
    spkdf_noise_a = spkdf_noise_a.iloc[:,noisepicka]
    spkdf_noise_v = spkdf_noise_v.iloc[:,noisepickv]
    print('afterbalancing: spkdf_sig_a'+str(spkdf_sig_a.shape)+'  spkdf_noise_a'+str(spkdf_noise_a.shape)+' spkdf_sig_av'+str(spkdf_sig_av.shape)+'  spkdf_noise_v'+str(spkdf_noise_v.shape))       
 
    # # check data distribution
    # for cc,cls in enumerate(AllSUspk_df['sess_cls'].unique()):
    #     fig, axess = plt.subplots(2,2,figsize=(8,8),sharex='col') # 
    #     axess[0,0].hist(spkdf_sig_a.iloc[cc,:].values[0],bins=15,label='sig_a') 
    #     axess[0,1].hist(spkdf_noise_a.iloc[cc,:].values[0],bins=15,label='noise_a')
    #     axess[1,0].hist(spkdf_sig_av.iloc[cc,:].values[0],bins=15,label='sig_av') 
    #     axess[1,1].hist(spkdf_noise_v.iloc[cc,:].values[0],bins=15,label='noise_v')        
    #     fig.tight_layout()
    #     fig.savefig(figSavePath+cls+'fr_distribution.png')
    #     plt.close(fig)
    
    if method=='MGD':  
        fitacc_nNueron_sig_a = np.empty((len(Nnlist),clsSamp,spkdf_sig_a.shape[1],1)) # Nneurons X clsSamp X trials X 1
        fitacc_nNueron_sig_av = np.empty((len(Nnlist),clsSamp,spkdf_sig_av.shape[1],1)) # Nneurons X clsSamp X trials X 1
        fitacc_nNueron_noise_a = np.empty((len(Nnlist),clsSamp,spkdf_noise_a.shape[1],1)) # Nneurons X clsSamp X trials X 1 
        fitacc_nNueron_noise_v = np.empty((len(Nnlist),clsSamp,spkdf_noise_v.shape[1],1)) # Nneurons X clsSamp X trials X 1   

        fitacc_nNueron_sig_a,fitacc_nNueron_noise_a,\
            fitacc_nNueron_sig_av,fitacc_nNueron_noise_v,test_cat = applyMGD(spkdf_sig_a,
                                                                                spkdf_noise_a,
                                                                                fitacc_nNueron_sig_a,
                                                                                fitacc_nNueron_noise_a,
                                                                                spkdf_sig_av,
                                                                                spkdf_noise_v,
                                                                                fitacc_nNueron_sig_av,
                                                                                fitacc_nNueron_noise_v,
                                                                                clsSamp,clsall,Nnlist,0)
        print('sig_a hit before concatenate '+str(np.sum(fitacc_nNueron_sig_a,axis=(1,2,3))))
        print('sig_av hit before concatenate '+str(np.sum(fitacc_nNueron_sig_av,axis=(1,2,3))))

        print('noise_a cr before concatenate '+str(np.sum(fitacc_nNueron_noise_a,axis=(1,2,3))))
        print('noise_v cr before concatenate '+str(np.sum(fitacc_nNueron_noise_v,axis=(1,2,3))))

        fitacc_nNueron_dict['fitacc_nNeuron'] = np.concatenate((fitacc_nNueron_sig_a,fitacc_nNueron_noise_a,fitacc_nNueron_sig_av,fitacc_nNueron_noise_v),axis=2)
        fitacc_nNueron_dict['trialLabel'] = test_cat
    if method == 'svm':
        fitacc_nNueron_sig_a = np.empty((len(Nnlist),clsSamp,1,1)) # Nneurons X clsSamp X 1 X 1
        fitacc_nNueron_sig_av = np.empty((len(Nnlist),clsSamp,1,1)) # Nneurons X clsSamp X 1 X 1
        fitacc_nNueron_sig_a,fitacc_nNueron_sig_av = applySVMD(spkdf_sig_a,
                                                                spkdf_noise_a,
                                                                fitacc_nNueron_sig_a,
                                                                spkdf_sig_av,
                                                                spkdf_noise_v,
                                                                fitacc_nNueron_sig_av,
                                                                clsSamp,clsall,Nnlist,0)

        fitacc_nNueron_dict['fitacc_nNeuron'] = np.concatenate((fitacc_nNueron_sig_a,fitacc_nNueron_sig_av),axis=2)
        fitacc_nNueron_dict['trialLabel'] = ['a','av']
    print(fitacc_nNueron_dict['fitacc_nNeuron'].shape)
    return fitacc_nNueron_dict 


def svmdecoder(xx_all,yy_all,method,traintestgrouplist = [],cpusSVM=cpus,kernel=['rbf']):
    fitresultdict = dict()

    if len(traintestgrouplist) == 0:
        ind = BalanceSamples(yy_all) # balance based on the cat with minimum trials
        xx = xx_all[ind]
        yy = yy_all[ind]        
        # split datasets
        X_train, X_test, y_train, y_test = train_test_split(xx, np.ravel(yy), test_size=0.2,random_state = 42,stratify=np.ravel(yy))    
        print('training trials: ' + str(X_train.shape[0])+'   testing trials: '+ str(X_test.shape[0]))
    else:
        X_train = traintestgrouplist[0]
        X_test = traintestgrouplist[1]
        y_train = traintestgrouplist[2]
        y_test = traintestgrouplist[3]

    # generating aligned bootstrap dim reduction components and visualization
    if len(method)>0:
        if method['method'] =='DR_pca':
            # Instantiate the PCA object 
            pca = PCA()
            # Calculate the principal components
            principal_components = pca.fit_transform(X_train)
            # Get the amount of variance explained by each principal component
            explained_variances = pca.explained_variance_ratio_
            # Set the minimum variance threshold
            min_var = 0.8
            # Calculate the number of components required to explain at least min_var amount of variance
            num_components = np.where(np.cumsum(explained_variances) >= min_var)[0][0] + 1
            # Instantiate a new PCA object with the desired number of components
            pca = PCA(n_components=num_components)

            # Fit the PCA model and transform the data
            X_train = pca.fit_transform(X_train)
            X_test = pca.fit_transform(X_test)

    kfolds = StratifiedKFold(n_splits=5)
    # # defining hyperparameter range
    if kernel[0]=='rbf':
        param_grid = {'C': np.logspace(-5,4,num=15,base=10), 
                    'gamma': np.logspace(-3,4,num=10,base=10),
                    'kernel': kernel}  
        gridsearch = GridSearchCV(SVC(decision_function_shape = 'ovr'), 
                                param_grid, scoring = 'accuracy' ,
                                n_jobs = cpusSVM,refit = 'accuracy', 
                                cv = kfolds.split(X_train, y_train), verbose = 0)          
    if kernel[0]=='linear':  
        param_grid = {'C': np.logspace(-5,4,num=15,base=10)}    
        gridsearch = GridSearchCV(LinearSVC(penalty='l2', loss='squared_hinge',dual=False,max_iter=5000), 
                                param_grid, scoring = 'accuracy' ,
                                n_jobs = cpusSVM,refit = 'accuracy', 
                                cv = kfolds.split(X_train, y_train), verbose = 0)                
 

    # fitting the model for grid search
    gridsearch.fit(X_train, y_train)
    # print('Best Gridsearch svm parameters in a dict:')
    # print(gridsearch.best_params_)
    y_pred = gridsearch.predict(X_test) 
    if kernel[0]=='linear':
        best_svm = gridsearch.best_estimator_
        fitresultdict['bestcoefs'] = best_svm.coef_
        # print('best_svm.coef_')
        # print(best_svm.coef_)
    if kernel[0]=='rbf':
        fitresultdict['bestcoefs'] = np.zeros((1,X_train.shape[1]))

    # classification metrics classification_report(y_test, y_pred, output_dict=True, target_names = target_names)
    fitresultdict['accuracy_score'] = accuracy_score(y_test, y_pred)
    fitresultdict['accuracy_chance'] = np.max(np.unique(y_test,return_counts=True)[1]/np.sum(np.unique(y_test,return_counts=True)[1]))
    
    return fitresultdict
         
def bstrialsfromeachcls(AllSUspk_df,mintrialNum_sig,extrcolstr,seeds=42):
    #balance num of trials in each signal category based on the mimtrialNum across cls
    spkdf_sig = pd.DataFrame()
    spkdf_noise_a = pd.DataFrame()
    spkdf_noise_v = pd.DataFrame()
    for cc,cls in enumerate(AllSUspk_df['sess_cls'].unique()):                
        SUspk_df_temp =  AllSUspk_df[AllSUspk_df['sess_cls']==cls]
        SUspk_df_temp.reset_index(drop=True,inplace=True) 
        # print('order of neurons in bootstraping: #'+str(SUspk_df_temp['clsNum'].unique())) 

        # shuffle trials with same snr,trialmod,resp to remove trial by trial correlations, if pool neurons across sessions
        # resample trials in each condition with replacement
        SUspk_df_temp_sig = SUspk_df_temp[SUspk_df_temp['respLabel'].isin(['hit','miss'])].reset_index(drop=True)    
        grouped = SUspk_df_temp_sig.groupby(by=['trialMod','snr-shift'])# the category group should appear the same order when loop across cls
        group_indices = {key: grouped.groups[key].tolist() for key in grouped.groups}
        spkdf_sig_cat_temp = pd.DataFrame()
        cc2 = 0
        for (gkey,gsnr),glist in group_indices.items():            
            if any(substr in gkey for substr in ['a','av']):# a/av signal trial 
                rng = np.random.default_rng(int(seeds+cc*100+cc2))
                # print(rng.choice(range(20),size=4,replace=False))
                ind_temp_sig = rng.choice(glist,size=mintrialNum_sig[gkey][mintrialNum_sig[gkey]['snr-shift']==gsnr].iloc[0,-1],replace=False) # randomly pick num of trials in each category without replacements for signal
                collist_sig=[extrcolstr+gkey+'_'+str(gsnr)+'_sig'+extrcolstr+str(tt) for tt in range(len(ind_temp_sig))]
                spkdf_sig_cat_temp = pd.concat([spkdf_sig_cat_temp,pd.DataFrame(SUspk_df_temp_sig.loc[ind_temp_sig]['spkRateSig'].values.reshape(1,-1),columns=collist_sig)],axis=1) # rowxcol: cls X trials
                cc2 = cc2+1

        # visual noise trial, code should only run one time for each cls
        rng = np.random.default_rng(int(seeds+cc*100+cc2+1))    
        ind_temp_sig = rng.choice(list(SUspk_df_temp[(SUspk_df_temp['trialMod']=='v')&(SUspk_df_temp['respLabel']=='CR')].index)
                                        ,size=mintrialNum_sig['vnoise'],replace=False) # randomly pick num of trials in each category without replacements for signal
        collist_sig=[extrcolstr+'v_CR_noise'+extrcolstr+str(tt) for tt in range(len(ind_temp_sig))]                
        spkdf_noise_v_cr_temp = pd.DataFrame(SUspk_df_temp.loc[ind_temp_sig]['spkRateSig'].values.reshape(1,-1),columns=collist_sig)                    
        # audio noise trial (CR), same num of sample as visual noise trial, some trial the spkRateNoise could be NaN 
        rng = np.random.default_rng(int(seeds+cc*100+cc2+2))  
        ind_temp_noise = rng.choice(list(SUspk_df_temp.dropna(subset=['spkRateNoise']).index),size=mintrialNum_sig['anoise'],replace=False)  
        collist_noise=[extrcolstr+'a_CR_noise'+extrcolstr+str(tt) for tt in range(len(ind_temp_noise))]
        spkdf_noise_a_cr_temp = pd.DataFrame(SUspk_df_temp.loc[ind_temp_noise]['spkRateNoise'].values.reshape(1,-1),columns=collist_noise) # rowxcol: 1 X trials
        collist_noise=[extrcolstr+'v_CR_noise'+extrcolstr+str(tt) for tt in range(len(ind_temp_noise))]
        spkdf_noise_v_cr_temp2 = pd.DataFrame(SUspk_df_temp.loc[ind_temp_noise]['spkRateNoise'].values.reshape(1,-1),columns=collist_noise) # rowxcol: 1 X trials
                
        # signal and noise from different trials are saved 
        # assume all sessions have same trial cetegories, will report error if a session don't have all the categories
        spkdf_sig = pd.concat([spkdf_sig,spkdf_sig_cat_temp],axis=0) #rowxcol: cls X trials
        spkdf_noise_a = pd.concat([spkdf_noise_a,spkdf_noise_a_cr_temp],axis=0) #rowxcol: cls X trials
        spkdf_noise_v = pd.concat([spkdf_noise_v,spkdf_noise_v_cr_temp2],axis=0) #rowxcol: cls X trials
        # spkdf_noise_a = pd.concat([spkdf_noise_a,spkdf_noise_a_fa_temp],axis=0)
        # spkdf_noise_v = pd.concat([spkdf_noise_v,spkdf_noise_v_fa_temp],axis=0)  

    spkdf_sig_a = spkdf_sig[[col for col in spkdf_sig.columns if '*a_' in col]]
    spkdf_sig_av = spkdf_sig[[col for col in spkdf_sig.columns if '*av_' in col]]        
    print('total: spkdf_sig_a'+str(spkdf_sig_a.shape)+'  spkdf_noise_a'+str(spkdf_noise_a.shape)+' spkdf_sig_av'+str(spkdf_sig_av.shape)+'  spkdf_noise_v'+str(spkdf_noise_v.shape))       

    #need to balance NoiseSigTrials for bayesian theorom, randomly pick noise trials for a and av cond separately 
    rng = np.random.default_rng(int(seeds+100000)) 
    noisepicka = np.sort(rng.choice(range(spkdf_noise_a.shape[1]),size=spkdf_sig_a.shape[1],replace=False))
    rng = np.random.default_rng(int(seeds+100001))  
    noisepickv = np.sort(rng.choice(range(spkdf_noise_v.shape[1]),size=spkdf_sig_av.shape[1],replace=False))
    spkdf_noise_a_sub = spkdf_noise_a.iloc[:,noisepicka]
    spkdf_noise_v_sub = spkdf_noise_v.iloc[:,noisepickv]
    print('afterbalancing: spkdf_sig_a'+str(spkdf_sig_a.shape)+'  spkdf_noise_a'+str(spkdf_noise_a_sub.shape)+' spkdf_sig_av'+str(spkdf_sig_av.shape)+'  spkdf_noise_v'+str(spkdf_noise_v_sub.shape))       
    return spkdf_sig_a,spkdf_sig_av,spkdf_noise_a_sub,spkdf_noise_v_sub

def decoder_detection(AllSUspk_df,method,CatunitIndlist=[],kernel=['rbf']): # randomly pick trials from each cluster across sessions, with replacement; use part of clusters, parallel sampling clusters process
    clsall = list(AllSUspk_df.sess_cls.unique())
    print('clsall'+str(len(clsall)))
    # Nnlist = [2]*2#
    # clsSamp = 1
    # bstimes = 2

    # # # uncomment this when using decoderSPKMain_populations_randompick.py
    # # Nnlist = np.arange(1,int(len(clsall)/2),int(len(clsall)/20))# possible num of neurons to compare
    # Nnlist = np.arange(1,int(len(clsall)/2),5)# possible num of neurons to compare
    # clsSamp = 20 # times randomly pick cls
    # bstimes = 1

    # uncomment this when using decoderSPKMain_populations_orderedUnits.py 
    Nnlist = np.arange(1,int(len(clsall)),int(len(clsall)/10))# possible num of neurons to compare
    clsSamp = 1
    bstimes = 20 # times to randomly select mintrialNum samples in each category+cls


    extrcolstr='*'
    fitacc_nNueron_dict = {}
    # print(AllSUspk_df.groupby(by=['sess_cls','trialMod','snr-shift','respLabel']).size().reset_index().to_string())
    mintrialNum_sig = {} # get the min num of trials in each condition trialmod+snr
    mintrialNum_sig['a'] = AllSUspk_df[(AllSUspk_df['trialMod']=='a')&(AllSUspk_df['respLabel'].isin(['hit','miss']))].groupby(by=['sess_cls','snr-shift']).size().reset_index().groupby('snr-shift').min().reset_index()
    mintrialNum_sig['av'] = AllSUspk_df[(AllSUspk_df['trialMod']=='av')&(AllSUspk_df['respLabel'].isin(['hit','miss']))].groupby(by=['sess_cls','snr-shift']).size().reset_index().groupby('snr-shift').min().reset_index()
    mintrialNum_sig['vnoise'] = AllSUspk_df[AllSUspk_df['respLabel'].isin(['CR'])].groupby(by=['sess_cls']).size().reset_index().iloc[:,-1].values.min()
    mintrialNum_sig['anoise'] = AllSUspk_df.dropna(subset=['spkRateNoise']).groupby(by=['sess_cls']).size().reset_index().iloc[:,-1].values.min()
    print(mintrialNum_sig)

    for bs in range(bstimes):# bootstrape to maximize the trial usage in sig categories with more than mimtrialNum samples
        print('decoding in bootstrape '+str(bs)+'.................')
        time1 = time.monotonic()
        # randomly pick equal numbers of trials from each category
        spkdf_sig_a,spkdf_sig_av,spkdf_noise_a,spkdf_noise_v = bstrialsfromeachcls(AllSUspk_df,mintrialNum_sig,extrcolstr,seeds=np.random.choice(range(10000), size=1, replace=False))        
        
        # decoder_SVMdetection(spkdf_sig_a,spkdf_noise_a,clsSamp,clsall,0,1,[],3,kernel=['linear'])

        if method=='MGD':   
            # initialize result arrays here, because only know noise trials after balancing 
            if bs==0:
                fitacc_nNueron_sig_a = np.empty((len(Nnlist),clsSamp,int(mintrialNum_sig['a'].iloc[:,-1].sum()),bstimes)) # Nneurons X clsSamp X trials X bootstrapTimes 
                fitacc_nNueron_sig_av = np.empty((len(Nnlist),clsSamp,int(mintrialNum_sig['av'].iloc[:,-1].sum()),bstimes)) # Nneurons X clsSamp X trials X bootstrapTimes 
                fitacc_nNueron_noise_a = np.empty((len(Nnlist),clsSamp,spkdf_noise_a.shape[1],bstimes)) # Nneurons X clsSamp X trials X bootstrapTimes 
                fitacc_nNueron_noise_v = np.empty((len(Nnlist),clsSamp,spkdf_noise_v.shape[1],bstimes)) # Nneurons X clsSamp X trials X bootstrapTimes 

            fitacc_nNueron_sig_a,fitacc_nNueron_noise_a,\
                fitacc_nNueron_sig_av,fitacc_nNueron_noise_v,test_cat = applyMGD(spkdf_sig_a,
                                                                                 spkdf_noise_a,
                                                                                 fitacc_nNueron_sig_a,
                                                                                 fitacc_nNueron_noise_a,
                                                                                 spkdf_sig_av,
                                                                                 spkdf_noise_v,
                                                                                 fitacc_nNueron_sig_av,
                                                                                 fitacc_nNueron_noise_v,
                                                                                 clsSamp,clsall,Nnlist,bs,CatunitIndlist)
            if bs==bstimes-1:
                print('sig_a hit before concatenate '+str(np.sum(fitacc_nNueron_sig_a,axis=(1,2,3))))
                print('sig_av hit before concatenate '+str(np.sum(fitacc_nNueron_sig_av,axis=(1,2,3))))
                print('noise_a cr before concatenate '+str(np.sum(fitacc_nNueron_noise_a,axis=(1,2,3))))
                print('noise_v cr before concatenate '+str(np.sum(fitacc_nNueron_noise_v,axis=(1,2,3))))
                fitacc_nNueron_dict['fitacc_nNeuron'] = np.concatenate((fitacc_nNueron_sig_a,fitacc_nNueron_noise_a,fitacc_nNueron_sig_av,fitacc_nNueron_noise_v),axis=2)
                fitacc_nNueron_dict['trialLabel'] = test_cat
                print(fitacc_nNueron_dict['fitacc_nNeuron'].shape)
        if method == 'svm':
            if bs ==0:
                fitacc_nNueron_sig_a = np.empty((len(Nnlist),clsSamp,1,bstimes)) # Nneurons X clsSamp X 1 X bootstrapTimes 
                fitacc_nNueron_sig_av = np.empty((len(Nnlist),clsSamp,1,bstimes)) # Nneurons X clsSamp X 1 X bootstrapTimes 
                fitacc_nNueron_sig_a_bsline = np.empty((len(Nnlist),clsSamp,1,bstimes)) # Nneurons X clsSamp X 1 X bootstrapTimes 
                fitacc_nNueron_sig_av_bsline = np.empty((len(Nnlist),clsSamp,1,bstimes)) # Nneurons X clsSamp X 1 X bootstrapTimes 

            fitacc_nNueron_sig_a,fitacc_nNueron_sig_av,\
                fitacc_nNueron_sig_a_bsline,fitacc_nNueron_sig_av_bsline = applySVMD(spkdf_sig_a,
                                                                    spkdf_noise_a,
                                                                    fitacc_nNueron_sig_a,
                                                                    fitacc_nNueron_sig_a_bsline,
                                                                    spkdf_sig_av,
                                                                    spkdf_noise_v,
                                                                    fitacc_nNueron_sig_av,
                                                                    fitacc_nNueron_sig_av_bsline,
                                                                    clsSamp,clsall,Nnlist,bs,CatunitIndlist,kernel)
            if bs==bstimes-1:
                fitacc_nNueron_dict['fitacc_nNeuron'] = np.concatenate((fitacc_nNueron_sig_a,fitacc_nNueron_sig_av),axis=2)
                fitacc_nNueron_dict['fitacc_nNeuron_bsline'] = np.concatenate((fitacc_nNueron_sig_a_bsline,fitacc_nNueron_sig_av_bsline),axis=2)
                fitacc_nNueron_dict['trialLabel'] = ['A','AV']
                print(fitacc_nNueron_dict['fitacc_nNeuron'].shape)
        time2 = time.monotonic()
        print('time spend to decode one time bootstrap '+str(bs))
        print(timedelta(seconds=time2 - time1)) 
    return fitacc_nNueron_dict  

def decoder_detection_v2(AllSUspk_df,method,CatunitIndlist=[],bstimes=40,clsSamp=1,kernel=['rbf'],Nnlist = []): # randomly pick trials from each cluster across sessions, with replacement; use all clusters, parallel bootstrap process
    clsall = list(AllSUspk_df.sess_cls.unique())
    print('clsall'+str(len(clsall)))
    if len(Nnlist)==0:       
        Nnlist = [int(len(clsall))]
    
    # clsSamp = 1
    # bstimes = 40 # times to randomly select mintrialNum samples in each category+cls

    extrcolstr='*'
    fitacc_nNueron_dict = {}
    # print(AllSUspk_df[AllSUspk_df['respLabel'].isin(['hit','miss'])].groupby(by=['sess_cls','trialMod','snr-shift']).size().reset_index().to_string())
    mintrialNum_sig = {} # get the min num of trials in each condition trialmod+snr
    mintrialNum_sig['a'] = AllSUspk_df[(AllSUspk_df['trialMod']=='a')&(AllSUspk_df['respLabel'].isin(['hit','miss']))].groupby(by=['sess_cls','snr-shift']).size().reset_index().groupby('snr-shift').min().reset_index()
    mintrialNum_sig['av'] = AllSUspk_df[(AllSUspk_df['trialMod']=='av')&(AllSUspk_df['respLabel'].isin(['hit','miss']))].groupby(by=['sess_cls','snr-shift']).size().reset_index().groupby('snr-shift').min().reset_index()
    mintrialNum_sig['vnoise'] = AllSUspk_df[AllSUspk_df['respLabel'].isin(['CR'])].groupby(by=['sess_cls']).size().reset_index().iloc[:,-1].values.min()
    mintrialNum_sig['anoise'] = AllSUspk_df.dropna(subset=['spkRateNoise']).groupby(by=['sess_cls']).size().reset_index().iloc[:,-1].values.min()
    print(mintrialNum_sig)

    # initialize arrays to save decoding results
    print('Initializing arrays for results save..............')
    if method=='MGD':  
        # test run to get num of a and av noise trials after balancing  
        _,_,spkdf_noise_ai,spkdf_noise_vi = bstrialsfromeachcls(AllSUspk_df,mintrialNum_sig,extrcolstr)
        # initialize result arrays here, because only know noise trials after balancing 
        fitacc_nNueron_sig_a = np.empty((len(Nnlist),clsSamp,int(mintrialNum_sig['a'].iloc[:,-1].sum()),bstimes)) # Nneurons X clsSamp X trials X bootstrapTimes 
        fitacc_nNueron_sig_av = np.empty((len(Nnlist),clsSamp,int(mintrialNum_sig['av'].iloc[:,-1].sum()),bstimes)) # Nneurons X clsSamp X trials X bootstrapTimes 
        fitacc_nNueron_noise_a = np.empty((len(Nnlist),clsSamp,spkdf_noise_ai.shape[1],bstimes)) # Nneurons X clsSamp X trials X bootstrapTimes 
        fitacc_nNueron_noise_v = np.empty((len(Nnlist),clsSamp,spkdf_noise_vi.shape[1],bstimes)) # Nneurons X clsSamp X trials X bootstrapTimes 
    if method == 'svm':
        fitacc_nNueron_sig_a = np.empty((len(Nnlist),clsSamp,1,bstimes)) # Nneurons X clsSamp X 1 X bootstrapTimes 
        fitacc_nNueron_sig_av = np.empty((len(Nnlist),clsSamp,1,bstimes)) # Nneurons X clsSamp X 1 X bootstrapTimes 
        fitacc_nNueron_sig_a_bsline = np.empty((len(Nnlist),clsSamp,1,bstimes)) # Nneurons X clsSamp X 1 X bootstrapTimes 
        fitacc_nNueron_sig_av_bsline = np.empty((len(Nnlist),clsSamp,1,bstimes)) # Nneurons X clsSamp X 1 X bootstrapTimes 
        fit_coefs_sig_a = pd.DataFrame()
        fit_coefs_sig_av = pd.DataFrame()

    print('Start boostrape decoding.......')
    argItems = [(AllSUspk_df,mintrialNum_sig,extrcolstr,bs,method,clsSamp,clsall,Nnlist,CatunitIndlist,seeds,kernel) for (bs,seeds) in zip(range(bstimes),np.random.choice(range(10000), size=bstimes, replace=False))]
    with Pool(processes=cpus) as p:
        # get the decode result for a bootstraped trials set
        for (fitacc_nNueron_sig_a_temp,fitacc_nNueron_noise_a_temp,fitacc_nNueron_sig_av_temp,fitacc_nNueron_noise_v_temp,
             fitacc_nNueron_sig_a_temp_sh,fitacc_nNueron_sig_av_temp_sh,fit_coefs_sig_a_temp,fit_coefs_sig_av_temp,test_cat,bs) in p.starmap(decoder1roundBStrials,argItems):
            fitacc_nNueron_sig_a[:,:,:,bs] = fitacc_nNueron_sig_a_temp.reshape((-1,1))            
            fitacc_nNueron_sig_av[:,:,:,bs] = fitacc_nNueron_sig_av_temp.reshape((-1,1)) 
            fitacc_nNueron_sig_a_bsline[:,:,:,bs] = fitacc_nNueron_sig_a_temp_sh.reshape((-1,1)) 
            fitacc_nNueron_sig_av_bsline[:,:,:,bs] = fitacc_nNueron_sig_av_temp_sh.reshape((-1,1)) 
            fit_coefs_sig_a_temp['bs'] = bs
            fit_coefs_sig_a_temp['mod'] = 'A'
            fit_coefs_sig_av_temp['bs'] = bs
            fit_coefs_sig_av_temp['mod'] = 'AV'
            fit_coefs_sig_a = pd.concat([fit_coefs_sig_a,fit_coefs_sig_a_temp],axis=0)
            fit_coefs_sig_av = pd.concat([fit_coefs_sig_av,fit_coefs_sig_av_temp],axis=0)
            try:
                fitacc_nNueron_noise_a[:,:,:,bs] = fitacc_nNueron_noise_a_temp.reshape((-1,1)) 
                fitacc_nNueron_noise_v[:,:,:,bs] = fitacc_nNueron_noise_v_temp.reshape((-1,1))              
            except (UnboundLocalError,AttributeError) as e:
                pass
    p.close()
    p.join() 

    # package all results into one dictionary
    if method=='MGD':
        print('sig_a hit before concatenate '+str(np.sum(fitacc_nNueron_sig_a,axis=(1,2,3))))
        print('sig_av hit before concatenate '+str(np.sum(fitacc_nNueron_sig_av,axis=(1,2,3))))
        print('noise_a cr before concatenate '+str(np.sum(fitacc_nNueron_noise_a,axis=(1,2,3))))
        print('noise_v cr before concatenate '+str(np.sum(fitacc_nNueron_noise_v,axis=(1,2,3))))
        fitacc_nNueron_dict['fitacc_nNeuron'] = np.concatenate((fitacc_nNueron_sig_a,fitacc_nNueron_noise_a,fitacc_nNueron_sig_av,fitacc_nNueron_noise_v),axis=2)
        fitacc_nNueron_dict['trialLabel'] = test_cat
        print(fitacc_nNueron_dict['fitacc_nNeuron'].shape)
    if method == 'svm':
        fitacc_nNueron_dict['fitacc_nNeuron'] = np.concatenate((fitacc_nNueron_sig_a,fitacc_nNueron_sig_av),axis=2)
        fitacc_nNueron_dict['fitacc_nNeuron_bsline'] = np.concatenate((fitacc_nNueron_sig_a_bsline,fitacc_nNueron_sig_av_bsline),axis=2)
        fitacc_nNueron_dict['fit_coefs'] = pd.concat([fit_coefs_sig_a,fit_coefs_sig_av],axis=0)
        fitacc_nNueron_dict['trialLabel'] = test_cat
        print(fitacc_nNueron_dict['fitacc_nNeuron'].shape)
 
    return fitacc_nNueron_dict  
           
def decoder1roundBStrials(AllSUspk_df,mintrialNum_sig,extrcolstr,bs,method,\
                            clsSamp,clsall,Nnlist,CatunitIndlist,seeds,kernel=['rbf']):
    print('decoding in bootstrape '+str(bs)+'.................')
    time1 = time.monotonic()
    # bootstrape to maximize the trial usage in sig categories with more than mimtrialNum samples
    spkdf_sig_a,spkdf_sig_av,spkdf_noise_a,spkdf_noise_v = bstrialsfromeachcls(AllSUspk_df,mintrialNum_sig,extrcolstr,seeds)

    if method=='MGD':   
        # decoding A and AV conditions 
        fitacc_nNueron_sig_a_temp,fitacc_nNueron_noise_a_temp,spk_test_cat_sig_a,spk_test_cat_noise_a,_ = decoder1round_MGDdetection(spkdf_sig_a,spkdf_noise_a,clsSamp,clsall,0,Nnlist[0],CatunitIndlist,seeds)
        fitacc_nNueron_sig_av_temp,fitacc_nNueron_noise_v_temp,spk_test_cat_sig_av,spk_test_cat_noise_av,_ = decoder1round_MGDdetection(spkdf_sig_av,spkdf_noise_v,clsSamp,clsall,0,Nnlist[0],CatunitIndlist,seeds)
        # concatenate all trial labels
        test_cat = spk_test_cat_sig_a+spk_test_cat_noise_a+spk_test_cat_sig_av+spk_test_cat_noise_av
        print('number of nan in the fitacc_nNueron matrix: sig-a'+str(np.sum(np.isnan(fitacc_nNueron_sig_a_temp)))+' sig_av'+str(np.sum(np.isnan(fitacc_nNueron_sig_av_temp)))+
                ' noise-a'+str(np.sum(np.isnan(fitacc_nNueron_noise_a_temp)))+' noise-v'+str(np.sum(np.isnan(fitacc_nNueron_noise_v_temp))))           
        print('number of 1s in the fitacc_nNueron matrix: sig-a'+str(np.sum(fitacc_nNueron_sig_a_temp))+' sig_av'+str(np.sum(fitacc_nNueron_sig_av_temp))+
                ' noise-a'+str(np.sum(fitacc_nNueron_noise_a_temp))+' noise-v'+str(np.sum(fitacc_nNueron_noise_v_temp)))           

    if method == 'svm':
        # decoding A and AV conditions 
        fitacc_nNueron_sig_a_temp,fitacc_nNueron_sig_a_temp_sh,fit_coefs_sig_a_temp,_ = decoder_SVMdetection(spkdf_sig_a,spkdf_noise_a,clsSamp,clsall,0,Nnlist[0],CatunitIndlist,seeds,kernel)
        fitacc_nNueron_sig_av_temp,fitacc_nNueron_sig_av_temp_sh,fit_coefs_sig_av_temp,_ = decoder_SVMdetection(spkdf_sig_av,spkdf_noise_v,clsSamp,clsall,0,Nnlist[0],CatunitIndlist,seeds,kernel)
        test_cat = ['A','AV']
        fitacc_nNueron_noise_a_temp = np.nan
        fitacc_nNueron_noise_v_temp = np.nan

    time2 = time.monotonic()
    print('time spend to decode one time bootstrap '+str(bs))
    print(timedelta(seconds=time2 - time1)) 
    return fitacc_nNueron_sig_a_temp,fitacc_nNueron_noise_a_temp,\
        fitacc_nNueron_sig_av_temp,fitacc_nNueron_noise_v_temp,\
    fitacc_nNueron_sig_a_temp_sh,fitacc_nNueron_sig_av_temp_sh,\
        fit_coefs_sig_a_temp,fit_coefs_sig_av_temp,test_cat,bs

def decoder_SVMdetection(spkdf_1,spkdf_2,clsSamp,clsall,tNn,Nn,CatunitIndlist,seeds,kernel):
    fitacc_nNueron = np.empty((clsSamp,)) # clsSamp
    fitacc_nNueron_sh = np.empty((clsSamp,)) # clsSamp
    fit_coefs = pd.DataFrame() 
    xx = np.transpose(pd.concat((spkdf_1,spkdf_2),axis=1).values) # trials X cls
    yy = np.array([0]*spkdf_1.shape[1]+[1]*spkdf_2.shape[1]).reshape((-1,1))

    # shuffle yy lables to get baseline accuracy, switch half sig to noise, half noise to sig randomly
    yy_1 = np.array([0]*spkdf_1.shape[1])
    yy_2 = np.array([1]*spkdf_2.shape[1])
    rng = np.random.default_rng(seeds+3000)
    Indshuffle = rng.choice(range(len(yy_1)),size=int(len(yy_1)/2),replace=False)
    yy_1[Indshuffle] = 1
    
    rng = np.random.default_rng(seeds+4000)
    Indshuffle = rng.choice(range(len(yy_2)),size=int(len(yy_2)/2),replace=False)
    yy_2[Indshuffle] = 0  
    print('yy_1:'+str(np.sum(yy_1))+'   yy_2:'+str(np.sum(yy_2)))
    yy_shuffle = np.concatenate((yy_1.reshape((-1,1)),yy_2.reshape((-1,1))),axis=0)
    # print(yy_shuffle)

    # if trials have been  balanced between signal and noise
    X_train, X_test, y_train, y_test = train_test_split(xx, np.ravel(yy), test_size=0.2,stratify=np.ravel(yy))    
    X_train_sh, X_test_sh, y_train_sh, y_test_sh = train_test_split(xx, np.ravel(yy_shuffle), test_size=0.2,stratify=np.ravel(yy_shuffle))    
    
    # randomly choose subset cls without replacements                  
    for rep in range(clsSamp):  
        # sequentially pick snr, v+snr, and v modulated units in the decoder
        if len(CatunitIndlist)>0:
            if all(isinstance(i, list) for i in CatunitIndlist): # if CatunitIndlist is a list of list: save different categories of neurons modified by A,A+V,V
                print('choose units in A method')
                cls1 = CatunitIndlist[0]
                cls2=CatunitIndlist[1]
                cls3=CatunitIndlist[2]
                # pickedcls = np.random.choice(cls1+cls2+cls3, size=Nn,replace=False)            
                if Nn<=len(cls1)/2:
                    rng1 = np.random.default_rng(seeds+rep)
                    pickedcls = rng1.choice(cls1, size=Nn,replace=False) 
                elif Nn<=(len(cls1)+len(cls2))/2:
                    rng1 = np.random.default_rng(seeds+rep)
                    pickedcls1 = rng1.choice(cls1, size=int(len(cls1)/2),replace=False)
                    rng2 = np.random.default_rng(seeds+rep+1000)
                    pickedcls2 = rng2.choice(cls2, size=Nn-int(len(cls1)/2),replace=False)
                    pickedcls = np.concatenate((pickedcls1,pickedcls2))
                elif Nn<=(len(cls1)+len(cls2)+len(cls3))/2:
                    rng1 = np.random.default_rng(seeds+rep)
                    pickedcls1 = rng1.choice(cls1, size=int(len(cls1)/2),replace=False)
                    rng2 = np.random.default_rng(seeds+rep+1000)
                    pickedcls2 = rng2.choice(cls2, size=Nn-int(len(cls1)/2),replace=False)
                    rng3 = np.random.default_rng(seeds+rep+2000)
                    pickedcls3 = rng3.choice(cls3, size=Nn-int(len(cls1)/2)-int(len(cls2)/2),replace=False)
                    pickedcls = np.concatenate((pickedcls1,pickedcls2,pickedcls3))
                    
            elif all(not isinstance(i, list) for i in CatunitIndlist): # if CatunitIndlist is a list: save ordered neuron index
                pickedcls = CatunitIndlist[:Nn]
                print('choose units in ordered method')
        else:
            rng1 = np.random.default_rng(seeds+rep)
            pickedcls = rng1.choice(np.arange(0,X_train.shape[1],1), size=Nn,replace=False) 
            print('choose '+str(Nn)+' units in random method')

        # decoding signal from noise
        X_train_temp = X_train[:,pickedcls] # trialsm X cls
        y_train_temp = y_train.copy()
        X_test_temp = X_test[:,pickedcls]
        y_test_temp = y_test.copy()     
        fitresultdict = svmdecoder([],[],[],[X_train_temp,X_test_temp,y_train_temp,y_test_temp],cpusSVM=1,kernel=kernel) 
        fitacc_nNueron[rep] = np.array(fitresultdict['accuracy_score'])
        # save fit coef of the best fit for each feature  
        fit_coefs_temp = pd.DataFrame({'coef':fitresultdict['bestcoefs'].tolist()[0],
                                       'coef_abs':np.abs(fitresultdict['bestcoefs']).tolist()[0],
                                       'clsNum':pickedcls.tolist(),
                                       'clsSamp':[rep]*len(pickedcls)})      
        fit_coefs = pd.concat([fit_coefs,fit_coefs_temp],axis=0)

        #  decoding shuffled signal from noise for baseline accuracy
        X_train_temp_sh = X_train_sh[:,pickedcls] # trialsm X cls
        y_train_temp_sh = y_train_sh.copy()
        X_test_temp_sh = X_test_sh[:,pickedcls]
        y_test_temp_sh = y_test_sh.copy()   
        # X_train_temp_sh = np.repeat(X_train[:,pickedcls[0]].reshape(-1,1),len(pickedcls),axis=1) # trialsm X cls
        # y_train_temp_sh = y_train.copy()
        # X_test_temp_sh = np.repeat(X_test[:,pickedcls[0]].reshape(-1,1),len(pickedcls),axis=1)
        # y_test_temp_sh = y_test.copy() 
        fitresultdict_sh = svmdecoder([],[],[],[X_train_temp_sh,X_test_temp_sh,y_train_temp_sh,y_test_temp_sh],cpusSVM=1,kernel=kernel) 
        fitacc_nNueron_sh[rep] = np.array(fitresultdict_sh['accuracy_score'])

        print('neurons selected for svm decoder')
        print(pickedcls)
        
    return fitacc_nNueron,fitacc_nNueron_sh,fit_coefs,tNn

def applySVMD(spkdf_sig_a,spkdf_noise_a,fitacc_nNueron_sig_a,fitacc_nNueron_sig_a_bsline,spkdf_sig_av,spkdf_noise_v,fitacc_nNueron_sig_av,fitacc_nNueron_sig_av_bsline,clsSamp,clsall,Nnlist,bs,CatunitIndlist,kernel=['rbf']):
    print('start detecting for sig_a and noise_a trials.....')
    argItems = [(spkdf_sig_a,spkdf_noise_a,clsSamp,clsall,tNn,Nn,CatunitIndlist['A'],seeds,kernel) for tNn,(Nn,seeds) in enumerate(zip(Nnlist,np.random.choice(range(10000), size=len(Nnlist), replace=False)))]
    with Pool(processes=cpus) as p:
        # get the decode result for a trial
        for fitacc_nNueron_trial_sig_a_temp,fitacc_nNueron_trial_sig_a_temp_sh,fit_coefs_a_temp,tNn in p.starmap(decoder_SVMdetection,argItems):
            # print('number of neurons/samprpts '+str(tNn))
            fitacc_nNueron_sig_a[tNn,:,0,bs] = fitacc_nNueron_trial_sig_a_temp #clsSamp X 0
            fitacc_nNueron_sig_a_bsline[tNn,:,0,bs] = fitacc_nNueron_trial_sig_a_temp_sh #clsSamp X 0
    p.close()
    p.join()  
    

    print('start detecting for sig_av and noise_v trials.....')
    argItems = [(spkdf_sig_av,spkdf_noise_v,clsSamp,clsall,tNn,Nn,CatunitIndlist['AV'],seeds,kernel) for tNn,(Nn,seeds) in enumerate(zip(Nnlist,np.random.choice(range(10000), size=len(Nnlist), replace=False)))]
    with Pool(processes=cpus) as p:
        # get the decode result for a trial
        for fitacc_nNueron_trial_sig_av_temp,fitacc_nNueron_trial_sig_av_temp_sh,fit_coefs_av_temp,tNn in p.starmap(decoder_SVMdetection,argItems):
            # print('number of neurons/samprpts '+str(tNn))
            fitacc_nNueron_sig_av[tNn,:,0,bs] = fitacc_nNueron_trial_sig_av_temp #clsSamp X 0
            fitacc_nNueron_sig_av_bsline[tNn,:,0,bs] = fitacc_nNueron_trial_sig_av_temp_sh #clsSamp X 0
    p.close()
    p.join()      

    return fitacc_nNueron_sig_a,fitacc_nNueron_sig_av,fitacc_nNueron_sig_a_bsline,fitacc_nNueron_sig_av_bsline

def zscore_sigNnoiseCols(AllSUspk_df_training):
    # zscore spkrate for each cluster
    colnames = list(set(AllSUspk_df_training.columns)^set(['spkRateSig','spkRateNoise']))
    AllSUspk_df_training_sig = AllSUspk_df_training[colnames].copy()
    AllSUspk_df_training_sig['sORn'] = 'Sig'
    AllSUspk_df_training_sig['spkRate'] = AllSUspk_df_training['spkRateSig']
    AllSUspk_df_training_noi = AllSUspk_df_training[colnames].copy()
    AllSUspk_df_training_noi['sORn'] = 'Noise'
    AllSUspk_df_training_noi['spkRate'] = AllSUspk_df_training['spkRateNoise']
    AllSUspk_df_training_cat = pd.concat([AllSUspk_df_training_sig,AllSUspk_df_training_noi],axis=0)
    AllSUspk_df_training_cat['spkRate'] = AllSUspk_df_training_cat.groupby('sess_cls')['spkRate'].transform(zscore)
    AllSUspk_df_training_zscore_sig = AllSUspk_df_training_cat[AllSUspk_df_training_cat['sORn']=='Sig'][colnames+['spkRate']].sort_values(by=colnames).reset_index(drop=True)
    AllSUspk_df_training_zscore_noi = AllSUspk_df_training_cat[AllSUspk_df_training_cat['sORn']=='Noise'][colnames+['spkRate']].sort_values(by=colnames).reset_index(drop=True)
    AllSUspk_df_training_zscore =AllSUspk_df_training_zscore_sig.rename(columns={'spkRate':'spkRateSig'})
    AllSUspk_df_training_zscore['spkRateNoise'] = AllSUspk_df_training_zscore_noi['spkRate']

    return AllSUspk_df_training_zscore

##############################################################
def decoder_detection_v1(AllSUspk_df,units=30,trainSample_eachCat = 200,testSample_eachCat = 50,bstimes=40,clsSamp=1,kernel=['rbf']): # randomly pick trials from each cluster across sessions, with replacement; use all clusters, parallel bootstrap process
    clsall = list(AllSUspk_df.sess_cls.unique())
    fitacc_nNueron_dict = {}

    # initialize arrays to save decoding results
    print('Initializing arrays for results save..............') 
    fitacc_nNueron_sig_a = np.empty((1,clsSamp,1,bstimes)) # Nneurons X clsSamp X 1 X bootstrapTimes 
    fitacc_nNueron_sig_av = np.empty((1,clsSamp,1,bstimes)) # Nneurons X clsSamp X 1 X bootstrapTimes 
    fitacc_nNueron_sig_a_bsline = np.empty((1,clsSamp,1,bstimes)) # Nneurons X clsSamp X 1 X bootstrapTimes 
    fitacc_nNueron_sig_av_bsline = np.empty((1,clsSamp,1,bstimes)) # Nneurons X clsSamp X 1 X bootstrapTimes 
    fit_coefs_sig_a = pd.DataFrame()
    fit_coefs_sig_av = pd.DataFrame()

    print('Start boostrape decoding.......')
    argItems = [(AllSUspk_df,bs,seeds,units,kernel,trainSample_eachCat,testSample_eachCat) for (bs,seeds) in zip(range(bstimes),np.random.choice(range(10000), size=bstimes, replace=False))]
    with Pool(processes=cpus) as p:
        # get the decode result for a bootstraped trials set
        for (fitacc_nNueron_sig_a_temp,fitacc_nNueron_noise_a_temp,fitacc_nNueron_sig_av_temp,fitacc_nNueron_noise_v_temp,
             fitacc_nNueron_sig_a_temp_sh,fitacc_nNueron_sig_av_temp_sh,fit_coefs_sig_a_temp,fit_coefs_sig_av_temp,test_cat,bs) in p.starmap(decoder1roundBStrials_v1,argItems):
            fitacc_nNueron_sig_a[:,:,:,bs] = fitacc_nNueron_sig_a_temp.reshape((-1,1))            
            fitacc_nNueron_sig_av[:,:,:,bs] = fitacc_nNueron_sig_av_temp.reshape((-1,1)) 
            fitacc_nNueron_sig_a_bsline[:,:,:,bs] = fitacc_nNueron_sig_a_temp_sh.reshape((-1,1)) 
            fitacc_nNueron_sig_av_bsline[:,:,:,bs] = fitacc_nNueron_sig_av_temp_sh.reshape((-1,1)) 
            fit_coefs_sig_a_temp['bs'] = bs
            fit_coefs_sig_a_temp['mod'] = 'A'
            fit_coefs_sig_av_temp['bs'] = bs
            fit_coefs_sig_av_temp['mod'] = 'AV'
            fit_coefs_sig_a = pd.concat([fit_coefs_sig_a,fit_coefs_sig_a_temp],axis=0)
            fit_coefs_sig_av = pd.concat([fit_coefs_sig_av,fit_coefs_sig_av_temp],axis=0)
    p.close()
    p.join() 

    # package all results into one dictionary
    fitacc_nNueron_dict['fitacc_nNeuron'] = np.concatenate((fitacc_nNueron_sig_a,fitacc_nNueron_sig_av),axis=2)
    fitacc_nNueron_dict['fitacc_nNeuron_bsline'] = np.concatenate((fitacc_nNueron_sig_a_bsline,fitacc_nNueron_sig_av_bsline),axis=2)
    fitacc_nNueron_dict['fit_coefs'] = pd.concat([fit_coefs_sig_a,fit_coefs_sig_av],axis=0)
    fitacc_nNueron_dict['trialLabel'] = test_cat
 
    return fitacc_nNueron_dict  
           
def decoder1roundBStrials_v1(AllSUspk_df,bs,seeds,units,kernel=['rbf'],trainSample_eachCat=200,testSample_eachCat=50):
    print('decoding in bootstrape '+str(bs)+'.................')
    ######## generate nonoverlap trials for training and testing set, 
    ####### activity recorded in the same trial keep in the same set
    trialNum_df = AllSUspk_df.groupby(by=['sess','trialMod','snr','respLabel','AVoffset','trialNum']).size().reset_index()
    traintrialsbysess = trialNum_df.groupby(by=['sess','trialMod','snr','respLabel','AVoffset']).sample(frac=0.7,random_state=seeds,replace=False)[['sess','trialNum']]
    Alltrialsbysess = trialNum_df[['sess','trialNum']]
    merged_df = Alltrialsbysess.merge(traintrialsbysess, on=['sess','trialNum'], how='left', indicator=True)
    testtrialsbysess = merged_df[merged_df['_merge'] == 'left_only'].drop(columns='_merge')
    #get non-overlap training and testing data from all clusters
    AllSUspk_df_training_All = AllSUspk_df.merge(traintrialsbysess,on=['sess','trialNum'],how='inner')
    AllSUspk_df_testing_All = AllSUspk_df.merge(testtrialsbysess,on=['sess','trialNum'],how='inner')

    # randomly select a set num of neurons
    np.random.seed(seeds)
    allunits = AllSUspk_df.sess_cls.unique()
    pickedunits = np.random.choice(allunits,size=units,replace=True)
    AllSUspk_df_training = pd.DataFrame()
    AllSUspk_df_testing = pd.DataFrame()
    # might sample the same cluster multiple times, so need differentiation
    for cc, pickedcls in enumerate(pickedunits):
        AllSUspk_df_training_temp = AllSUspk_df_training_All[AllSUspk_df_training_All['sess_cls']==pickedcls]
        AllSUspk_df_training_temp = AllSUspk_df_training_temp.copy()
        AllSUspk_df_training_temp.loc[:,'sess_cls'] = pickedcls+'_'+str(cc)
        AllSUspk_df_training = pd.concat([AllSUspk_df_training,AllSUspk_df_training_temp],axis=0)
        AllSUspk_df_testing_temp = AllSUspk_df_testing_All[AllSUspk_df_testing_All['sess_cls']==pickedcls]
        AllSUspk_df_testing_temp = AllSUspk_df_testing_temp.copy()
        AllSUspk_df_testing_temp.loc[:,'sess_cls'] = pickedcls+'_'+str(cc)
        AllSUspk_df_testing = pd.concat([AllSUspk_df_testing,AllSUspk_df_testing_temp],axis=0)

    # ensure same cls and number of groups in testing and training set
    if AllSUspk_df_training.groupby(by=['sess_cls','trialMod','snr','respLabel','AVoffset']).ngroups \
        == AllSUspk_df_testing.groupby(by=['sess_cls','trialMod','snr','respLabel','AVoffset']).ngroups:
        # decoding A and AV conditions 
        fitacc_nNueron_sig_a_temp,fitacc_nNueron_sig_a_temp_sh,fit_coefs_sig_a_temp = decoder_SVM_detection_v1(AllSUspk_df_training[AllSUspk_df_training['trialMod']=='a'],
                                                                                                            AllSUspk_df_testing[AllSUspk_df_testing['trialMod']=='a'],
                                                                                                            seeds,kernel,trainSample_eachCat,testSample_eachCat)
        fitacc_nNueron_sig_av_temp,fitacc_nNueron_sig_av_temp_sh,fit_coefs_sig_av_temp = decoder_SVM_detection_v1(AllSUspk_df_training[AllSUspk_df_training['trialMod']=='av'],
                                                                                                            AllSUspk_df_testing[AllSUspk_df_testing['trialMod']=='av'],
                                                                                                            seeds,kernel,trainSample_eachCat,testSample_eachCat)
        test_cat = ['A','AV']
        fitacc_nNueron_noise_a_temp = np.nan
        fitacc_nNueron_noise_v_temp = np.nan
    else:
        # print(str(AllSUspk_df_training.groupby(by=['sess_cls','trialMod','snr','respLabel']).ngroups)+'_'
        #       +str(AllSUspk_df_testing.groupby(by=['sess_cls','trialMod','snr','respLabel']).ngroups)+'  '
        #       +str(len(clsall))+'  '+str(len(AllSUspk_df_training['sess_cls'].unique())))        
        raise ValueError('Mismatch cluster in training and testing data for the decoder!!')
    
    # print('time spend to decode one time bootstrap '+str(bs)+' '+str(timedelta(seconds=time.monotonic() - time1)))
    return fitacc_nNueron_sig_a_temp,fitacc_nNueron_noise_a_temp,\
        fitacc_nNueron_sig_av_temp,fitacc_nNueron_noise_v_temp,\
    fitacc_nNueron_sig_a_temp_sh,fitacc_nNueron_sig_av_temp_sh,\
        fit_coefs_sig_a_temp,fit_coefs_sig_av_temp,test_cat,bs

def decoder_SVM_detection_v1(AllSUspk_df_training,AllSUspk_df_testing,seeds,kernel,trainSample_eachCat = 200,testSample_eachCat = 50):
    clsall = len(AllSUspk_df_training.sess_cls.unique())
    # get spk for sig and noise
    # ensure when generating population activity, the trial conditions are the same for each cluster
    trainingdf= AllSUspk_df_training.groupby(by=['sess_cls','trialMod','respLabel','snr','AVoffset']).\
        sample(n=trainSample_eachCat,random_state=seeds+5,replace=True).sort_values(by=['sess_cls','trialMod','respLabel','snr','AVoffset']).reset_index(drop=True)
    testingdf= AllSUspk_df_testing.groupby(by=['sess_cls','trialMod','respLabel','snr','AVoffset']).\
        sample(n=testSample_eachCat,random_state=seeds+10,replace=True).sort_values(by=['sess_cls','trialMod','respLabel','snr','AVoffset']).reset_index(drop=True)

    # generate population activity trials for training set
    chunksize_tr = int(trainingdf.shape[0]/clsall)
    X_train_sig =  trainingdf['spkRateSig'].values.reshape(clsall,chunksize_tr).T  
    X_train_noise =  trainingdf['spkRateNoise'].values.reshape(clsall,chunksize_tr).T 
    X_train = np.concatenate((X_train_sig,X_train_noise),axis=0)
    y_train = np.array([1]*X_train_sig.shape[0]+[0]*X_train_noise.shape[0])

    # generate population activity trials for testing set
    chunksize_te = int(testingdf.shape[0]/clsall)
    X_test_sig =  testingdf['spkRateSig'].values.reshape(clsall,chunksize_te).T  
    X_test_noise =  testingdf['spkRateNoise'].values.reshape(clsall,chunksize_te).T    
    X_test = np.concatenate((X_test_sig,X_test_noise),axis=0) #trials X cls
    y_test = np.array([1]*X_test_sig.shape[0]+[0]*X_test_noise.shape[0]) #(trials,)
     
    # shuffle yy lables to get baseline accuracy,  
    rng = np.random.default_rng(seeds+3000)
    yy_train_shuffle = rng.choice(y_train,size=len(y_train),replace=False)
    yy_test_shuffle = rng.choice(y_test,size=len(y_test),replace=False)

    #classification decoding    
    fitresultdict = svmdecoder([],[],[],[X_train,X_test,y_train,y_test],cpusSVM=1,kernel=kernel) 
    fitacc_nNueron = np.array(fitresultdict['accuracy_score'])
    # save fit coef of the best fit for each feature  
    fit_coefs = pd.DataFrame({'coef':fitresultdict['bestcoefs'].tolist()[0],
                                'coef_abs':np.abs(fitresultdict['bestcoefs']).tolist()[0]})      

    #  decoding shuffled signal from noise for baseline accuracy
    fitresultdict_sh = svmdecoder([],[],[],[X_train,X_test,yy_train_shuffle,yy_test_shuffle],cpusSVM=1,kernel=kernel) 
    fitacc_nNueron_sh = np.array(fitresultdict_sh['accuracy_score'])
       
    return fitacc_nNueron,fitacc_nNueron_sh,fit_coefs

##############################################################
def decoder_decision(AllSUspk_df,catsdict,units=30,trainSample_eachCat = 200,testSample_eachCat = 50,bstimes=40,kernel=['rbf']): # randomly pick trials from each cluster across sessions, with replacement; use all clusters, parallel bootstrap process
    # # output 'fitacc_nNeuron': clsSamp X 1 X A/AV X bootstrapTimes 
    fitacc_nNueron_dict = {}
    # initialize arrays to save decoding results
    print('Initializing arrays for results save..............')
    fitacc_nNueron_sig = np.empty((1,1,1,bstimes)) # Nneurons X clsSamp X 1 X bootstrapTimes 
    fitacc_nNueron_sig_bsline = np.empty((1,1,1,bstimes)) # Nneurons X clsSamp X 1 X bootstrapTimes 
    fit_coefs_sig = pd.DataFrame()

    print('Start boostrape decoding.......')
    argItems = [(AllSUspk_df,bs,units,catsdict,seeds,kernel,trainSample_eachCat,testSample_eachCat) for (bs,seeds) in zip(range(bstimes),np.random.choice(range(10000), size=bstimes, replace=False))]
    with Pool(processes=cpus) as p:
        # get the decode result for a bootstraped trials set
        for (fitacc_nNueron_sig_temp,fitacc_nNueron_sig_temp_sh,
             fit_coefs_sig_temp,test_cat,bs) in p.starmap(decoderdecision1roundBStrials,argItems):
            fitacc_nNueron_sig[:,:,:,bs] = fitacc_nNueron_sig_temp.reshape((-1,1))            
            fitacc_nNueron_sig_bsline[:,:,:,bs] = fitacc_nNueron_sig_temp_sh.reshape((-1,1)) 
            fit_coefs_sig_temp['bs'] = bs
            fit_coefs_sig_temp['mod'] = 'nan'          
            fit_coefs_sig = pd.concat([fit_coefs_sig,fit_coefs_sig_temp],axis=0)
    p.close()
    p.join() 
    # package all results into one dictionary
    fitacc_nNueron_dict['fitacc_nNeuron'] = fitacc_nNueron_sig.copy()
    fitacc_nNueron_dict['fitacc_nNeuron_bsline'] = fitacc_nNueron_sig_bsline.copy()
    fitacc_nNueron_dict['fit_coefs'] = fit_coefs_sig.copy()
    fitacc_nNueron_dict['trialLabel'] = test_cat
    # print(fitacc_nNueron_dict['fitacc_nNeuron'].shape)
    return fitacc_nNueron_dict  

def decoderdecision1roundBStrials(AllSUspk_df,bs,units,catsdict,seeds,kernel=['rbf'],trainSample_eachCat=200,testSample_eachCat=50):
    print('decoding in bootstrape '+str(bs)+'.................')
    ######## generate nonoverlap trials for training and testing set, 
    ####### activity of different neurons recorded in the same trial keep in the same set
    AllSUspk_df = AllSUspk_df.reset_index(drop=True)
    trialNum_df = AllSUspk_df.groupby(by=['sess','trialMod','snr','respLabel','AVoffset','trialNum']).size().reset_index()
    traintrialsbysess = trialNum_df.groupby(by=['sess','trialMod','snr','respLabel','AVoffset']).sample(frac=0.7,random_state=seeds,replace=False)[['sess','trialNum']]
    Alltrialsbysess = trialNum_df[['sess','trialNum']]
    merged_df = Alltrialsbysess.merge(traintrialsbysess, on=['sess','trialNum'], how='left', indicator=True)
    testtrialsbysess = merged_df[merged_df['_merge'] == 'left_only'].drop(columns='_merge')
    AllSUspk_df_training_All = AllSUspk_df.merge(traintrialsbysess,on=['sess','trialNum'],how='inner')
    AllSUspk_df_testing_All = AllSUspk_df.merge(testtrialsbysess,on=['sess','trialNum'],how='inner')

    # randomly select a set num of neurons
    np.random.seed(seeds)
    allunits = AllSUspk_df.sess_cls.unique()
    pickedunits = np.random.choice(allunits,size=units,replace=True)
    AllSUspk_df_training = pd.DataFrame()
    AllSUspk_df_testing = pd.DataFrame()
    # might sample the same cluster multiple times, so need differentiation
    for cc, pickedcls in enumerate(pickedunits):
        AllSUspk_df_training_temp = AllSUspk_df_training_All[AllSUspk_df_training_All['sess_cls']==pickedcls]
        AllSUspk_df_training_temp = AllSUspk_df_training_temp.copy()
        AllSUspk_df_training_temp.loc[:,'sess_cls'] = pickedcls+'_'+str(cc)
        AllSUspk_df_training = pd.concat([AllSUspk_df_training,AllSUspk_df_training_temp],axis=0)
        AllSUspk_df_testing_temp = AllSUspk_df_testing_All[AllSUspk_df_testing_All['sess_cls']==pickedcls]
        AllSUspk_df_testing_temp = AllSUspk_df_testing_temp.copy()
        AllSUspk_df_testing_temp.loc[:,'sess_cls'] = pickedcls+'_'+str(cc)
        AllSUspk_df_testing = pd.concat([AllSUspk_df_testing,AllSUspk_df_testing_temp],axis=0)

    # ensure same cls and number of groups in testing and training set
    if AllSUspk_df_training.groupby(by=['sess_cls','trialMod','snr','respLabel','AVoffset']).ngroups \
        == AllSUspk_df_testing.groupby(by=['sess_cls','trialMod','snr','respLabel','AVoffset']).ngroups:
        # decoding A from AV conditions 
        fitacc_nNueron_sig_a_temp,fitacc_nNueron_sig_a_temp_sh,fit_coefs_sig_a_temp = decoder_SVM_decision(AllSUspk_df_training,
                                                                                                            AllSUspk_df_testing,
                                                                                                            catsdict,seeds,kernel,trainSample_eachCat,testSample_eachCat)
        test_cat = ['nan']
    else:      
        raise ValueError('Mismatch cluster in training and testing data for the decoder!!')
    
    return fitacc_nNueron_sig_a_temp,fitacc_nNueron_sig_a_temp_sh,\
        fit_coefs_sig_a_temp,test_cat,bs

def decoder_SVM_decision(AllSUspk_df_training,AllSUspk_df_testing,catsdict,seeds,kernel,trainSample_eachCat = 200,testSample_eachCat = 50):
    clsall = len(AllSUspk_df_training.sess_cls.unique())
    ###### generate training and testing arrays for neuron populations
    X_train = np.empty((0,clsall)) #trials X cls
    X_test = np.empty((0,clsall)) #trials X cls
    y_train = np.empty((0,)) #(trials,)
    y_test = np.empty((0,)) #(trials,)
    ###### balance trials in different conditions for test and train sets      
    for cc,cat in enumerate(list(catsdict.values())[0]):
        AllSUspk_df_training_cc = AllSUspk_df_training[AllSUspk_df_training[list(catsdict.keys())[0]]==cat]
        AllSUspk_df_testing_cc = AllSUspk_df_testing[AllSUspk_df_testing[list(catsdict.keys())[0]]==cat] 

        trainingdf_temp= AllSUspk_df_training_cc.groupby(by=['sess_cls','trialMod','respLabel','snr','AVoffset']).\
            sample(n=trainSample_eachCat,random_state=seeds+5,replace=True).sort_values(by=['sess_cls','trialMod','respLabel','snr','AVoffset'])       
        chunksize_tr = int(trainingdf_temp.shape[0]/clsall)
        X_train_temp = trainingdf_temp['spkRateSig'].values.reshape(clsall,chunksize_tr).T  
        X_train = np.concatenate((X_train,X_train_temp),axis=0)
        y_train = np.concatenate((y_train,np.array([cc]*X_train_temp.shape[0])),axis=0)

        testingdf_temp= AllSUspk_df_testing_cc.groupby(by=['sess_cls','trialMod','respLabel','snr','AVoffset']).\
            sample(n=testSample_eachCat,random_state=seeds+10,replace=True).sort_values(by=['sess_cls','trialMod','respLabel','snr','AVoffset'])
        chunksize_te = int(testingdf_temp.shape[0]/clsall)
        X_test_temp = testingdf_temp['spkRateSig'].values.reshape(clsall,chunksize_te).T    #
        X_test = np.concatenate((X_test,X_test_temp),axis=0)
        y_test = np.concatenate((y_test,np.array([cc]*X_test_temp.shape[0])),axis=0)
    
    # print('chance cr:'+str(np.round(min(np.unique(y_test, return_counts=True)[1]/len(y_test))*100,decimals=2))+'%')
    # shuffle yy lables to get baseline accuracy,  
    rng = np.random.default_rng(seeds+3000)
    yy_train_shuffle = rng.choice(y_train,size=len(y_train),replace=False)
    yy_test_shuffle = rng.choice(y_test,size=len(y_test),replace=False)
     
    #classification decoding    
    fitresultdict = svmdecoder([],[],[],[X_train,X_test,y_train,y_test],cpusSVM=1,kernel=kernel) 
    fitacc_nNueron = np.array(fitresultdict['accuracy_score'])
    # save fit coef of the best fit for each feature  
    fit_coefs = pd.DataFrame({'coef':fitresultdict['bestcoefs'].tolist()[0],
                                'coef_abs':np.abs(fitresultdict['bestcoefs']).tolist()[0]})      

    #  decoding shuffled signal from noise for baseline accuracy
    fitresultdict_sh = svmdecoder([],[],[],[X_train,X_test,yy_train_shuffle,yy_test_shuffle],cpusSVM=1,kernel=kernel) 
    fitacc_nNueron_sh = np.array(fitresultdict_sh['accuracy_score'])
       
    return fitacc_nNueron,fitacc_nNueron_sh,fit_coefs


##############################################################

def applysvm(xx,yy,timeRange,binTim,catNames,outputPathway,jsonnameStr,method,seriesFeature,singleSess):
    # small time bins slide on whole epoch  
    binLen = int(xx.shape[1]*binTim/(timeRange[1]-timeRange[0]))
    startts = np.arange(0,xx.shape[1]-binLen+1,binLen)
    all_fit_result_slide = []
    all_raw_timeseries = []
    for rpt in range(10):
        print('svm repetition ' + str(rpt))
        fit_result_slide = []
        # balance trials in each category
        pickTrialsInd = BalanceSamples(yy)
        xx_partial = xx[pickTrialsInd]
        yy_partial = yy[pickTrialsInd]
        # category mean+sem and numtrials of each cat
        raw_timeseriesMean  = catmean(xx_partial,yy_partial,catNames,seriesFeature)

        for ind, tt in enumerate(startts): 
            # xx_bin = np.mean(xx[:,tt:tt+binLen],axis=1).reshape(-1,1)
            xx_bin = xx_partial[:,tt:tt+binLen] 
            fit_result_slide_temp = svmdecoder(xx_bin,yy_partial,method=method) 
            # fit_result_slide_temp = []                                        
            fit_result_slide.append(fit_result_slide_temp)  

        all_fit_result_slide.append(fit_result_slide)  
        all_raw_timeseries.append(raw_timeseriesMean)

    if not singleSess:
        with open(outputPathway+jsonnameStr+'_svmFitscore.json', 'w') as fp:
            json.dump({'all_fit_result_slide':all_fit_result_slide,'all_raw_timeseries':all_raw_timeseries}, fp)
            print("Done writing svm fitting results into .json file")  
    
    return all_fit_result_slide, all_raw_timeseries  

# pseudo determinant
def pseudodet(matrix):
    # Compute the singular values using SVD
    U, s, V = np.linalg.svd(matrix)   
    # Filter out the zero singular values
    non_zero_singular_values = s[s > np.finfo(s.dtype).eps]   
    # Compute the pseudo-determinant
    return np.prod(non_zero_singular_values)

def applyMGD(spkdf_sig_a,spkdf_noise_a,fitacc_nNueron_sig_a,fitacc_nNueron_noise_a,spkdf_sig_av,spkdf_noise_v,fitacc_nNueron_sig_av,fitacc_nNueron_noise_v,clsSamp,clsall,Nnlist,bs,CatunitIndlist):
    # test all trials for clsSamp times, with Nn randomly pick without replacement clusters
    # detect for sig_a trial, get hit/miss, cr/fa
    print('start detecting for sig_a and noise_a trials.....')
    argItems = [(spkdf_sig_a,spkdf_noise_a,clsSamp,clsall,tNn,Nn,CatunitIndlist['A'],seeds) for tNn,(Nn,seeds) in enumerate(zip(Nnlist,np.random.choice(range(10000), size=len(Nnlist), replace=False)))]
    with Pool(processes=cpus) as p:
        # get the decode result for a trial
        for fitacc_nNueron_trial_sig_a_temp,fitacc_nNueron_trial_noise_a_temp,spk_test_cat_sig_a,spk_test_cat_noise_a,tNn in p.starmap(decoder1round_MGDdetection,argItems):
            fitacc_nNueron_sig_a[tNn,:,:,bs] = fitacc_nNueron_trial_sig_a_temp #clsSamp X trial
            fitacc_nNueron_noise_a[tNn,:,:,bs] = fitacc_nNueron_trial_noise_a_temp
    p.close()
    p.join() 

    print('start detecting for sig_av and noise_v trials.....')
    # detect for av signal trial, get hit/miss, cr/fa
    argItems = [(spkdf_sig_av,spkdf_noise_v,clsSamp,clsall,tNn,Nn,CatunitIndlist['AV'],seeds) for tNn,(Nn,seeds) in enumerate(zip(Nnlist,np.random.choice(range(10000), size=len(Nnlist), replace=False)))]
    with Pool(processes=cpus) as p:
        # get the decode result for a trial
        for fitacc_nNueron_trial_sig_av_temp,fitacc_nNueron_trial_noise_av_temp,spk_test_cat_sig_av,spk_test_cat_noise_av,tNn in p.starmap(decoder1round_MGDdetection,argItems):
            fitacc_nNueron_sig_av[tNn,:,:,bs] = fitacc_nNueron_trial_sig_av_temp #clsSamp X trial
            fitacc_nNueron_noise_v[tNn,:,:,bs] = fitacc_nNueron_trial_noise_av_temp
    p.close()
    p.join()

    # concatenate all trial labels
    test_cat = spk_test_cat_sig_a+spk_test_cat_noise_a+spk_test_cat_sig_av+spk_test_cat_noise_av
    print('number of nan in the fitacc_nNueron matrix: sig-a'+str(np.sum(np.isnan(fitacc_nNueron_sig_a)))+' sig_av'+str(np.sum(np.isnan(fitacc_nNueron_sig_av)))+
          ' noise-a'+str(np.sum(np.isnan(fitacc_nNueron_noise_a)))+' noise-v'+str(np.sum(np.isnan(fitacc_nNueron_noise_v))))

    return fitacc_nNueron_sig_a,fitacc_nNueron_noise_a,fitacc_nNueron_sig_av,fitacc_nNueron_noise_v,test_cat

def decoder1round_MGDdetection(spkdf_1r,spkdf_2r,clsSamp,clsall,tNn,Nn,CatunitIndlist,seeds):#spkdf_1 sig, spkdf_2 noise
    def decode1trial(spk_pickedcls_test,spk_train,spkdf_2):
        catpdf_df = pd.DataFrame()  
        for cat in ['sig','noise']:
            if cat =='sig':
                spk_pickedcls_train_temp = spk_train.copy()  #cls X trials
            if cat == 'noise':
                spk_pickedcls_train_temp = spkdf_2.copy()

            cat_mean_temp = spk_pickedcls_train_temp.mean(axis=1).values.reshape(-1,1)
            cat_cov_temp = np.cov(spk_pickedcls_train_temp.values,rowvar=True,bias=True)  
            if Nn==1:
                if cat_cov_temp==0:
                    # print('divided by zero error when decoding from '+cat)
                    # print(spk_pickedcls_train_temp.values)
                    pdftemp = [[np.nan]] 
                    # print('cls selected:')
                    # print([clsall[ss] for ss in pickedcls])
                else:
                    pdftemp = (1/(((2*np.pi)**(Nn/2))*cat_cov_temp))\
                        *np.exp(-0.5*((spk_pickedcls_test-cat_mean_temp)/cat_cov_temp)**2)
                catpdf_df[cat] = [pdftemp[0][0]]
            elif Nn>1:  
                # remove between neuron correlations
                cat_cov_temp = np.diag(np.diag(cat_cov_temp)) 
                # print('the present decoding covariance matrix only keep diagonal values!')

                if pseudodet(cat_cov_temp)==0:
                    pdftemp = [[np.nan]]
                    # print('divided by zero error when decoding from '+cat)
                else:
                    try:
                        pdftemp = (1/(((2*np.pi)**(Nn/2))*(pseudodet(cat_cov_temp))**0.5))\
                            *np.exp(-0.5*(spk_pickedcls_test-cat_mean_temp).T*np.linalg.pinv(cat_cov_temp)*(spk_pickedcls_test-cat_mean_temp))                                                                                          
                    except RuntimeWarning:
                        # print('overflow error: covmatrix')
                        # print('overflow error when decoding from '+cat)
                        pdftemp = [[np.nan]]                
                catpdf_df[cat] = [pdftemp[0][0]]
        # if any condition cannot get the pdf value 
        if catpdf_df.isna().any().any():
            correctcat = np.nan
        else:
            correctcat = str(catpdf_df.idxmax(axis=1).values[0]) # decoded category of the test trial

        # check whether hit/miss decoded results 
        try:
            if np.isnan(correctcat):
                fitacc_nNueron_trial_temp = np.nan  
        except TypeError:
            if correctcat =='sig':
                fitacc_nNueron_trial_temp = True # hit(spk1=spk_sig)/cr(spk1=spk_noise) 
            else:
                fitacc_nNueron_trial_temp = False  # miss(spk1=spk_sig)/FA(spk1=spk_noise)  
        return fitacc_nNueron_trial_temp
   
    sig_trials = spkdf_1r.shape[1]
    noise_trials = spkdf_2r.shape[1]
    fitacc_nNueron_trial_sig_temp = np.empty((clsSamp,sig_trials)) #clsSamp X trials 
    fitacc_nNueron_trial_noise_temp = np.empty((clsSamp,noise_trials)) #clsSamp X trials 

    # randomly choose same subset cls without replacements of both sig&noise                  
    for rep in range(clsSamp):  
        rng1 = np.random.default_rng(seeds+rep)
        rng2 = np.random.default_rng(seeds+rep+1000)
        rng3 = np.random.default_rng(seeds+rep+2000)
        # sequentially pick snr, v+snr, and v modulated units in the decoder
        if len(CatunitIndlist)>0:
            if all(isinstance(i, list) for i in CatunitIndlist): # if CatunitIndlist is a list of list: save different categories of neurons modified by A,A+V,V
                cls1 = CatunitIndlist[0]
                cls2=CatunitIndlist[1]
                cls3=CatunitIndlist[2]
                # pickedcls = np.random.choice(cls1+cls2+cls3, size=Nn,replace=False)            
                if Nn<=len(cls1)/2:
                    pickedcls = rng1.choice(cls1, size=Nn,replace=False) 
                elif Nn<=(len(cls1)+len(cls2))/2:
                    pickedcls = np.concatenate((rng1.choice(cls1, size=int(len(cls1)/2),replace=False),
                                                rng2.choice(cls2, size=Nn-int(len(cls1)/2),replace=False)))
                elif Nn<=(len(cls1)+len(cls2)+len(cls3))/2:
                    pickedcls = np.concatenate((rng1.choice(cls1, size=int(len(cls1)/2),replace=False),
                                                rng2.choice(cls2, size=int(len(cls2)/2),replace=False),
                                                rng3.choice(cls3, size=Nn-int(len(cls1)/2)-int(len(cls2)/2),replace=False)))
                    
            elif all(not isinstance(i, list) for i in CatunitIndlist): # if CatunitIndlist is a list: save ordered neuron index
                pickedcls = CatunitIndlist[:Nn]
                # print(('pickedcls',pickedcls))
        else:
            pickedcls = rng1.choice(np.arange(0,X_train.shape[1],1), size=Nn,replace=False) 

        spkdf_1 = spkdf_1r.iloc[pickedcls,:]
        spkdf_2 = spkdf_2r.iloc[pickedcls,:]
        spk_test_cat_sig = [] 
        spk_test_cat_noise = []         
        # detect for signal 
        for tt in range(sig_trials):            
            # test each trial on MG distribution of each cat
            spk_pickedcls_test = spkdf_1.iloc[:,tt].values.reshape(-1,1) #test trial array: clsX1
            spk_test_cat_sig = spk_test_cat_sig+[spkdf_1.columns[tt]]
            spk_train = spkdf_1.drop(columns=spkdf_1.columns[tt],inplace=False) # training trials dataframe
            fitacc_nNueron_trial_sig_temp[rep,tt]=decode1trial(spk_pickedcls_test,spk_train,spkdf_2) #1hit,0miss   
        # detect for noise
        for tt in range(noise_trials):
            # test each trial on MG distribution of each cat
            spk_pickedcls_test = spkdf_2.iloc[:,tt].values.reshape(-1,1) #test trial array: clsX1
            spk_test_cat_noise = spk_test_cat_noise+[spkdf_2.columns[tt]]
            spk_train = spkdf_2.drop(columns=spkdf_2.columns[tt],inplace=False) # training trials dataframe
            fitacc_nNueron_trial_noise_temp[rep,tt]=decode1trial(spk_pickedcls_test,spk_train,spkdf_1) #1cr,0fa 
            # print('noise detect trial '+str(tt)+' ' +str(fitacc_nNueron_trial_noise_temp[rep,tt]))   

    return fitacc_nNueron_trial_sig_temp,fitacc_nNueron_trial_noise_temp,spk_test_cat_sig,spk_test_cat_noise,tNn

