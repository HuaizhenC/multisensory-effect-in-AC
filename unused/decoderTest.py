from sklearn import datasets
import json
from decoders import svmdecoder
import numpy as np
import pandas as pd

def bstrialsfromeachcls(AllSUspk_df,mintrialNum_sig,extrcolstr):
    #balance num of trials in each sig category based on the mimtrialNum
    spkdf_sig = pd.DataFrame()
    spkdf_noise_a = pd.DataFrame()
    spkdf_noise_v = pd.DataFrame()
    for cls in AllSUspk_df['sess_cls'].unique():  
        SUspk_df_temp =  AllSUspk_df[AllSUspk_df['sess_cls']==cls]
        SUspk_df_temp.reset_index(drop=True,inplace=True) 

        # shuffle trials with same snr,trialmod,resp to remove trial by trial correlations, if pool neurons across sessions
        # resample trials in each condition with replacement
        SUspk_df_temp_sig = SUspk_df_temp[SUspk_df_temp['respLabel'].isin(['hit','miss'])].reset_index(drop=True)    
        grouped = SUspk_df_temp_sig.groupby(by=['trialMod','snr-shift'])# the category group should appear the same order when loop across cls
        group_indices = {key: grouped.groups[key].tolist() for key in grouped.groups}
        spkdf_sig_cat_temp = pd.DataFrame()
        for (gkey,gsnr),glist in group_indices.items():            
            if any(substr in gkey for substr in ['a','av']):# a/av signal trial
                ind_temp_sig = np.random.choice(glist,size=mintrialNum_sig[gkey][mintrialNum_sig[gkey]['snr-shift']==gsnr].iloc[0,-1],replace=False) # randomly pick num of trials in each category without replacements for signal
                collist_sig=[extrcolstr+gkey+'_'+str(gsnr)+'_sig'+extrcolstr+str(tt) for tt in range(len(ind_temp_sig))]
                spkdf_sig_cat_temp = pd.concat([spkdf_sig_cat_temp,pd.DataFrame(SUspk_df_temp_sig.loc[ind_temp_sig]['spkRateSig'].values.reshape(1,-1),columns=collist_sig)],axis=1) # rowxcol: cls X trials
                # print((cls,gkey,gsnr,SUspk_df_temp_sig.loc[ind_temp_sig]['spkRateSig'].values[:10]))
        # visual noise trial, code should only run one time for each cls
        ind_temp_sig = np.random.choice(list(SUspk_df_temp[(SUspk_df_temp['trialMod']=='v')&(SUspk_df_temp['respLabel']=='CR')].index)
                                        ,size=mintrialNum_sig['vnoise'],replace=False) # randomly pick num of trials in each category without replacements for signal
        collist_sig=[extrcolstr+'v_CR_noise'+extrcolstr+str(tt) for tt in range(len(ind_temp_sig))]                
        spkdf_noise_v_cr_temp = pd.DataFrame(SUspk_df_temp.loc[ind_temp_sig]['spkRateSig'].values.reshape(1,-1),columns=collist_sig)                    
        # audio noise trial (CR), same num of sample as visual noise trial, some trial the spkRateNoise could be NaN  
        ind_temp_noise = np.random.choice(list(SUspk_df_temp.dropna(subset=['spkRateNoise']).index),size=mintrialNum_sig['anoise'],replace=False)  
        collist_noise=[extrcolstr+'a_CR_noise'+extrcolstr+str(tt) for tt in range(len(ind_temp_noise))]
        spkdf_noise_a_cr_temp = pd.DataFrame(SUspk_df_temp.loc[ind_temp_noise]['spkRateNoise'].values.reshape(1,-1),columns=collist_noise) # rowxcol: cls X trials
        collist_noise=[extrcolstr+'v_CR_noise'+extrcolstr+str(tt) for tt in range(len(ind_temp_noise))]
        spkdf_noise_v_cr_temp2 = pd.DataFrame(SUspk_df_temp.loc[ind_temp_noise]['spkRateNoise'].values.reshape(1,-1),columns=collist_noise) # rowxcol: cls X trials
                
        # signal and noise from different trials are saved 
        # assume all sessions have same cetegories, will report error if a session don't have all the categories
        spkdf_sig = pd.concat([spkdf_sig,spkdf_sig_cat_temp],axis=0)
        spkdf_noise_a = pd.concat([spkdf_noise_a,spkdf_noise_a_cr_temp],axis=0)
        spkdf_noise_v = pd.concat([spkdf_noise_v,spkdf_noise_v_cr_temp2],axis=0)
        # spkdf_noise_a = pd.concat([spkdf_noise_a,spkdf_noise_a_fa_temp],axis=0)
        # spkdf_noise_v = pd.concat([spkdf_noise_v,spkdf_noise_v_fa_temp],axis=0)  

    spkdf_sig_a = spkdf_sig[[col for col in spkdf_sig.columns if '*a_' in col]]
    spkdf_sig_av = spkdf_sig[[col for col in spkdf_sig.columns if '*av_' in col]]        
    print('total: spkdf_sig_a'+str(spkdf_sig_a.shape)+'  spkdf_noise_a'+str(spkdf_noise_a.shape)+' spkdf_sig_av'+str(spkdf_sig_av.shape)+'  spkdf_noise_v'+str(spkdf_noise_v.shape))       

    #need to balance NoiseSigTrials for bayesian theorom, randomly pick noise trials for each sig trial    
    noisepicka = np.sort(np.random.choice(range(spkdf_noise_a.shape[1]),size=spkdf_sig_a.shape[1],replace=False))
    noisepickv = np.sort(np.random.choice(range(spkdf_noise_v.shape[1]),size=spkdf_sig_av.shape[1],replace=False))
    spkdf_noise_a_sub = spkdf_noise_a.iloc[:,noisepicka]
    spkdf_noise_v_sub = spkdf_noise_v.iloc[:,noisepickv]
    print('afterbalancing: spkdf_sig_a'+str(spkdf_sig_a.shape)+'  spkdf_noise_a'+str(spkdf_noise_a_sub.shape)+' spkdf_sig_av'+str(spkdf_sig_av.shape)+'  spkdf_noise_v'+str(spkdf_noise_v_sub.shape))       
    return spkdf_sig_a,spkdf_sig_av,spkdf_noise_a_sub,spkdf_noise_v_sub

