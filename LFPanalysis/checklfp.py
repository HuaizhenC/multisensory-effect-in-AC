import time
from datetime import timedelta
import numpy as np
import scipy
import seaborn as sns
import h5py
import pickle
import os
import sys
import pandas as pd
from matplotlib import pyplot as plt 
import matplotlib.ticker as ticker
from multiprocessing import Pool
import re
from spikeUtilities import SortfilterDF 
from mne.stats import permutation_cluster_test,permutation_cluster_1samp_test,combine_adjacency

cpus = 10

numFreq = 80
morletFreq = np.logspace(np.log10(1),np.log10(150),numFreq)

# def getITC(compareCond,filterCond,lfpSeg_chan_temp,Behavdata_df,input,bootstapSam):              
#     alltrialNum_new = np.sort(Behavdata_df['trialNum'].unique(),kind='mergesort') # each num corresponding to a row in lfpSeg_chan_temp 

#     #get bootstripped ITC samples for each condition, and saved in a dict with condition as keys
#     ITC = {} 
#     for compCon in next(iter(compareCond.values())):
#         # get all row indexes belong to this compCond
#         cond_filter_temp = {**filterCond,**{list(compareCond.keys())[0]:[compCon]}}
#         _,rowind = SortfilterDF(Behavdata_df,cond_filter_temp)

#         # Bootstrip to avoid bias caused by trial imbalance between conditions         
#         ITC_temp =[] 
#         for bs in range(bootstapSam):
#             sampsize = 150
#             ind_temp = np.sort(np.random.choice(rowind,size=sampsize,replace=False),kind='mergesort') # randomly pick rows without replacement
#             Behavdata_df_choice_temp = Behavdata_df.loc[ind_temp]
#             # get corresponding lfprows according to the selected spktim trialNum
#             trialNum_temp =Behavdata_df_choice_temp['trialNum'].values
#             lfpRowind = [np.where(alltrialNum_new==kk)[0][0] for kk in trialNum_temp]
#             lfpSeg_chan_BStemp = list(lfpSeg_chan_temp[lfpRowind])# spks X[freqXtime]
#             # est ITC
#             if input =='phase':
#                 ITC_temp.append(np.abs(np.sum(np.exp(np.array(lfpSeg_chan_BStemp)*1j),axis=0))/sampsize)
#             if input == 'power':
#                 ITC_temp.append(np.sum(np.array(lfpSeg_chan_BStemp),axis=0)/sampsize)
#         ITC[compCon] = np.array(ITC_temp) # bootstapSam X freq X time
#     return ITC

def getITC(ITCbsCond,filterCond,lfpSeg_chan_temp,Behavdata_df,input,bootstapSam):    
    #get bootstrapped ITC samples for each condition, and saved in a dict with condition as keys
    ITC = {} 
    for compCon in next(iter(ITCbsCond.values())):
        # get all row indexes belong to this compCond
        cond_filter_temp = {**filterCond,**{list(ITCbsCond.keys())[0]:[compCon]}}
        Behavdata_df_temp,rowind = SortfilterDF(Behavdata_df,cond_filter_temp)

        # Bootstrap to avoid bias caused by trial imbalance between conditions         
        ITC_temp =[] 
        for bs in range(bootstapSam):
            print('bootstrap ITC of repeat '+str(bs))
            sampsize = 150
            if len(list(Behavdata_df_temp.index))>sampsize:
                ind_temp = np.sort(np.random.choice(list(Behavdata_df_temp.index),size=sampsize,replace=False),kind='mergesort') # randomly pick rows without replacement
            else:
                print('condition '+str(cond_filter_temp) + 'do not have enought trials for boothstrap without replace')
                ind_temp = np.sort(np.random.choice(list(Behavdata_df_temp.index),size=sampsize,replace=True),kind='mergesort') # randomly pick rows without replacement

            Behavdata_df_choice_temp = Behavdata_df.loc[ind_temp]
            # get corresponding lfprows according to the selected spktim trialNum
            lfpRowind = list(Behavdata_df_choice_temp.index)
            lfpSeg_chan_BStemp = list(lfpSeg_chan_temp[lfpRowind])# spks X[freqXtime]
            #check nan in the selected lfpseg
            trialsincludenan = np.where(np.any(np.isnan(lfpSeg_chan_temp[lfpRowind]),axis=(1,2)))[0]
            if len(trialsincludenan)==0:
                pass
            else:
                print('nan appears in these trials of lfp!')
                print(Behavdata_df.loc[[lfpRowind[oo] for oo in trialsincludenan]].to_string())
            # est ITC
            if input =='phase':
                ITC_temp.append(np.abs(np.sum(np.exp(np.array(lfpSeg_chan_BStemp)*1j),axis=0))/sampsize)
            if input == 'power':
                ITC_temp.append(np.sum(np.array(lfpSeg_chan_BStemp),axis=0)/sampsize)
        ITC[compCon] = np.array(ITC_temp) # bootstapSam X freq X time
    return ITC



def getChanNum(df_avMod_all_sig, MonkeyDate):
   def getchan(session_clslist):
       # Regular expression pattern to match 'ch' followed by one or more digits
       pattern = r'ch(\d+)'
       # Extract numbers after 'ch' for each element
       ch_numbers = np.unique(np.array([int(re.search(pattern, element).group(1)) for element in session_clslist if re.search(pattern, element)]))
       return ch_numbers
       
   chanAudNeuDF = pd.DataFrame()
   for Monkey,Date in MonkeyDate.items():
        for dd in Date:
            chanAudNeuDF_temp = pd.DataFrame()
            df_avMod_sess = df_avMod_all_sig[(df_avMod_all_sig['Monkey']==Monkey) & (df_avMod_all_sig['session_cls'].str.contains(dd))]   
            chanAudNeuDF_temp['Sigchan'] = getchan(df_avMod_sess.session_cls.values.tolist())
            chanAudNeuDF_temp['sess'] = dd
            chanAudNeuDF_temp['Monkey'] = Monkey
            chanAudNeuDF = pd.concat((chanAudNeuDF,chanAudNeuDF_temp))
   return chanAudNeuDF


# MonkeyDate_all = {'Elay':['230420','230503']}

matlabfilePathway='/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/data/'
genH5Pathway ='/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/data/'
figsavPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/'
ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/LFP/'
AVmodPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/AVmodIndex/'
glmfitPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/glmfit/'

# MonkeyDate_all = {'Elay':['230420','230503','230509','230525','230531',
#                         '230602','230606','230608','230613','230616','230620','230627',
#                         '230705','230711','230717','230718','230719','230726','230728',
#                         '230802','230808','230810','230814','230818','230822','230829',
#                         '230906','230908','230915','230919','230922','230927',
#                          '231003','231004','231010'], 
#                   'Wu':['230809','230815','230821','230830',
#                         '230905','230911','230913','230918','230925',
#                           '231002','231006','231009']}

# MonkeyDate_all = {'Elay':['230420','230503','230509','230525','230531',
#                         '230602','230606','230613','230616','230620','230627',
#                         '230705','230711','230717','230718','230719','230726','230728',
#                         '230802','230808','230810','230814','230818','230822','230829',
#                         '230906','230908','230915','230919','230922','230927',
#                          '231003','231004','231010']}

MonkeyDate_all = {'Elay':['230613']}

matlabfilePathway='/data/by-user/Huaizhen/preprocNeuralMatfiles/Vprobe+EEG/'
genH5Pathway = '/data/by-user/Huaizhen/LFPcut/2Dpower+phase/'
figsavPathway = '/data/by-user/Huaizhen/Figures/lfpXtrialCoh/'
ResPathway = '/data/by-user/Huaizhen/Fitresults/lfpXtrialCoh/'
AVmodPathway = '/data/by-user/Huaizhen/Fitresults/AVmodIndex/'
glmfitPathway = '/data/by-user/Huaizhen/Fitresults/glmfit/'

# timerange parameter, need to have the same 0
timeRange_lfp = [-0.8,0.6] #fs 1220.703125, shouldn't change 

alignStr = 'align2coo'
alignkeys = 'cooOnsetIndwithVirtual'#'cooOnsetIndwithVirtual','VidOnsetIndwithVirtual','JSOnsetIndwithVirtual'
x0str = 'cooOn'
input='phase'# 'phase' 'power'

# respLabel decode:  
# nosoundplayed: [nan]
# A:  hit[0],miss/latemiss[1],FAa[2],erlyResp[88]
# AV: hit[0],miss/latemiss[1],FAv[22],erlyResp[88]
# V:  hit[0],CRv[10,100],FAv[22]
filterCond ={'trialMod':['a','av','v'],'respLabel':['hit','miss','CR']}


for Monkey,Date in MonkeyDate_all.items():
    ITClfp_all={}
    for Date_temp in Date:
        figsavPathway_temp = os.path.join(figsavPathway,Monkey+Date_temp)
        try:
            os.mkdir(figsavPathway_temp)
        except FileExistsError:
            pass        
        print('...............'+'process session '+Monkey+'-'+Date_temp+'...............')
        start_time = time.monotonic()

        filename = genH5Pathway+Monkey+Date_temp+'_2Dpowerphase_trial_'+alignStr+'_lfpXtrialCohtimrange'                          
        Behavdata_df_new = pickle.load(open(genH5Pathway+Monkey+Date_temp+'_trial_Behavdata_'+alignStr+'.pkl','rb'))            
        with h5py.File(filename+'_'+input+'.h5','r') as file:
            lfpSeg = file.get('LFP'+input+'Seg')[:] #trial X chan X freq X time
        print('lfpSeg shape is: '+str(lfpSeg.shape) + ' Behavdata_df_new shape is'+str(Behavdata_df_new.shape))
        # lfpSeg = np.random.rand(1139,24,5,10)
        # average 3 channels of lfp around the median channel in chanAudNeuDF of this session
        # medChanNum = np.median(chanAudNeuDF[chanAudNeuDF['sess']==Date_temp]['Sigchan'].values)
        # medChanNum =12
        lfpSeg_ave_temp = np.mean(lfpSeg,axis=1)#trial X freq X time
        # filter out trials
        Behavdata_df_temp,rowind = SortfilterDF(Behavdata_df_new,filterCond)
        Behavdata_df_temp = Behavdata_df_temp.reset_index(drop=True)
        lfpSeg_ave_temp_temp = lfpSeg_ave_temp[rowind,:,:].copy()

        #check nan in the selected lfpseg
        trialsincludenan = np.where(np.any(np.isnan(lfpSeg_ave_temp_temp),axis=(1,2)))[0]
        print(trialsincludenan)
        if len(trialsincludenan)==0:
            pass
        else:
            print('nan appears in these trials of lfp!')
            print(Behavdata_df_temp.loc[trialsincludenan].to_string())
            print('full behavdf')
            print(Behavdata_df_new.to_string())