import time
from datetime import timedelta
import numpy as np
import scipy
from scipy.stats import f as fstat
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
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

cpus = 1

numFreq = 80
morletFreq = np.logspace(np.log10(1),np.log10(150),numFreq)

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
            # print('bootstrap ITC of repeat '+str(bs))
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
            trialsincludenan = np.where(np.any(np.isnan(lfpSeg_chan_BStemp),axis=(1,2)))[0]
            if len(trialsincludenan)==0:
                pass
            else:
                print('nan appears in these trials of lfp in bootstrap ITC of repeat'+str(bs))
                print(Behavdata_df_temp.iloc[trialsincludenan,:].to_string()) 

            # est ITC
            if input =='phase':
                ITC_temp.append(np.abs(np.sum(np.exp(np.array(lfpSeg_chan_BStemp)*1j),axis=0))/sampsize)
            if input == 'power':
                ITC_temp.append(np.sum(np.array(lfpSeg_chan_BStemp),axis=0)/sampsize)
        ITC[compCon] = np.array(ITC_temp) # bootstapSam X freq X time
    return ITC


def PermTest(lfpSegA,lfpSegB,n_permutations,p_thresh):
    # permutation test for power/iTC in one channel 
    listarray = [lfpSegA,lfpSegB]  
    signs = np.sign(lfpSegA.mean(axis=0)-lfpSegB.mean(axis=0)) 
    if  np.isnan(lfpSegA).any() or np.isnan(lfpSegB).any():
        print('nan included in power/ITC data!! ')
    dfn = len(listarray) - 1
    dfd = np.sum([len(x) for x in listarray]) - len(listarray)
    threshold = fstat.ppf(1.0 - p_thresh, dfn, dfd)
    print(threshold)
    adjacency = combine_adjacency(lfpSegA.shape[1], lfpSegA.shape[2])                               
    F_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(listarray,n_permutations=n_permutations,adjacency=adjacency,
                                                                     threshold=threshold,tail=0,n_jobs=cpus,seed=np.random.RandomState().seed(42),
                                                                     out_type='mask',buffer_size = None)
    P_obs = 1-fstat.cdf(F_obs,dfn,dfd)
    # Create stats image with only significant clusters
    F_obs_plot = np.nan * np.ones_like(F_obs)
    p_obs_plot = np.nan * np.zeros_like(F_obs)
    for c, p_val in zip(clusters, cluster_p_values):             
        if p_val <= 0.05:
            F_obs_plot[c] = F_obs[c] * signs[c]
            p_obs_plot[c] = P_obs[c] * signs[c]
    # # save sig map of all channels
    # F_obs_all = np.concatenate((F_obs_all,np.expand_dims(F_obs,axis=0)),axis=0)
    # F_obs_plot_all = np.concatenate((F_obs_plot_all,np.expand_dims(F_obs_plot,axis=0)),axis=0)
    max_F = np.nanmax(abs(F_obs_plot))
    return F_obs_plot,p_obs_plot,F_obs,max_F

def PermTest1samp(lfpSeg,n_permutations):
    # permutation test for power/iTC in one channel 
    if  np.isnan(lfpSeg).any():
        print('nan included in power/ITC data!! ')
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(lfpSeg,n_permutations=n_permutations,adjacency=None,tail=0,threshold=None,n_jobs=cpus,
                                                                    seed=np.random.RandomState().seed(42),out_type='mask',
                                                                    buffer_size = None)
    # Create stats image with only significant clusters
    T_obs_plot = np.nan * np.ones_like(T_obs)
    for c, p_val in zip(clusters, cluster_p_values):             
        if p_val <= 0.05:
            T_obs_plot[c] = T_obs[c] 
    max_T = np.nanmax(abs(T_obs_plot))
    return T_obs_plot,T_obs,max_T

def add0Str2xtick(xt,x0str):            
    xt = np.delete(xt,np.where(xt==0)[0])
    xt = np.append(xt,0)
    xtl = xt.tolist()
    xtl = [np.round(xtl[i],1) for i in range(len(xtl))]
    xtl[-1] = x0str
    return xt,xtl

def plot2Dsig(F_obs_plot,morletFreq,timeRange_lfp,titlestr,axes,max_F,x0str):   
    try:
        if len(max_F)==0:
            max_F = np.nanmax(abs(F_obs_plot)) 
    except TypeError:
        pass

    c = axes.imshow(F_obs_plot,
        extent=[timeRange_lfp[0], timeRange_lfp[1], morletFreq[0],morletFreq[-1]],
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-max_F,
        vmax=max_F,alpha=1)
    plt.colorbar(c,ax=axes,location='bottom',orientation='horizontal')
    axes.text(timeRange_lfp[0],morletFreq[-1],titlestr,fontsize=12)
    # Set the x-axis and y-axis tick labels
    axes.set_xlabel('Time (s)',fontsize=fontsizeNo)
    axes.set_ylabel('Frequency (Hz)',fontsize=fontsizeNo) 
    axes.set_xticks(list(np.arange(timeRange_lfp[0],timeRange_lfp[1],0.4)))                    
    xt = axes.get_xticks()                     
    xt,xtl = add0Str2xtick(xt,x0str)
    axes.set_xticks(xt)
    axes.set_xticklabels(xtl)
    axes.set_xlim(timeRange_lfp[0],timeRange_lfp[1])
    y_values = morletFreq[[0,20,40,60,len(morletFreq)-1]]
    axes.yaxis.set_major_locator(ticker.LinearLocator(numticks=len(y_values)))
    axes.set_yticks(axes.get_yticks())
    axes.set_yticklabels([f'{label:.2f}' for label in np.round(y_values,decimals=1)],fontsize=8)
    axes.tick_params(axis='both', which='major', labelsize=fontsizeNo-2)

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


# MonkeyDate_all = {'Elay':['230420']}

# matlabfilePathway='/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/data/'
# genH5Pathway ='/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/data/'
# figsavPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/'
# ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/LFP/'
# AVmodPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/AVmodIndex/'
# glmfitPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/glmfit/'

# MonkeyDate_all = {'Elay':['230420','230503','230509','230525','230531',
#                         '230602','230606','230613','230616','230620','230627',
#                         '230705','230711','230717','230718','230719','230726','230728',
#                         '230802','230808','230810','230814','230818','230822','230829',
#                         '230906','230908','230915','230919','230922','230927',
#                          '231003','231004','231010'], 
#                   'Wu':['230809','230815','230821','230830',
#                         '230905','230911','230913','230918','230925',
#                           '231002','231006','231009']}

MonkeyDate_all = {'Elay':['230420','230503','230509','230525','230531',
                        '230602','230606','230613','230616','230620','230627',
                        '230705','230711','230717','230718','230719','230726','230728',
                        '230802','230808','230810','230814','230818','230822','230829',
                        '230906','230908','230915','230919','230922','230927',
                         '231003','231004','231010']}

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
fontsizeNo = 20

# respLabel decode:  
# nosoundplayed: [nan]
# A:  hit[0],miss/latemiss[1],FAa[2],erlyResp[88]
# AV: hit[0],miss/latemiss[1],FAv[22],erlyResp[88]
# V:  hit[0],CRv[10,100],FAv[22]
filterCond ={'respLabel':['hit','miss','CR']}
ITCbsCond = {'trialMod':['a','av','v']}

# namstrfit = '_hitmissCR_GLMSig_rmdriftunits_allsession_medianchansave' 
namstrfit = '_hitmissCR_GLMSig_rmdriftunits_allsession_medianchan' 

ITCbootstapSam = 100 # bootstrip times of each condition for a session

# filter neurons based on different rules
# df_avMod_all = pickle.load(open(AVmodPathway+'AVmodTTestDF_baselinecorrected.pkl','rb'))
df_avMod_all = pickle.load(open(glmfitPathway+'glmfitCoefDF_cooOnsetIndwithVirtual_2mod2lab.pkl','rb'))
df_avMod_all = df_avMod_all[(df_avMod_all['GLMmod']=="['snr-shift', 'V']") & (df_avMod_all['time']==0)]
df_avMod_all_sig = df_avMod_all[df_avMod_all['pval']<0.05] 
clscolstr = 'session_cls'

# filter out drifted neurons
dfinspect = pd.read_excel('AllClusters4inspectionSheet.xlsx')
driftedUnites = list(dfinspect[dfinspect['driftYES/NO/MAYBE(1,0,2)']==1]['session_cls'].values)
df_avMod_all_sig = df_avMod_all_sig[~df_avMod_all_sig['session_cls'].isin(driftedUnites)]

# print(df_avMod_all_sig.to_string())
# a dataframe saves channels with sig auditory neurons in all sessions
chanAudNeuDF = getChanNum(df_avMod_all_sig,MonkeyDate_all)

start_time = time.monotonic()
for Monkey,Date in MonkeyDate_all.items(): 
    # ITClfp_all={}
    # for Date_temp in Date:      
    #     print('...............'+'process session '+Monkey+'-'+Date_temp+'...............')
        
    #     filename = genH5Pathway+Monkey+Date_temp+'_2Dpowerphase_trial_'+alignStr+'_lfpXtrialCohtimrange'                          
    #     Behavdata_df_new = pickle.load(open(genH5Pathway+Monkey+Date_temp+'_trial_Behavdata_'+alignStr+'.pkl','rb'))            
    #     with h5py.File(filename+'_'+input+'.h5','r') as file:
    #         lfpSeg = file.get('LFP'+input+'Seg')[:] #trial X chan X freq X time
    #     print('lfpSeg shape is: '+str(lfpSeg.shape) + ' Behavdata_df_new shape is'+str(Behavdata_df_new.shape))
    #     # lfpSeg = np.random.rand(1139,24,5,10)

    #     # median channel for phase, 
    #     # average 3 channels of lfp around the median channel for power 
    #     medChanNum = int(np.median(chanAudNeuDF[chanAudNeuDF['sess']==Date_temp]['Sigchan'].values))
    #     # medChanNum =12
    #     aveChans = np.arange(medChanNum-1,medChanNum+2,1,dtype=np.int16)
    #     if input=='power':
    #         lfpSeg_ave_temp = np.mean(lfpSeg[:,aveChans,:,:],axis=1)#trial X freq X time
    #     if input == 'phase':
    #         lfpSeg_ave_temp = lfpSeg[:,medChanNum,:,:] #trial X freq X time

    #     # estimate cross trial coherence on averaged lfp
    #     start_time1 = time.monotonic()
    #     ITClfp = getITC(ITCbsCond,filterCond,lfpSeg_ave_temp,Behavdata_df_new,input,ITCbootstapSam)
    #     # concatenate bootstraped ITC samples session by session
    #     try:
    #         ITClfp_all['a'] = np.concatenate((ITClfp_all['a'],ITClfp['a']),axis=0)
    #         ITClfp_all['av'] = np.concatenate((ITClfp_all['av'],ITClfp['av']),axis=0)
    #         ITClfp_all['v'] = np.concatenate((ITClfp_all['v'],ITClfp['v']),axis=0)
    #     except:
    #         ITClfp_all = {'a':np.empty((0,ITClfp['a'].shape[1],ITClfp['a'].shape[2])),
    #                       'av':np.empty((0,ITClfp['av'].shape[1],ITClfp['av'].shape[2])),
    #                       'v':np.empty((0,ITClfp['v'].shape[1],ITClfp['v'].shape[2]))}
    #         ITClfp_all['a'] = np.concatenate((ITClfp_all['a'],ITClfp['a']),axis=0)
    #         ITClfp_all['av'] = np.concatenate((ITClfp_all['av'],ITClfp['av']),axis=0)
    #         ITClfp_all['v'] = np.concatenate((ITClfp_all['v'],ITClfp['v']),axis=0)            

    #     print('bootstrape time spend for session '+Monkey+'-'+Date_temp+'...............') 
    #     print(timedelta(seconds=time.monotonic() - start_time)) 

    # pickle.dump(ITClfp_all,open(ResPathway+Monkey+'_lfpXtrialCoh_ITClfp_allsession_'+alignStr+'_'+input+'_'+namstrfit+'.pkl','wb'))

    ITClfp_all = pickle.load(open(ResPathway+Monkey+'_lfpXtrialCoh_ITClfp_allsession_'+alignStr+'_'+input+'_'+namstrfit+'.pkl','rb'))            

    ## permutation test between conditions
    TF_obs_plot = {}
    p_obs_plot = {}
    TF_obs = {}
    max_TF = {}
    n_permutations=300
    trialsincludenanA = np.where(np.any(np.isnan(ITClfp_all['a']),axis=(1,2)))[0]
    trialsincludenanAV = np.where(np.any(np.isnan(ITClfp_all['av']),axis=(1,2)))[0]
    trialsincludenanV = np.where(np.any(np.isnan(ITClfp_all['v']),axis=(1,2)))[0]

    print('ITClfp_all[a] '+str(ITClfp_all['a'].shape)+'nan trials '+str(len(trialsincludenanA))+
          'ITClfp_all[av] '+str(ITClfp_all['av'].shape)+'nan trials '+str(len(trialsincludenanAV))+
          'ITClfp_all[v] '+str(ITClfp_all['v'].shape)+'nan trials '+str(len(trialsincludenanV)))
    
    # TF_obs_plot['A'],TF_obs['A'], max_TF['A']= PermTest1samp(ITClfp_all['a'],n_permutations)
    # TF_obs_plot['AV'],TF_obs['AV'],max_TF['AV'] = PermTest1samp(ITClfp_all['av'],n_permutations)
    # TF_obs_plot['AV-A_diff'],p_obs_plot['AV-A_diff'],TF_obs['AV-A_diff'],max_TF['AV-A_diff'] = PermTest(ITClfp_all['av'],ITClfp_all['a'],n_permutations,0.0001)      
    # ITClfp_all_v_update=np.delete(ITClfp_all['v'],trialsincludenanV,axis=0)
    # TF_obs_plot['A-V_diff'],TF_obs['A-V_diff'],max_TF['A-V_diff'] = PermTest(ITClfp_all['a'],ITClfp_all_v_update,n_permutations,0.001)      
    # TF_obs_plot['AV-V_diff'],TF_obs['AV-V_diff'],max_TF['AV-V_diff'] = PermTest(ITClfp_all['av'],ITClfp_all_v_update,n_permutations,0.001)      

    # print('time spend for permutation test ')
    # print(timedelta(seconds=time.monotonic() - start_time1))              
    # pickle.dump([TF_obs_plot,p_obs_plot,TF_obs,max_TF],open(ResPathway+Monkey+'_lfpXtrialCoh_PermTestFobs2D_'+alignStr+'_'+input+'_'+namstrfit+'.pkl','wb'))
    
    TF_obs_plot,p_obs_plot,TF_obs,max_TF =  pickle.load(open(ResPathway+Monkey+'_lfpXtrialCoh_PermTestFobs2D_'+alignStr+'_'+input+'_'+namstrfit+'.pkl','rb'))    
    #plot sig cluster
    for cmp in ['AV-A_diff']:#['A','AV','A-AV_diff']:
        fig, axes = plt.subplots(1,1,figsize=(6, 6))  
        plot2Dsig(TF_obs_plot[cmp],morletFreq,timeRange_lfp,'',axes,[],x0str) 
        # plot2Dsig(np.sign(p_obs_plot[cmp]) * np.log10(np.abs(p_obs_plot[cmp])),morletFreq,timeRange_lfp,'AV-A',axes,[],x0str)             

        fig.tight_layout()
        plt.savefig(figsavPathway+Monkey+'_'+alignStr+'_lfpXtrialCoh_'+input+'_'+cmp+'_'+namstrfit+'.jpg')
        plt.close()  

print('total time spend: ')
print(timedelta(seconds=time.monotonic() - start_time))              

        # #plot sig cluster
        # for cmp in ['A','AV','diff']:
        #     maxTF = np.max([max_TF_allchan[cc][cmp] for cc in range(len(max_TF_allchan))])
        #     # maxTF = []
        #     for ch in range(len(max_TF_allchan)):            
        #         fig, axes = plt.subplots(1,1,figsize=(6, 6))  
        #         plot2Dsig(TF_obs_plot_allchan[ch][cmp],morletFreq,timeRange_lfp,cmp+str(ch),axes,maxTF,x0str)             
        #         fig.tight_layout()
        #         plt.savefig(figsavPathway_temp+os.path.sep+Monkey+Date_temp+alignStr+'_lfpXtrialCoh_'+input+'_'+cmp+'_ch'+str(ch)+'.jpg')
        #         plt.close()  

        # #plot sig cluster
        # chanAudNeu_temp = chanAudNeuDF[(chanAudNeuDF['sess']==Date_temp) & (chanAudNeuDF['Monkey']==Monkey)].Sigchan.values            
        # chanAudNeu_temp_range = np.arange(chanAudNeu_temp.min(),chanAudNeu_temp.max()+1,1)

        # for cmp in ['A','AV','diff']:
        #     # maxTF = np.max([max_TF_allchan[cc][cmp] for cc in range(len(max_TF_allchan))])
        #     # maxTFch = np.where([max_TF_allchan[cc][cmp] for cc in range(len(max_TF_allchan))]==maxTF)[0][0]
        #     maxTF = np.max([max_TF_allchan[cc][cmp] for cc in chanAudNeu_temp_range])
        #     maxTFch = np.where([max_TF_allchan[cc][cmp] for cc in chanAudNeu_temp_range]==maxTF)[0][0]+chanAudNeu_temp_range[0]
        #     print(Date_temp+'   maxACChan'+str(maxTFch))
        #     fig, axes = plt.subplots(1,1,figsize=(6, 6))  
        #     plot2Dsig(TF_obs_plot_allchan[maxTFch][cmp],morletFreq,timeRange_lfp,cmp+str(maxTFch),axes,maxTF,x0str)             
        #     fig.tight_layout()
        #     plt.savefig(figsavPathway_temp+os.path.sep+Monkey+Date_temp+alignStr+'_lfpXtrialCoh_'+input+'_'+cmp+'_maxACch.jpg')
        #     plt.close()         

        
                     

            
              
                    




