import time
from datetime import timedelta
from eegUtilities import loadTimWindowedLFP
from spikeUtilities import loadBehavMat,decodrespLabel2str
import numpy as np
import pandas as pd
import pickle
import imageio.v3 as iio
import imageio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mne.stats import permutation_cluster_test,combine_adjacency
import h5py
import os

numFreq = 80
morletFreq = np.logspace(np.log10(1),np.log10(150),numFreq)


def getcondMeanSpectrum(xx_data,Behavdata_df_new,filter={'chorus':[1]},input='phase'):
    #xx_data: trialsXchanXfreqXtime
    # filter trials 
    ind = np.arange(0,xx_data.shape[0])
    for ii,(kk,vv) in enumerate(filter.items()): 
        rowind_filtered_temp = np.array(Behavdata_df_new[Behavdata_df_new[kk].isin(vv)].index)
        ind = np.intersect1d(ind,rowind_filtered_temp)
    # get mean power or average angle of all trials in this condition
    xx_temp = xx_data[ind] 
    if input=='power':
        xx_data_updated = np.sum(xx_temp,axis=0)/xx_temp.shape[0] # mean power
    if input =='phase':
        xx_data_updated = np.abs(np.sum(np.exp(xx_temp*1j),axis=0))/xx_temp.shape[0] # trials corrected ITC within condition
    return xx_data_updated, ind # chan X freq X time

def add0Str2xtick(xt,x0str):            
    xt = np.delete(xt,np.where(xt==0)[0])
    xt = np.append(xt,0)
    xtl = xt.tolist()
    xtl = [np.round(xtl[i],1) for i in range(len(xtl))]
    xtl[-1] = x0str
    return xt,xtl

def PermTest(lfpSegA,lfpSegB,n_permutations,F_obs_all,F_obs_plot_all):
    # permutation test for power/iTC in one channel 
    listarray = [lfpSegA,lfpSegB]  
    signs = np.sign(lfpSegA.mean(axis=0)-lfpSegB.mean(axis=0)) 
    if  np.isnan(lfpSegA).any() or np.isnan(lfpSegB).any():
        print('nan included in power/ITC data!! ')
    adjacency = combine_adjacency(lfpSegA.shape[1], lfpSegA.shape[2])                               
    F_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(listarray,n_permutations=n_permutations,adjacency=adjacency,tail=0,n_jobs=24,
                                                                    seed=np.random.RandomState().seed(42),out_type='mask',
                                                                    buffer_size = None)
    # Create stats image with only significant clusters
    F_obs_plot = np.nan * np.ones_like(F_obs)
    for c, p_val in zip(clusters, cluster_p_values):             
        if p_val <= 0.05:
            F_obs_plot[c] = F_obs[c] * signs[c]
    # save sig map of all channels
    F_obs_all = np.concatenate((F_obs_all,np.expand_dims(F_obs,axis=0)),axis=0)
    F_obs_plot_all = np.concatenate((F_obs_plot_all,np.expand_dims(F_obs_plot,axis=0)),axis=0)

    return F_obs_plot,F_obs_all,F_obs_plot_all

def itcBS(lfpSeg,itcsam=100,bstim=500):
    #bootstrap itc from an angle array trialsXtimeXfreq
    itcBSSeg = np.empty((0,lfpSeg.shape[1],lfpSeg.shape[2]))
    for ss in range(bstim):
        samInd_temp = np.random.choice(lfpSeg.shape[0],size=itcsam,replace=False)
        lfpSeg_temp = lfpSeg[samInd_temp]
        itcBSSeg = np.concatenate((itcBSSeg,np.expand_dims(np.abs(np.sum(np.exp(lfpSeg_temp*1j),axis=0))/itcsam,axis=0)),axis=0)
    return itcBSSeg


# MonkeyDate_all = {'Elay':['230420','230627','230705','230711']} #
# MonkeyDate_all = {'Elay':['230420']} #

# Pathway='/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/LFP/'
# ResPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Fitresults/LFP/'
# FigPathway = '/Users/caihuaizhen/Box Sync (huaizhen.cai@pennmedicine.upenn.edu)/Cohen Lab/Projects/MonkeyEEG/Figures/LFP/'

MonkeyDate_all = {'Elay':['230503','230509','230525','230531',
                        '230602','230606','230608','230613','230616','230620',
                        '230717','230718','230719','230726','230728',
                        '230802','230808','230810'],\
                  'Wu':['230807','230809']}

Pathway='/data/by-user/Huaizhen/preprocNeuralMatfiles/EEG+Vprobe/'
ResPathway = '/data/by-user/Huaizhen/LFPcut/2Dpower+phase/'
FigPathway = '/data/by-user/Huaizhen/LFPcut/2Dpower+phase/Figures/'

# timeRange_all = [[-0.5,0.5],[-0.9,0.5],[-0.5,0.5]] #fs 1220.703125
# AllalignStr = ['align2cho','align2coo','align2js']
# x0str_all = ['chOn','cooOn','JSOn']
# alignkeys_all = ['chorusOnsetInd','cooOnsetIndwithVirtual','joystickOnsetInd']
# filterCond_all =[{'chorus':[1]},{'respaLabel':['hit']},{'trialMod':['a','av']}]
# compareCond_all = [[],{'trialMod':['a','av']},{'respLabel':['hit','FA']}]

timeRange_all = [[-0.9,0.5]] #fs 1220.703125
AllalignStr = ['align2coo']
x0str_all = ['cooOn']
alignkeys_all = ['cooOnsetIndwithVirtual']
filterCond_all =[{'respLabel':['hit']}]
compareCond_all = [{'trialMod':['a','av']}]

# timeRange_all = [[-0.2,1.2]] #fs 1220.703125
# AllalignStr = ['align2vid']
# x0str_all = ['vidOn']
# alignkeys_all = ['VidOnsetIndwithVirtual']
# filterCond_all =[{'respLabel':['hit']}]
# compareCond_all = [{'trialMod':['a','av']}]

for Monkey,Date in MonkeyDate_all.items():
    for dd in Date:        
        print('cutting lfp in session: '+dd)   
        start_time0 = time.monotonic()     
        Behavdata_dict,behavefs = loadBehavMat(Monkey,dd,Pathway)# trials X cats dataframe
        Behavdata_df = decodrespLabel2str(pd.DataFrame.from_dict(Behavdata_dict)).copy()        
        LFPfilePathway = Pathway+Monkey+'_'+dd+'_preprocLFPtimeseries.h5'
        for alignkeys,alignStr,timeRange,x0str,filterCond,compareCond in zip(alignkeys_all,AllalignStr,timeRange_all,x0str_all,filterCond_all,compareCond_all): 
            filename = ResPathway+Monkey+dd+'_2Dpowerphase_trial_'+alignStr                          
            Behavdata_df_new= loadTimWindowedLFP(LFPfilePathway,Behavdata_df,behavefs,alignkeys,timeRange,[],'complex',[],filename,baselineCorrect=True) #[CATkey] chan X freq X time [CATkeytrials] #                        
            pickle.dump(Behavdata_df_new,open(ResPathway+Monkey+dd+'_trial_Behavdata_'+alignStr+'.pkl','wb'))            

            # Behavdata_df_new = pickle.load(open(ResPathway+Monkey+dd+'_trial_Behavdata_'+alignStr+'.pkl','rb'))            
            
             #'power' #'phase' # have to plot separatly because of memory
             # power is power, phase is angle between [-pi,pi]
            start_time = time.monotonic()
            for input in ['phase','power']:
                with h5py.File(filename+'_'+input+'.h5','r') as filepower:
                    lfpSeg = filepower.get('LFP'+input+'Seg')[:]
                end_time = time.monotonic()
                print('time spend to load .h5 file')
                print(timedelta(seconds=end_time - start_time))
                # # # prepare conditional array for comparison in each alignment
                gif_path = ResPathway+Monkey+dd+'_2Dpowerphase_CATsum'+alignStr+'_'+input+'.gif'            
                xx_data_cond_diff = {}  
                catindex = {}
                xx_data_cond_dict = {}
                if alignStr == 'align2cho':                                       
                    xx_data_cond_diff[input],ind = getcondMeanSpectrum(lfpSeg,Behavdata_df_new,filter=filterCond,input=input)
                    lfpSeg = lfpSeg[ind]
                    # generate comparison arrays
                    time0 = int(lfpSeg.shape[3]/2)
                    time1 = time0*2
                    lfpSegA = np.concatenate((np.zeros((lfpSeg.shape[0],lfpSeg.shape[1],lfpSeg.shape[2],lfpSeg.shape[3]-time0)),
                                                lfpSeg[:,:,:,0:time0]),axis=3)
                    lfpSegB = np.concatenate((np.zeros((lfpSeg.shape[0],lfpSeg.shape[1],lfpSeg.shape[2],lfpSeg.shape[3]-time0)),
                                                lfpSeg[:,:,:,time0:time1]),axis=3) 
                                    
                if alignStr == 'align2coo' or alignStr == 'align2vid':                  
                    for compCon in next(iter(compareCond.values())):
                        xx_data_cond_dict[compCon], catindex[compCon] = getcondMeanSpectrum(lfpSeg,Behavdata_df_new,filter={**filterCond,**{list(compareCond.keys())[0]:[compCon]}},input=input)
                    xx_data_cond_diff[input] = xx_data_cond_dict['a']-xx_data_cond_dict['av']
                    lfpSegA = lfpSeg[catindex['a']]
                    lfpSegB = lfpSeg[catindex['av']]

                if alignStr == 'align2js': 
                    for compCon in next(iter(compareCond.values())):
                        xx_data_cond_dict[compCon], catindex[compCon] = getcondMeanSpectrum(lfpSeg,Behavdata_df_new,filter={**filterCond,**{list(compareCond.keys())[0]:[compCon]}},input=input)
                    xx_data_cond_diff[input] = xx_data_cond_dict['hit']-xx_data_cond_dict['FA']
                    lfpSegA = lfpSeg[catindex['hit']]# trialXchanXFreqXtime
                    lfpSegB = lfpSeg[catindex['FA']]                           

                # delete the large variable to save memory
                del lfpSeg
                # stats for tf measurements
                F_obs_all = np.empty((0,lfpSegA.shape[2],lfpSegA.shape[3]))
                F_obs_plot_all = np.empty((0,lfpSegA.shape[2],lfpSegA.shape[3]))
                for chan in range(24):  
                    if input=='power':
                        lfpSegA_temp = lfpSegA[:,chan,:,:]
                        lfpSegB_temp = lfpSegB[:,chan,:,:]                       
                    if input =='phase':
                        # BOOTSTRAP ITC in each condition
                        lfpSegA_temp = itcBS(lfpSegA[:,chan,:,:],itcsam=200,bstim=400)
                        lfpSegB_temp = itcBS(lfpSegB[:,chan,:,:],itcsam=200,bstim=400)                     
                    start_time = time.monotonic()
                    n_permutations=200
                    F_obs_plot,F_obs_all,F_obs_plot_all = PermTest(lfpSegA_temp,lfpSegB_temp,n_permutations,F_obs_all,F_obs_plot_all)      
                    end_time = time.monotonic()
                    print('time spend for permutation test of channel'+str(chan))
                    print(timedelta(seconds=end_time - start_time))
                    pickle.dump([F_obs_all,F_obs_plot_all,xx_data_cond_diff],open(ResPathway+Monkey+dd+'_PermTestFobs2D_'+alignStr+'_'+input+'.pkl','wb'))

                    # F_obs_all,F_obs_plot_all,xx_data_cond_diff = pickle.load(open(ResPathway+Monkey+dd+'_PermTestFobs2D_'+alignStr+'_'+input+'.pkl','rb')) 
                    F_obs = F_obs_all[chan,:,:]
                    F_obs_plot = F_obs_plot_all[chan,:,:]

                    fig, axes = plt.subplots(1,2,figsize=(12, 6)) 
                    #plot raw contrast diff
                    vminval = xx_data_cond_diff[input].min()
                    vmaxval = xx_data_cond_diff[input].max()                    
                    c=axes[0].imshow(xx_data_cond_diff[input][chan,:,:],
                                    cmap='viridis',vmin=vminval,vmax=vmaxval,
                                        extent=[timeRange[0], timeRange[-1], morletFreq[0],morletFreq[-1]],
                                            origin='lower',aspect='auto')
                    plt.colorbar(c,ax=axes[0],location='bottom',orientation='horizontal')
                    axes[0].set_title('chan'+str(chan)+' '+input,fontsize=8)
                    # Set the x-axis and y-axis tick labels                   
                    axes[0].set_xlabel('time (s)')
                    axes[0].set_ylabel('Frequency (Hz)')
                    xt = axes[0].get_xticks()  
                    xt,xtl = add0Str2xtick(xt,x0str)
                    axes[0].set_xticks(xt)
                    axes[0].set_xticklabels(xtl)
                    axes[0].set_xlim(-0.9,0.5)
                    y_values = morletFreq[[0,20,40,60,numFreq-1]]
                    axes[0].yaxis.set_major_locator(ticker.LinearLocator(numticks=len(y_values)))
                    axes[0].set_yticks(axes[0].get_yticks())
                    axes[0].set_yticklabels([f'{label:.2f}' for label in np.round(y_values,decimals=1)],fontsize=8)

                    #plot sig contrast diff
                    max_F = np.nanmax(abs(F_obs_plot))
                    c = axes[1].imshow(F_obs_plot,
                        extent=[timeRange[0], timeRange[-1], morletFreq[0],morletFreq[-1]],
                        aspect="auto",
                        origin="lower",
                        cmap="RdBu_r",
                        vmin=-max_F,
                        vmax=max_F,alpha=1)
                    plt.colorbar(c,ax=axes[1],location='bottom',orientation='horizontal')
                    axes[1].set_title('chan'+str(chan)+' '+input,fontsize=8)
                    # Set the x-axis and y-axis tick labels
                    axes[1].set_xlabel('time (s)')
                    axes[1].set_ylabel('Frequency (Hz)')                    
                    xt = axes[1].get_xticks()                     
                    xt,xtl = add0Str2xtick(xt,x0str)
                    axes[1].set_xticks(xt)
                    axes[1].set_xticklabels(xtl)
                    axes[1].set_xlim(-0.9,0.5)
                    y_values = morletFreq[[0,20,40,60,numFreq-1]]
                    axes[1].yaxis.set_major_locator(ticker.LinearLocator(numticks=len(y_values)))
                    axes[1].set_yticks(axes[1].get_yticks())
                    axes[1].set_yticklabels([f'{label:.2f}' for label in np.round(y_values,decimals=1)],fontsize=8)
                    fig.tight_layout()
                    plt.savefig(FigPathway+Monkey+dd+alignStr+'_'+input+f"{chan}.jpg")
                    plt.close()
                # # remove the huge trial by trial .h5 file
                # os.remove(filename+'_'+input+'.h5')
                frames = np.stack([iio.imread(FigPathway+Monkey+dd+alignStr+'_'+input+f"{cc}.jpg") for cc in range(24)], axis=0)
                imageio.mimsave(gif_path, frames,format='GIF',duration=300)# duration in ms/frame
        end_time = time.monotonic()
        print('time spend for session'+dd+' '+alignStr)
        print(timedelta(seconds=end_time - start_time0))








